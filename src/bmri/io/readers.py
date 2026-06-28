"""DICOM and NIfTI file readers with robust error handling.

This module provides functions for reading medical imaging data with proper
validation, logging, and error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
from natsort import natsorted

from bmri.exceptions import DICOMReadError, DICOMStructureError, MaskError
from bmri.logger import get_logger
from bmri.types import FloatArray, FloatArray3D, IntArray3D

logger = get_logger(__name__)


def get_dcm_list(folder: Path) -> list[Path]:
    """Get naturally sorted list of DICOM files from directory.

    Args:
        folder: Directory containing DICOM files

    Returns:
        Sorted list of DICOM file paths

    Raises:
        DICOMStructureError: If folder doesn't exist or contains no DICOM files

    Example:
        >>> dcm_files = get_dcm_list(Path("/data/patient01/t2star"))
        >>> print(f"Found {len(dcm_files)} DICOM files")
    """
    if not folder.exists():
        raise DICOMStructureError(
            f"DICOM folder not found: {folder}",
            details="Please check the path and try again.",
        )

    if not folder.is_dir():
        raise DICOMStructureError(
            f"Path is not a directory: {folder}",
            details="Expected a folder containing DICOM files",
        )

    dcm_files = natsorted(folder.glob("*.dcm"))

    if not dcm_files:
        logger.warning(f"No DICOM files found in {folder}")
        raise DICOMStructureError(
            f"No DICOM files (*.dcm) found in {folder}",
            details="Check if files have .dcm extension or are in subfolders",
        )

    logger.debug(f"Found {len(dcm_files)} DICOM files in {folder}")
    return dcm_files


def split_dcm_list(dcm_list: list[Path]) -> list[list[Path]]:
    """Split DICOM files by slice location.

    Organizes multi-echo DICOM files into echo-wise lists based on SliceLocation
    metadata. This is used for 3D multi-echo sequences where each echo time has
    multiple slices.

    Args:
        dcm_list: List of DICOM file paths

    Returns:
        List of echo lists, where each echo list contains paths for all slices

    Raises:
        DICOMReadError: If DICOM files cannot be read
        DICOMStructureError: If slice organization is inconsistent

    Example:
        >>> dcm_files = get_dcm_list(Path("/data/t2star"))
        >>> echo_lists = split_dcm_list(dcm_files)
        >>> print(f"Found {len(echo_lists)} echoes")
    """
    locations: dict[float, list[Path]] = {}
    failed_files: list[Path] = []

    for dcm_file in dcm_list:
        try:
            dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError as e:
            logger.warning(f"Invalid DICOM file {dcm_file.name}: {e}")
            failed_files.append(dcm_file)
            continue
        except Exception as e:
            logger.error(f"Failed to read {dcm_file.name}: {e}")
            failed_files.append(dcm_file)
            continue

        # Extract slice location
        try:
            slice_location = float(dcm.SliceLocation)
        except AttributeError:
            logger.warning(f"No SliceLocation in {dcm_file.name}, skipping")
            failed_files.append(dcm_file)
            continue

        if slice_location in locations:
            locations[slice_location].append(dcm_file)
        else:
            locations[slice_location] = [dcm_file]

    if not locations:
        raise DICOMStructureError(
            "No valid DICOM files with SliceLocation found",
            details=f"Failed to read {len(failed_files)}/{len(dcm_list)} files",
        )

    if failed_files:
        logger.info(f"Skipped {len(failed_files)}/{len(dcm_list)} invalid files")

    # Validate and clean locations
    locations = _validate_locations(locations)

    # Reorganize: slice-wise → echo-wise
    sorted_slices = natsorted(locations.keys())
    num_echoes = len(locations[sorted_slices[0]])

    echo_lists: list[list[Path]] = [[] for _ in range(num_echoes)]

    for slice_loc in sorted_slices:
        slice_files = locations[slice_loc]
        for echo_idx, dcm_file in enumerate(slice_files):
            echo_lists[echo_idx].append(dcm_file)

    logger.debug(
        f"Organized {len(dcm_list)} files into {num_echoes} echoes "
        f"with {len(sorted_slices)} slices each"
    )

    return echo_lists


def _validate_locations(locations: dict[float, list[Path]]) -> dict[float, list[Path]]:
    """Validate and clean slice location dictionary.

    Ensures all slices have the same number of echo files. If two slices
    differ, attempts to merge them (legacy behavior from original code).

    Args:
        locations: Dictionary mapping slice locations to file lists

    Returns:
        Validated and potentially cleaned locations dictionary

    Raises:
        DICOMStructureError: If slice counts are inconsistent
    """
    slice_counts = {loc: len(files) for loc, files in locations.items()}
    unique_counts = set(slice_counts.values())

    if len(unique_counts) == 1:
        # All slices have same number of echoes - perfect!
        return locations

    # Find slices with non-median count
    median_count = int(np.median(list(slice_counts.values())))
    outliers = [loc for loc, count in slice_counts.items() if count != median_count]

    if len(outliers) == 2:
        # Legacy behavior: merge two outlier slices
        loc_a, loc_b = outliers[0], outliers[1]
        logger.warning(
            f"Merging slices at {loc_a} and {loc_b} "
            f"({slice_counts[loc_a]} + {slice_counts[loc_b]} files)"
        )
        locations[loc_a].extend(locations[loc_b])
        del locations[loc_b]
        return locations

    # Too many inconsistencies
    raise DICOMStructureError(
        f"Inconsistent echo counts across slices: {slice_counts}",
        details=f"Expected {median_count} echoes per slice, but found {unique_counts}",
    )


def get_dcm_array(dcm_files: list[Path], apply_rescale: bool = True) -> FloatArray3D | IntArray3D:
    """Load DICOM files into numpy array.

    Reads pixel data from DICOM files and optionally applies RescaleSlope and
    RescaleIntercept transformations to convert to physical units.

    Args:
        dcm_files: List of DICOM file paths (one per slice)
        apply_rescale: Apply RescaleSlope/Intercept if available

    Returns:
        3D numpy array with shape (num_slices, height, width)

    Raises:
        DICOMReadError: If files cannot be read or contain no valid data

    Example:
        >>> files = get_dcm_list(Path("/data/echo1"))
        >>> array = get_dcm_array(files)
        >>> print(array.shape)  # (64, 256, 256)
    """
    arrays: list[FloatArray] = []
    failed_files: list[Path] = []

    for dcm_file in dcm_files:
        try:
            dcm = pydicom.dcmread(dcm_file)

            # Get pixel array
            pixel_array = dcm.pixel_array.astype(np.float64)

            # Apply rescaling if requested and available
            if apply_rescale:
                try:
                    slope = float(dcm.RescaleSlope)
                    intercept = float(dcm.RescaleIntercept)
                    pixel_array = pixel_array * slope + intercept
                    logger.debug(f"Applied rescale: slope={slope}, intercept={intercept}")
                except AttributeError:
                    logger.debug(f"No RescaleSlope/Intercept in {dcm_file.name}")

            arrays.append(pixel_array)

        except pydicom.errors.InvalidDicomError as e:
            logger.error(f"Invalid DICOM file {dcm_file.name}: {e}")
            failed_files.append(dcm_file)
        except Exception as e:
            logger.error(f"Failed to read pixel data from {dcm_file.name}: {e}")
            failed_files.append(dcm_file)

    if not arrays:
        raise DICOMReadError(
            "No valid DICOM pixel data could be read",
            details=f"Failed to read all {len(dcm_files)} files",
        )

    if failed_files:
        logger.warning(
            f"Successfully read {len(arrays)}/{len(dcm_files)} files ({len(failed_files)} failed)"
        )

    return np.array(arrays)


@dataclass
class NIfTIMask:
    """NIfTI mask data with metadata.

    Attributes:
        array: 3D mask array with labeled regions (0=background, >0=ROIs)
        affine: 4x4 affine transformation matrix
        header: NIfTI header with metadata
        file_path: Original file path (optional)
    """

    array: FloatArray3D
    affine: FloatArray
    header: Any
    file_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate mask data."""
        if self.array.ndim not in (2, 3):
            raise ValueError(f"Mask must be 2D or 3D, got {self.array.ndim}D: {self.array.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Get mask shape."""
        return self.array.shape

    @property
    def num_masked_voxels(self) -> int:
        """Get number of non-zero voxels."""
        return int(np.sum(self.array > 0))

    @property
    def num_rois(self) -> int:
        """Get number of unique ROIs (excluding background)."""
        unique_values = np.unique(self.array)
        return int(np.sum(unique_values > 0))


def load_nii(file_path: Path, validate: bool = True) -> NIfTIMask:
    """Load NIfTI mask file.

    Args:
        file_path: Path to NIfTI file (.nii or .nii.gz)
        validate: Perform validation checks on loaded data

    Returns:
        NIfTIMask object with array, affine, and header

    Raises:
        MaskError: If file cannot be loaded or is invalid

    Example:
        >>> mask = load_nii(Path("mask.nii.gz"))
        >>> print(f"Mask shape: {mask.shape}")
        >>> print(f"ROIs: {mask.num_rois}")
    """
    if not file_path.exists():
        raise MaskError(
            f"NIfTI file not found: {file_path}",
            details="Please check the path and try again.",
        )

    try:
        nii_img = nib.load(file_path)
        array = nii_img.get_fdata()
        affine = nii_img.affine
        header = nii_img.header
    except Exception as e:
        raise MaskError(
            f"Failed to load NIfTI file: {file_path}",
            details=f"nibabel error: {str(e)}",
        ) from e

    # Create mask object
    mask = NIfTIMask(array=array, affine=affine, header=header, file_path=file_path)

    # Validation
    if validate:
        if mask.num_masked_voxels == 0:
            raise MaskError(
                f"Mask is empty (all zeros): {file_path}",
                details="Mask must contain at least one non-zero voxel",
            )

        logger.info(
            f"Loaded mask: {mask.shape}, {mask.num_masked_voxels} voxels, {mask.num_rois} ROIs"
        )

    return mask


# Legacy compatibility: import old Mask class as alias
Mask = NIfTIMask
