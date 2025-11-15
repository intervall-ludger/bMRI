"""Input validation functions for bMRI.

This module provides validators for all user inputs including file paths,
DICOM folders, masks, and configuration parameters.
"""

from enum import Enum
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom

from bmri.exceptions import DICOMStructureError, MaskError, ValidationError
from bmri.logger import get_logger
from bmri.types import BoundaryTuple

logger = get_logger(__name__)


class DICOMLayout(Enum):
    """Supported DICOM folder layouts."""

    FLAT = "flat"  # All DICOM files in single folder
    ECHO_FOLDERS = "echo_folders"  # Subfolders per echo/timepoint
    SLICE_FOLDERS = "slice_folders"  # Subfolders per slice


def validate_dicom_folder(folder: Path) -> tuple[Path, DICOMLayout]:
    """Validate DICOM folder and detect its layout.

    Args:
        folder: Path to DICOM folder

    Returns:
        Tuple of (validated path, detected layout)

    Raises:
        ValidationError: If folder doesn't exist
        DICOMStructureError: If no valid DICOM files found

    Example:
        >>> folder, layout = validate_dicom_folder(Path("/data/patient01"))
        >>> print(f"Found {layout.value} layout")
    """
    if not folder.exists():
        raise ValidationError(
            f"DICOM folder not found: {folder}",
            details="Please check the path and try again.",
        )

    if not folder.is_dir():
        raise ValidationError(
            f"Path is not a directory: {folder}",
            details="DICOM folder must be a directory containing .dcm files.",
        )

    # Check for flat layout (all .dcm files in root)
    dcm_files = list(folder.glob("*.dcm"))
    if dcm_files:
        logger.debug(f"Detected FLAT layout with {len(dcm_files)} DICOM files")
        return folder, DICOMLayout.FLAT

    # Check for subfolder layouts
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    if not subfolders:
        raise DICOMStructureError(
            f"No DICOM files found in {folder}",
            details="Expected either:\n"
            "  - DICOM files (*.dcm) in the folder, or\n"
            "  - Subfolders containing DICOM files",
        )

    # Check if subfolders contain DICOM files
    subfolder_dcm_counts = {}
    for subfolder in subfolders:
        dcm_in_subfolder = list(subfolder.glob("*.dcm"))
        if dcm_in_subfolder:
            subfolder_dcm_counts[subfolder.name] = len(dcm_in_subfolder)

    if not subfolder_dcm_counts:
        raise DICOMStructureError(
            f"No DICOM files found in subfolders of {folder}",
            details=f"Checked {len(subfolders)} subfolders, none contained .dcm files",
        )

    # Detect layout type based on subfolder structure
    # Echo folders typically have similar file counts
    counts = list(subfolder_dcm_counts.values())
    if len(set(counts)) == 1:
        # All subfolders have same count -> likely echo folders
        layout = DICOMLayout.ECHO_FOLDERS
        logger.debug(
            f"Detected ECHO_FOLDERS layout: {len(subfolder_dcm_counts)} echoes "
            f"with {counts[0]} files each"
        )
    else:
        # Different counts -> likely slice folders
        layout = DICOMLayout.SLICE_FOLDERS
        logger.debug(
            f"Detected SLICE_FOLDERS layout: {len(subfolder_dcm_counts)} slices "
            f"with varying file counts"
        )

    return folder, layout


def validate_mask_file(mask_path: Path, expected_shape: tuple[int, ...] | None = None) -> Path:
    """Validate NIfTI mask file.

    Args:
        mask_path: Path to mask file (.nii or .nii.gz)
        expected_shape: Optional expected shape to validate against

    Returns:
        Validated mask path

    Raises:
        ValidationError: If mask file doesn't exist
        MaskError: If mask file is invalid or shape doesn't match

    Example:
        >>> mask = validate_mask_file(Path("mask.nii.gz"), expected_shape=(128, 128, 32))
    """
    if not mask_path.exists():
        raise ValidationError(
            f"Mask file not found: {mask_path}",
            details="Please check the path and ensure the file exists.",
        )

    if mask_path.suffix not in [".nii", ".gz"]:
        raise ValidationError(
            f"Invalid mask file format: {mask_path.suffix}",
            details="Mask must be a NIfTI file (.nii or .nii.gz)",
        )

    try:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
    except Exception as e:
        raise MaskError(
            f"Failed to load mask file: {mask_path}",
            details=f"NIfTI loading error: {str(e)}",
        ) from e

    # Validate shape if provided
    if expected_shape is not None:
        if mask_data.shape != expected_shape:
            raise MaskError(
                f"Mask shape mismatch: expected {expected_shape}, got {mask_data.shape}",
                details=f"Mask file: {mask_path}",
            )

    # Check if mask contains any non-zero values
    if not np.any(mask_data > 0):
        raise MaskError(
            f"Mask is empty (all zeros): {mask_path}",
            details="Mask must contain at least one non-zero voxel",
        )

    num_masked = np.sum(mask_data > 0)
    logger.debug(f"Validated mask: {mask_data.shape}, {num_masked} masked voxels")

    return mask_path


def validate_boundary(
    boundary: tuple[float, ...] | tuple[tuple[float, ...], tuple[float, ...]],
    num_params: int,
) -> BoundaryTuple:
    """Validate and normalize fitting boundaries.

    Args:
        boundary: Fitting boundaries as ((lower...), (upper...)) or shorthand
        num_params: Expected number of parameters

    Returns:
        Validated boundary tuple in canonical form ((lower...), (upper...))

    Raises:
        ValidationError: If boundary format is invalid

    Example:
        >>> bounds = validate_boundary(([0, 0, -100], [1000, 100, 100]), num_params=3)
        >>> print(bounds)
        ((0.0, 0.0, -100.0), (1000.0, 100.0, 100.0))
    """
    if len(boundary) != 2:
        raise ValidationError(
            f"Boundary must be a tuple of (lower, upper), got length {len(boundary)}",
            details="Example: boundary=([0, 0, -100], [1000, 100, 100])",
        )

    lower, upper = boundary

    # Ensure tuples
    if not isinstance(lower, (tuple, list)):
        raise ValidationError(
            f"Lower boundary must be a tuple/list, got {type(lower).__name__}",
        )
    if not isinstance(upper, (tuple, list)):
        raise ValidationError(
            f"Upper boundary must be a tuple/list, got {type(upper).__name__}",
        )

    lower = tuple(float(x) for x in lower)
    upper = tuple(float(x) for x in upper)

    # Validate parameter count
    if len(lower) != num_params:
        raise ValidationError(
            f"Lower boundary has {len(lower)} values, expected {num_params}",
            details=f"Fitting function requires {num_params} parameters",
        )
    if len(upper) != num_params:
        raise ValidationError(
            f"Upper boundary has {len(upper)} values, expected {num_params}",
            details=f"Fitting function requires {num_params} parameters",
        )

    # Validate that lower < upper
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo >= hi:
            raise ValidationError(
                f"Invalid boundary for parameter {i}: lower ({lo}) >= upper ({hi})",
                details="Lower bounds must be strictly less than upper bounds",
            )

    return (lower, upper)


def validate_output_directory(output_dir: Path, create: bool = True) -> Path:
    """Validate output directory.

    Args:
        output_dir: Path to output directory
        create: Whether to create directory if it doesn't exist

    Returns:
        Validated output directory path

    Raises:
        ValidationError: If directory is invalid or cannot be created

    Example:
        >>> output = validate_output_directory(Path("./results"), create=True)
    """
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValidationError(
                f"Output path exists but is not a directory: {output_dir}",
            )
        logger.debug(f"Using existing output directory: {output_dir}")
        return output_dir

    if create:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except PermissionError as e:
            raise ValidationError(
                f"Permission denied creating output directory: {output_dir}",
                details=str(e),
            ) from e
        except Exception as e:
            raise ValidationError(
                f"Failed to create output directory: {output_dir}",
                details=str(e),
            ) from e
    else:
        raise ValidationError(
            f"Output directory does not exist: {output_dir}",
            details="Use create=True to create it automatically",
        )

    return output_dir


def validate_tsl_times(tsl: list[float] | str) -> list[float]:
    """Validate T1rho spin-lock times.

    Args:
        tsl: TSL times as list or comma-separated string

    Returns:
        Validated TSL times as sorted list

    Raises:
        ValidationError: If TSL times are invalid

    Example:
        >>> tsl = validate_tsl_times("0,10,40,70")
        >>> print(tsl)
        [0.0, 10.0, 40.0, 70.0]
    """
    if isinstance(tsl, str):
        try:
            tsl = [float(x.strip()) for x in tsl.split(",")]
        except ValueError as e:
            raise ValidationError(
                f"Invalid TSL format: {tsl}",
                details="TSL must be comma-separated numbers, e.g., '0,10,40,70'",
            ) from e

    if not tsl:
        raise ValidationError(
            "TSL times cannot be empty",
            details="Provide at least 2 timepoints for T1rho fitting",
        )

    if len(tsl) < 2:
        raise ValidationError(
            f"TSL requires at least 2 timepoints, got {len(tsl)}",
            details="Example: tsl=[0, 10, 40, 70]",
        )

    # Check for duplicates
    if len(set(tsl)) != len(tsl):
        raise ValidationError(
            "TSL times contain duplicates",
            details=f"Unique times: {sorted(set(tsl))}",
        )

    # Ensure non-negative
    if any(t < 0 for t in tsl):
        raise ValidationError(
            "TSL times must be non-negative",
            details=f"Got: {tsl}",
        )

    return sorted(tsl)
