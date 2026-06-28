from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _create_mask(path: Path, value: float = 1.0) -> Path:
    """Create a simple 3D NIfTI mask used across multiple tests."""
    data = np.full((2, 2, 2), value, dtype=np.float32)
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, path)
    return path


def _create_dicom_file(
    path: Path,
    *,
    slice_location: float,
    echo_time: float,
    pixel_value: int = 1,
) -> None:
    """Create a tiny valid DICOM file for testing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

    dataset.PatientName = "bMRI Test"
    dataset.PatientID = "TEST-001"
    dataset.Modality = "MR"
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = generate_uid()
    dataset.InstanceNumber = 1
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.SliceLocation = float(slice_location)
    dataset.EchoTime = float(echo_time)
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1.0
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False

    pixels = np.full((2, 2), pixel_value, dtype=np.uint16)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(str(path), write_like_original=False)


def create_dicom_series(
    folder: Path, slices: int = 2, echoes: Iterable[float] | None = None
) -> None:
    """Generate a small DICOM series with multiple echoes."""
    folder.mkdir(parents=True, exist_ok=True)

    if echoes is None:
        echoes = (10.0, 20.0)

    for slice_idx in range(slices):
        slice_location = float(slice_idx)
        for echo_idx, echo_time in enumerate(echoes):
            filename = folder / f"slice{slice_idx:02d}_echo{echo_idx:02d}.dcm"
            _create_dicom_file(
                filename,
                slice_location=slice_location,
                echo_time=echo_time,
                pixel_value=echo_idx + 1,
            )


@pytest.fixture
def mask_file(tmp_path: Path) -> Path:
    """Fixture providing a valid mask file."""
    return _create_mask(tmp_path / "mask.nii.gz")


@pytest.fixture
def empty_mask_file(tmp_path: Path) -> Path:
    """Fixture providing an empty mask (all zeros)."""
    return _create_mask(tmp_path / "empty_mask.nii.gz", value=0.0)


@pytest.fixture
def dicom_folder(tmp_path: Path) -> Path:
    """Fixture providing a folder with placeholder DICOM files."""
    folder = tmp_path / "dicom"
    folder.mkdir()
    # Validators only check for *.dcm, content is irrelevant
    (folder / "image001.dcm").write_bytes(b"DICOM")
    return folder


@pytest.fixture
def dicom_series_dir(tmp_path: Path) -> Path:
    """Fixture providing a folder with real tiny DICOM files."""
    folder = tmp_path / "dicom_series"
    create_dicom_series(folder)
    return folder
