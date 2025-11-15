from __future__ import annotations

from pathlib import Path

import pytest

from bmri.exceptions import MaskError, ValidationError
from bmri.validators import (
    DICOMLayout,
    validate_boundary,
    validate_dicom_folder,
    validate_mask_file,
    validate_output_directory,
    validate_tsl_times,
)


def test_validate_dicom_folder_flat_layout(dicom_folder) -> None:
    folder, layout = validate_dicom_folder(dicom_folder)
    assert folder == dicom_folder
    assert layout == DICOMLayout.FLAT


def test_validate_dicom_folder_echo_and_slice_layouts(tmp_path: Path) -> None:
    echo_root = tmp_path / "echo_layout"
    echo_a = echo_root / "echo_a"
    echo_b = echo_root / "echo_b"
    echo_a.mkdir(parents=True)
    echo_b.mkdir(parents=True)
    (echo_a / "a.dcm").write_bytes(b"1")
    (echo_b / "b.dcm").write_bytes(b"1")

    _, layout = validate_dicom_folder(echo_root)
    assert layout == DICOMLayout.ECHO_FOLDERS

    slice_root = tmp_path / "slice_layout"
    slice_x = slice_root / "slice_x"
    slice_y = slice_root / "slice_y"
    slice_x.mkdir(parents=True)
    slice_y.mkdir(parents=True)
    (slice_x / "a.dcm").write_bytes(b"1")
    (slice_y / "b.dcm").write_bytes(b"1")
    (slice_y / "c.dcm").write_bytes(b"1")

    _, layout = validate_dicom_folder(slice_root)
    assert layout == DICOMLayout.SLICE_FOLDERS


def test_validate_mask_file_variants(empty_mask_file, mask_file, tmp_path: Path) -> None:
    assert validate_mask_file(mask_file) == mask_file
    with pytest.raises(MaskError):
        validate_mask_file(empty_mask_file)

    with pytest.raises(MaskError):
        validate_mask_file(mask_file, expected_shape=(10, 10, 10))

    bad_extension = tmp_path / "mask.txt"
    bad_extension.write_text("not a nifti")
    with pytest.raises(ValidationError):
        validate_mask_file(bad_extension)


def test_validate_boundary_success_and_errors() -> None:
    bounds = validate_boundary(((0, 0, -1), (1, 1, 1)), num_params=3)
    assert bounds[0][0] == 0

    with pytest.raises(ValidationError):
        validate_boundary(((0, 0), (1, 1)), num_params=3)

    with pytest.raises(ValidationError):
        validate_boundary(((0, 0, 0), (0, 1, 1)), num_params=3)


def test_validate_output_directory_variants(tmp_path: Path) -> None:
    target = tmp_path / "results"
    validated = validate_output_directory(target, create=True)
    assert validated == target

    file_path = tmp_path / "file"
    file_path.write_text("content")
    with pytest.raises(ValidationError):
        validate_output_directory(file_path)

    missing = tmp_path / "missing_dir"
    with pytest.raises(ValidationError):
        validate_output_directory(missing, create=False)


def test_validate_tsl_times_variants() -> None:
    assert validate_tsl_times("0, 10, 40") == [0.0, 10.0, 40.0]

    with pytest.raises(ValidationError):
        validate_tsl_times([10])

    with pytest.raises(ValidationError):
        validate_tsl_times("0,0,5")

    with pytest.raises(ValidationError):
        validate_tsl_times([-5, 10])
