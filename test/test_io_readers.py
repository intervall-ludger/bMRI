from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bmri.exceptions import DICOMReadError, DICOMStructureError, MaskError
from bmri.io import readers


def test_get_dcm_list_and_split(dicom_series_dir) -> None:
    dcm_files = readers.get_dcm_list(dicom_series_dir)
    assert len(dcm_files) == 4  # 2 slices × 2 echoes

    echo_lists = readers.split_dcm_list(dcm_files)
    assert len(echo_lists) == 2
    assert all(len(echo) == 2 for echo in echo_lists)


def test_validate_locations_merges_outliers(tmp_path) -> None:
    locs = {
        0.0: [tmp_path / "a_echo1", tmp_path / "a_echo2"],
        1.0: [tmp_path / "b_echo1"],
        2.0: [tmp_path / "c_echo1", tmp_path / "c_echo2", tmp_path / "c_echo3"],
    }
    merged = readers._validate_locations(locs)
    assert len(merged) == 2


def test_get_dcm_array_applies_rescale(dicom_series_dir) -> None:
    dcm_files = readers.get_dcm_list(dicom_series_dir)
    echo_lists = readers.split_dcm_list(dcm_files)
    array = readers.get_dcm_array(echo_lists[0], apply_rescale=True)
    assert array.shape == (2, 2, 2)
    assert np.allclose(array, 1.0)


def test_get_dcm_list_errors(tmp_path) -> None:
    with pytest.raises(DICOMStructureError):
        readers.get_dcm_list(tmp_path / "missing")


def test_get_dcm_array_errors(tmp_path) -> None:
    bad_dcm = tmp_path / "invalid.dcm"
    bad_dcm.write_text("not a dicom")
    with pytest.raises(DICOMReadError):
        readers.get_dcm_array([bad_dcm])


def test_load_nii(mask_file, empty_mask_file) -> None:
    mask = readers.load_nii(mask_file)
    assert mask.num_masked_voxels > 0
    assert mask.num_rois == 1

    with pytest.raises(MaskError):
        readers.load_nii(empty_mask_file)

    with pytest.raises(MaskError):
        readers.load_nii(Path("missing_mask.nii.gz"))
