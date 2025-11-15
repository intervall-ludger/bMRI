from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bmri.cli.commands.view import prepare_t2star_viewer_inputs


def create_results_dir(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name in ("dicom.nii.gz", "params.nii.gz"):
        (results / name).write_text("fake nifti content")
    np.savetxt(results / "acquisition_times.txt", np.array([0.0, 10.0, 40.0]))
    return results


def test_prepare_t2star_viewer_inputs_defaults(tmp_path: Path) -> None:
    results = create_results_dir(tmp_path)
    dicom, params, times, parameter_index = prepare_t2star_viewer_inputs(results)
    assert dicom.name == "dicom.nii.gz"
    assert params.name == "params.nii.gz"
    assert np.allclose(times, [0.0, 10.0, 40.0])
    assert parameter_index == 1  # t2_t2star


def test_prepare_t2star_viewer_inputs_aliases(tmp_path: Path) -> None:
    results = create_results_dir(tmp_path)
    _, _, _, idx = prepare_t2star_viewer_inputs(results, parameter="s0")
    assert idx == 0
    _, _, _, idx = prepare_t2star_viewer_inputs(results, parameter="t2*")
    assert idx == 1


def test_prepare_t2star_viewer_inputs_custom_paths(tmp_path: Path) -> None:
    results = create_results_dir(tmp_path)
    dicom_path = tmp_path / "custom_dicom.nii.gz"
    dicom_path.write_text("custom")
    params_path = tmp_path / "custom_params.nii.gz"
    params_path.write_text("custom")
    times_path = tmp_path / "custom_times.txt"
    np.savetxt(times_path, np.array([5.0]))

    dicom, params, times, _ = prepare_t2star_viewer_inputs(
        results,
        parameter="t2_t2star",
        dicom_file=dicom_path,
        params_file=params_path,
        times_file=times_path,
    )

    assert dicom == dicom_path
    assert params == params_path
    assert np.allclose(times, [5.0])


def test_prepare_t2star_viewer_missing_files(tmp_path: Path) -> None:
    results = tmp_path / "empty"
    results.mkdir()

    with pytest.raises(FileNotFoundError):
        prepare_t2star_viewer_inputs(results)

    valid = create_results_dir(tmp_path / "ready")
    with pytest.raises(ValueError):
        prepare_t2star_viewer_inputs(valid, parameter="unknown")
