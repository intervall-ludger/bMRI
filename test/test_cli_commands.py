from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

legacy_t2star = importlib.import_module("src.Fitting.T2_T2star")
legacy_t1rho = importlib.import_module("src.Fitting.T1rho_T2prep")
from bmri.cli.commands.fit import (
    _copy_results_and_display,
    _print_csv_results,
    estimate_boundaries_from_data,
    fit_t1rho,
    fit_t2,
    fit_t2star,
    parse_boundary_string,
)
from bmri.cli.main import app

FAKE_NIFTI_FILES = [
    "map.nii.gz",
    "S0_map.nii.gz",
    "offset_map.nii.gz",
    "r2.nii.gz",
]


def _write_fake_results(dicom_folder: Path, csv_suffixes: list[str]) -> None:
    """Create fake result files that mimic the legacy pipeline output."""
    for name in FAKE_NIFTI_FILES:
        (dicom_folder / name).write_text(f"dummy data for {name}")

    header = "mask_index;mean;std;min;max;Pixels;Mean R^2\n"
    row = "1;1.0;0.1;0.5;1.5;25;0.95\n"
    base = dicom_folder.name
    for suffix in csv_suffixes:
        csv_path = dicom_folder.parent / f"{base}_{suffix}.csv"
        csv_path.write_text(header + row)

    (dicom_folder / "acquisition_times.txt").write_text("0 10 20")


class FakeT2StarFitter:
    """Stub replacing the heavy legacy fitter."""

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, *, dicom_folder: Path, mask_file: Path, **kwargs: object) -> None:
        _write_fake_results(dicom_folder, ["S0", "offset", "t2_t2star"])


class FakeT1rhoFitter:
    """Stub replacing the heavy T1rho fitter."""

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, *, dicom_folder: Path, mask_file: Path, **kwargs: object) -> None:
        _write_fake_results(dicom_folder, ["S0", "offset", "t1rho"])


@pytest.fixture(autouse=True)
def patch_legacy_fitters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace heavy fitters with tiny test doubles."""
    monkeypatch.setattr(legacy_t2star, "T2_T2star", FakeT2StarFitter)
    monkeypatch.setattr(legacy_t1rho, "T1rho_T2prep", FakeT1rhoFitter)


def _assert_copied_results(output_dir: Path, csv_suffixes: list[str]) -> None:
    for name in FAKE_NIFTI_FILES:
        assert (output_dir / name).exists()

    for suffix in csv_suffixes:
        assert any(output_dir.glob(f"*_{suffix}.csv"))

    assert (output_dir / "acquisition_times.txt").exists()


def test_parse_boundary_string() -> None:
    lower, upper = parse_boundary_string("0,1;5,10;-1,1")
    assert lower == [0.0, 5.0, -1.0]
    assert upper == [1.0, 10.0, 1.0]


def test_print_csv_results(tmp_path: Path) -> None:
    csv_file = tmp_path / "results_t2.csv"
    csv_file.write_text("mask_index;mean;std;min;max;Pixels;Mean R^2\n1;2;3;1;4;10;0.9\n")
    # Should not raise even though console output isn't asserted here
    _print_csv_results(csv_file, "T2")


def test_copy_results_moves_files(tmp_path: Path) -> None:
    dicom_folder = tmp_path / "dicom"
    output_dir = tmp_path / "out"
    dicom_folder.mkdir()
    output_dir.mkdir()

    _write_fake_results(dicom_folder, ["S0", "offset"])
    _copy_results_and_display(
        source_dir=dicom_folder,
        output_dir=output_dir,
        param_names={"S0": "S0", "offset": "Offset"},
    )

    _assert_copied_results(output_dir, ["S0", "offset"])


def test_fit_t2star_command(tmp_path: Path, dicom_folder: Path, mask_file: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "results"
    args = [
        "fit",
        "t2star",
        str(dicom_folder),
        "--mask",
        str(mask_file),
        "--output",
        str(output_dir),
        "--min-r2",
        "0.8",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    _assert_copied_results(output_dir, ["S0", "offset", "t2_t2star"])


def test_fit_t2_function(tmp_path: Path, dicom_folder: Path, mask_file: Path) -> None:
    output_dir = tmp_path / "t2_results"
    fit_t2(
        dicom_folder=dicom_folder,
        mask=mask_file,
        output=output_dir,
    )
    _assert_copied_results(output_dir, ["S0", "offset", "t2_t2star"])


def test_fit_t1rho_function(tmp_path: Path, dicom_folder: Path, mask_file: Path) -> None:
    output_dir = tmp_path / "t1rho_results"
    fit_t1rho(
        dicom_folder=dicom_folder,
        mask=mask_file,
        tsl="0,20,40",
        output=output_dir,
    )
    _assert_copied_results(output_dir, ["S0", "offset", "t1rho"])


def test_fit_t2star_with_custom_boundary(
    tmp_path: Path, dicom_folder: Path, mask_file: Path
) -> None:
    output_dir = tmp_path / "boundary_results"
    fit_t2star(
        dicom_folder=dicom_folder,
        mask=mask_file,
        output=output_dir,
        boundary="0.9,2.0;0,50;-1,1",
    )
    _assert_copied_results(output_dir, ["S0", "offset", "t2_t2star"])


def test_fit_t2_with_config_file(tmp_path: Path, dicom_folder: Path, mask_file: Path) -> None:
    config_path = tmp_path / "t2_config.toml"
    config_path.write_text(
        """
boundary = [[1.0, 5.0, -0.5], [3.0, 40.0, 0.5]]
normalize = false
min_r2 = 0.9
"""
    )
    output_dir = tmp_path / "t2_config_results"
    fit_t2(
        dicom_folder=dicom_folder,
        mask=mask_file,
        output=output_dir,
        config_file=config_path,
    )
    _assert_copied_results(output_dir, ["S0", "offset", "t2_t2star"])


def test_fit_t1rho_with_config_file_for_aronen(
    tmp_path: Path, dicom_folder: Path, mask_file: Path
) -> None:
    config_path = tmp_path / "t1rho_config.toml"
    config_path.write_text(
        """
model = "aronen"
[sequence]
TR = 1000.0
T1 = 1200.0
alpha = 90.0
TE = 20.0
T2star = 40.0
"""
    )
    output_dir = tmp_path / "t1rho_config_results"
    fit_t1rho(
        dicom_folder=dicom_folder,
        mask=mask_file,
        tsl="0,10,40",
        output=output_dir,
        config_file=config_path,
    )
    _assert_copied_results(output_dir, ["S0", "offset", "t1rho"])


def test_estimate_boundaries_from_data(dicom_series_dir, mask_file) -> None:
    bounds = estimate_boundaries_from_data(dicom_series_dir, mask_file, modality="t2star")
    assert bounds == ((0.9, 0.0, -1.0), (2.0, 50.0, 1.0))
