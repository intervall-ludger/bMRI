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


def test_launch_t2star_viewer_uses_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bmri.cli.commands.view as view

    captured: dict[str, object] = {}

    def fake_prepare(*_args, **_kwargs):
        return (
            tmp_path / "d.nii.gz",
            tmp_path / "p.nii.gz",
            np.array([0.0, 5.0]),
            2,
        )

    class FakeApp:
        @classmethod
        def instance(cls):
            return None

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            captured["exec"] = True

    class FakeViewer:
        def __init__(self):
            captured["viewer_created"] = True

        def start(self, **kwargs):
            captured["start_kwargs"] = kwargs

        def setWindowTitle(self, title):
            captured["title"] = title

        def show(self):
            captured["show"] = True

    def fake_import():
        return FakeApp, FakeViewer, (lambda *a, **k: None)

    monkeypatch.setattr(view, "prepare_t2star_viewer_inputs", fake_prepare)
    monkeypatch.setattr(view, "_import_viewer_components", fake_import)

    view.launch_t2star_viewer(
        tmp_path / "results",
        parameter="t2_t2star",
        alpha=0.4,
        auto_cut=False,
        normalize=False,
        vmin=1.0,
        vmax=5.0,
    )

    assert captured["title"] == "bMRI Viewer – T2*"
    kwargs = captured["start_kwargs"]
    assert kwargs["alpha"] == 0.4
    assert kwargs["normalize"] is False
    assert kwargs["auto_cut"] is False
    assert kwargs["vmin"] == 1.0
    assert kwargs["vmax"] == 5.0
    assert kwargs["c_int"] == 2


def test_view_cli_passes_color_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import bmri.cli.commands.view as view
    from bmri.cli import main
    from typer.testing import CliRunner

    res_dir = tmp_path / "results"
    res_dir.mkdir()

    captured: dict[str, object] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        view,
        "launch_t2star_viewer",
        lambda results, **kwargs: fake_launch(results=results, **kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        main.app,
        [
            "view",
            "t2star",
            str(res_dir),
            "--parameter",
            "s0",
            "--vmin",
            "2.5",
            "--vmax",
            "9.5",
            "--alpha",
            "0.2",
            "--no-auto-crop",
            "--no-normalize",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["results"] == res_dir
    assert captured["vmin"] == 2.5
    assert captured["vmax"] == 9.5
    assert captured["alpha"] == 0.2
    assert captured["auto_cut"] is False
    assert captured["normalize"] is False


def test_view_cli_handles_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import bmri.cli.commands.view as view
    from bmri.cli import main
    from typer.testing import CliRunner

    res_dir = tmp_path / "results"
    res_dir.mkdir()

    def raise_value_error(*args, **kwargs):
        raise ValueError("boom")

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("kaboom")

    runner = CliRunner()

    monkeypatch.setattr(view, "launch_t2star_viewer", raise_value_error)
    result = runner.invoke(main.app, ["view", "t2star", str(res_dir)])
    assert result.exit_code == 1
    assert "Viewer error" in result.stdout

    monkeypatch.setattr(view, "launch_t2star_viewer", raise_runtime_error)
    result = runner.invoke(main.app, ["view", "t2star", str(res_dir)])
    assert result.exit_code == 1
    assert "Unable to start viewer" in result.stdout


def test_resolve_results_dir_errors(tmp_path: Path) -> None:
    import bmri.cli.commands.view as view

    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    with pytest.raises(FileNotFoundError):
        view._resolve_results_dir(file_path)


def test_import_function_metadata_runs() -> None:
    import bmri.cli.commands.view as view

    mono_exp, params_fn = view._import_function_metadata()
    names = params_fn(mono_exp)
    assert len(names) >= 3
