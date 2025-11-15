"""Visualization commands for the bMRI CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.panel import Panel

from bmri.logger import console, get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="view",
    help="Visualize fitted relaxation time maps with the PyQt viewer",
    no_args_is_help=True,
)

PARAMETER_ALIASES = {
    "s0": "S0",
    "t2": "t2_t2star",
    "t2*": "t2_t2star",
    "t2star": "t2_t2star",
    "t2_t2star": "t2_t2star",
    "offset": "offset",
}


def _project_root() -> Path:
    """Return project root so legacy modules can be imported."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def _import_function_metadata():
    """Import only the components required for parameter metadata."""
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))

    from src.Fitting.T2_T2star import mono_exp
    from src.Utilitis.utils import get_function_parameter

    return mono_exp, get_function_parameter


def _import_viewer_components():
    """Import the PyQt viewer and fitting function."""
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))

    from PyQt5.QtWidgets import QApplication
    from src.Fitting.T2_T2star import mono_exp
    from src.Visualization.image_viewer import ImageViewer

    return QApplication, ImageViewer, mono_exp


def _resolve_results_dir(path: Path) -> Path:
    """Verify that the provided path exists and is a directory."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Results directory not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Expected a directory, got: {path}")
    return path


def _normalize_parameter(name: str) -> str:
    """Map user-provided parameter names to canonical names."""
    canonical = PARAMETER_ALIASES.get(name.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown parameter '{name}'. Valid options: {', '.join(sorted(set(PARAMETER_ALIASES.values())))}"
        )
    return canonical


def prepare_t2star_viewer_inputs(
    results_dir: Path,
    *,
    parameter: str = "t2_t2star",
    dicom_file: Path | None = None,
    params_file: Path | None = None,
    times_file: Path | None = None,
) -> tuple[Path, Path, np.ndarray, int]:
    """Prepare paths and metadata required by the viewer."""
    results_dir = _resolve_results_dir(results_dir)

    dicom_path = (dicom_file or (results_dir / "dicom.nii.gz")).resolve()
    if not dicom_path.exists():
        raise FileNotFoundError(
            f"dicom.nii.gz not found. Provide it explicitly with --dicom. Looked in {dicom_path}"
        )

    params_path = (params_file or (results_dir / "params.nii.gz")).resolve()
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.nii.gz not found. Provide it explicitly with --params. Looked in {params_path}"
        )

    times_path = (times_file or (results_dir / "acquisition_times.txt")).resolve()
    if not times_path.exists():
        raise FileNotFoundError(
            f"acquisition_times.txt not found. Provide it explicitly with --times. Looked in {times_path}"
        )

    try:
        times = np.atleast_1d(np.loadtxt(times_path).astype(float))
    except Exception as exc:  # pragma: no cover - numpy already tested elsewhere
        raise ValueError(f"Could not read acquisition times: {times_path}") from exc

    mono_exp, get_params = _import_function_metadata()
    parameter_names = get_params(mono_exp)
    canonical = _normalize_parameter(parameter)
    try:
        parameter_index = [name.lower() for name in parameter_names].index(canonical.lower())
    except ValueError:
        # Fallback if viewer function order changes
        parameter_index = 1

    return dicom_path, params_path, times, parameter_index


def launch_t2star_viewer(
    results_dir: Path,
    *,
    parameter: str = "t2_t2star",
    dicom_file: Path | None = None,
    params_file: Path | None = None,
    times_file: Path | None = None,
    alpha: float = 0.35,
    auto_cut: bool = True,
    normalize: bool = True,
) -> None:
    """Launch the PyQt viewer for T2*/T2 results."""
    dicom_path, params_path, times, parameter_index = prepare_t2star_viewer_inputs(
        results_dir,
        parameter=parameter,
        dicom_file=dicom_file,
        params_file=params_file,
        times_file=times_file,
    )

    QApplication, ImageViewer, mono_exp = _import_viewer_components()

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.start(
        dicom=dicom_path,
        fit_maps=params_path,
        fit_function=mono_exp,
        time_points=times.tolist(),
        c_int=parameter_index,
        alpha=alpha,
        normalize=normalize,
        auto_cut=auto_cut,
    )
    viewer.setWindowTitle("bMRI Viewer – T2*")
    viewer.show()
    app.exec()  # pragma: no cover


@app.command("t2star")
def view_t2star_command(
    results: Annotated[
        Path,
        typer.Argument(
            help="Directory containing dicom.nii.gz, params.nii.gz and acquisition_times.txt",
        ),
    ],
    parameter: Annotated[
        str,
        typer.Option(
            "--parameter",
            "-p",
            help="Parameter to highlight (S0, t2_t2star, offset)",
        ),
    ] = "t2_t2star",
    dicom: Annotated[
        Path | None,
        typer.Option(
            "--dicom",
            help="Path to dicom.nii.gz (defaults to RESULTS/dicom.nii.gz)",
        ),
    ] = None,
    params: Annotated[
        Path | None,
        typer.Option(
            "--params",
            help="Path to params.nii.gz (defaults to RESULTS/params.nii.gz)",
        ),
    ] = None,
    times: Annotated[
        Path | None,
        typer.Option(
            "--times",
            help="Path to acquisition_times.txt (defaults to RESULTS/acquisition_times.txt)",
        ),
    ] = None,
    alpha: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Overlay opacity for the colored map",
        ),
    ] = 0.35,
    auto_crop: Annotated[
        bool,
        typer.Option(
            "--auto-crop/--no-auto-crop",
            help="Automatically crop to mask bounding box",
        ),
    ] = True,
    normalize: Annotated[
        bool,
        typer.Option(
            "--normalize/--no-normalize",
            help="Normalize signal intensities before plotting",
        ),
    ] = True,
) -> None:
    """Launch the interactive viewer for T2*/T2 fitted maps."""
    console.print(
        Panel.fit(
            "[bold cyan]Launching bMRI Viewer[/bold cyan]\n"
            "[dim]Close the window to return to the terminal[/dim]",
            border_style="cyan",
        )
    )
    try:
        launch_t2star_viewer(
            results,
            parameter=parameter,
            dicom_file=dicom,
            params_file=params,
            times_file=times,
            alpha=alpha,
            auto_cut=auto_crop,
            normalize=normalize,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]✗ Viewer error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[bold red]✗ Unable to start viewer:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
