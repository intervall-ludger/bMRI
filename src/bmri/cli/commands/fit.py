"""Fitting commands for bMRI CLI.

This module provides CLI commands for fitting relaxation time maps from
DICOM data (T1, T2, T2*, T1rho).
"""

import csv
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from bmri.config import FittingModel, T1Config, T1rhoConfig, T1rhoSequenceConfig, T2Config
from bmri.config import load_config_from_toml
from bmri.exceptions import BMRIError
from bmri.logger import console, get_logger
from bmri.validators import (
    validate_boundary,
    validate_dicom_folder,
    validate_mask_file,
    validate_output_directory,
    validate_tsl_times,
)

logger = get_logger(__name__)

# Create fit subcommand group
app = typer.Typer(
    name="fit",
    help="Fit relaxation time maps from DICOM data",
    no_args_is_help=True,
)


def _print_csv_results(csv_file: Path, param_name: str) -> None:
    """Print CSV results in a beautiful table.

    Args:
        csv_file: Path to CSV file with ROI statistics
        param_name: Parameter name for table title (e.g., "T2*", "S0")
    """
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            rows = list(reader)

        if not rows:
            return

        # Create table
        table = Table(
            title=f"[bold cyan]{param_name} Results[/bold cyan]",
            show_header=True,
            header_style="bold yellow",
        )

        table.add_column("ROI", justify="right", style="cyan")
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Std", justify="right", style="blue")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Pixels", justify="right", style="dim")
        table.add_column("R²", justify="right", style="magenta")

        for row in rows:
            table.add_row(
                row.get("mask_index", "-"),
                row.get("mean", "-"),
                row.get("std", "-"),
                row.get("min", "-"),
                row.get("max", "-"),
                row.get("Pixels", "-"),
                row.get("Mean R^2", "-"),
            )

        console.print(table)
        console.print()

    except Exception as e:
        logger.debug(f"Could not parse {csv_file.name}: {e}")


def _copy_results_and_display(
    source_dir: Path, output_dir: Path, param_names: dict[str, str]
) -> None:
    """Copy results from source to output directory and display statistics.

    Args:
        source_dir: Directory where results were saved (usually DICOM folder)
        output_dir: Target output directory
        param_names: Mapping of CSV suffix to display name (e.g., {"t2_t2star": "T2*"})
    """
    # Copy NIfTI files
    nifti_files = list(source_dir.glob("*.nii.gz"))
    for nii_file in nifti_files:
        dest = output_dir / nii_file.name
        shutil.copy2(nii_file, dest)
        logger.debug(f"Copied {nii_file.name} to {output_dir}")

    # Copy CSV files
    folder_name = source_dir.name
    csv_files = list(source_dir.parent.glob(f"{folder_name}_*.csv"))
    for csv_file in csv_files:
        dest = output_dir / csv_file.name
        shutil.copy2(csv_file, dest)
        logger.debug(f"Copied {csv_file.name} to {output_dir}")

    # Copy acquisition times (if available)
    acquisition_file = source_dir / "acquisition_times.txt"
    if acquisition_file.exists():
        dest = output_dir / acquisition_file.name
        shutil.copy2(acquisition_file, dest)
        logger.debug(f"Copied {acquisition_file.name} to {output_dir}")

    # Display file summary
    console.print(f"\n[bold]Output Files:[/bold] {len(nifti_files)} NIfTI, {len(csv_files)} CSV")
    console.print(f"[dim]Location:[/dim] {output_dir}\n")

    # Display ROI statistics from CSV files
    if csv_files:
        for csv_file in sorted(csv_files):
            # Extract parameter name from filename (e.g., "folder_t2_t2star.csv" -> "t2_t2star")
            suffix = csv_file.stem.replace(folder_name + "_", "")
            display_name = param_names.get(suffix, suffix.upper())
            csv_path = output_dir / csv_file.name
            _print_csv_results(csv_path, display_name)


def estimate_boundaries_from_data(
    dicom_folder: Path, mask_file: Path, modality: str = "t2star"
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Estimate optimal boundaries from DICOM data.

    Analyzes signal characteristics to suggest reasonable parameter bounds.

    Args:
        dicom_folder: Path to DICOM folder
        mask_file: Path to mask file
        modality: Fitting modality ("t2star", "t2", "t1rho")

    Returns:
        Estimated boundary tuple ((lower...), (upper...))
    """
    import numpy as np
    from bmri.io.readers import get_dcm_list, split_dcm_list, get_dcm_array, load_nii

    logger.info("Estimating boundaries from data...")

    # Read DICOM data to analyze signal
    dcm_files = get_dcm_list(dicom_folder)
    echo_lists = split_dcm_list(dcm_files)

    # Get first echo data
    first_echo_array = get_dcm_array(echo_lists[0], apply_rescale=True)

    # Use all non-zero signal values for statistics (ignore background)
    # This is more robust than using the mask which might have different dimensions
    signal_values = first_echo_array[first_echo_array > 0].flatten()

    if len(signal_values) == 0:
        logger.warning("No signal data! Using default boundaries.")
        return ((0.9, 0, -1.0), (2, 50, 1.0))

    # Analyze signal statistics
    signal_mean = np.mean(signal_values)
    signal_std = np.std(signal_values)
    signal_min = np.min(signal_values)
    signal_max = np.max(signal_values)

    logger.info(
        f"Signal stats: mean={signal_mean:.1f}, std={signal_std:.1f}, "
        f"range=[{signal_min:.1f}, {signal_max:.1f}]"
    )

    # Use proven conservative boundaries instead of data-driven estimation
    # Data-driven estimation is too sensitive to noise and outliers

    if modality == "t2star":
        # Proven T2* boundaries for cartilage imaging
        s0_lower, s0_upper = 0.9, 2.0
        relax_lower, relax_upper = 0.0, 50.0  # Cap at 50ms for stable fitting
        offset_lower, offset_upper = -1.0, 1.0
    elif modality == "t2":
        # Proven T2 boundaries for cartilage imaging
        s0_lower, s0_upper = 0.9, 3.0
        relax_lower, relax_upper = 5.0, 40.0
        offset_lower, offset_upper = -0.5, 0.5
    else:  # t1rho
        # Proven T1rho boundaries
        s0_lower, s0_upper = 1.0, 10000.0
        relax_lower, relax_upper = 1.0, 500.0
        offset_lower, offset_upper = -1000.0, 1000.0

    boundaries = (
        (s0_lower, relax_lower, offset_lower),
        (s0_upper, relax_upper, offset_upper),
    )

    logger.info(f"Estimated boundaries: {boundaries}")
    return boundaries


def parse_boundary_string(boundary_str: str) -> tuple[list[float], list[float]]:
    """Parse boundary string like '0,1000;5,100;-10,10' into lower/upper tuples.

    Format: 'lower1,upper1;lower2,upper2;...'

    Example:
        >>> parse_boundary_string('0,1000;5,100;-10,10')
        ([0.0, 5.0, -10.0], [1000.0, 100.0, 10.0])
    """
    try:
        pairs = boundary_str.split(";")
        lower = []
        upper = []

        for pair in pairs:
            lo, hi = pair.split(",")
            lower.append(float(lo.strip()))
            upper.append(float(hi.strip()))

        return lower, upper
    except Exception as e:
        raise typer.BadParameter(
            f"Invalid boundary format: {boundary_str}\n"
            "Expected: 'lower1,upper1;lower2,upper2;...'\n"
            f"Example: '0,1000;5,100;-10,10'\n"
            f"Error: {str(e)}"
        )


def print_fitting_summary(
    dicom_folder: Path,
    mask_file: Path,
    output_dir: Path,
    config: T2Config | T1Config | T1rhoConfig,
) -> None:
    """Print summary of fitting configuration."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan bold")
    table.add_column(style="white")

    table.add_row("DICOM Folder:", str(dicom_folder))
    table.add_row("Mask File:", str(mask_file))
    table.add_row("Output Dir:", str(output_dir))
    table.add_row("Boundary:", f"{config.boundary}")
    table.add_row("Normalize:", str(config.normalize))
    table.add_row("Min R²:", f"{config.min_r2:.2f}")
    table.add_row("CPU Cores:", str(config.pools) if config.pools > 0 else "auto")

    console.print("\n[bold]Configuration:[/bold]")
    console.print(table)
    console.print()


@app.command(name="t2star")
def fit_t2star(
    dicom_folder: Annotated[
        Path,
        typer.Argument(
            help="Path to DICOM folder containing multi-echo T2* data",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    mask: Annotated[
        Path,
        typer.Option(
            "--mask",
            "-m",
            help="Path to NIfTI mask file",
            exists=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
        ),
    ] = Path("./bmri_output"),
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="TOML configuration file",
        ),
    ] = None,
    boundary: Annotated[
        str | None,
        typer.Option(
            help="Parameter boundaries as 'S0_min,S0_max;T2_min,T2_max;offset_min,offset_max'",
        ),
    ] = None,
    auto_boundary: Annotated[
        bool,
        typer.Option(
            "--auto-boundary",
            help="Show signal statistics and use proven boundaries (for debugging)",
        ),
    ] = False,
    normalize: Annotated[
        bool,
        typer.Option(
            help="Normalize signal intensities before fitting",
        ),
    ] = True,
    min_r2: Annotated[
        float,
        typer.Option(
            help="Minimum R² threshold for valid fits (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.75,
    pools: Annotated[
        int,
        typer.Option(
            help="Number of CPU cores for parallel processing (0=auto)",
            min=0,
        ),
    ] = 0,
    dim: Annotated[
        int,
        typer.Option(
            help="Dimensionality: 2D or 3D",
            min=2,
            max=3,
        ),
    ] = 3,
) -> None:
    """Fit T2* relaxation time maps from multi-echo DICOM data.

    T2* (T2-star) is the effective transverse relaxation time measured in
    gradient echo sequences. It combines T2 decay with magnetic field
    inhomogeneity effects.

    Example:
        # Basic usage with mask
        bmri fit t2star /data/patient01/t2star -m mask.nii.gz

        # With custom boundaries and output
        bmri fit t2star /data/patient01/t2star -m mask.nii.gz \\
            --boundary "0.9,2;0,50;-inf,inf" \\
            --output ./results \\
            --min-r2 0.8

        # Using configuration file
        bmri fit t2star /data/patient01/t2star -m mask.nii.gz -c t2star.toml
    """
    console.print(
        Panel.fit(
            "[bold cyan]T2* Relaxation Time Fitting[/bold cyan]\n"
            "[dim]Bio-sensitive MRI Analysis Framework[/dim]",
            border_style="cyan",
        )
    )

    try:
        # Validation
        dicom_folder, layout = validate_dicom_folder(dicom_folder)
        mask_file = validate_mask_file(mask)
        output_dir = validate_output_directory(output, create=True)

        logger.info(f"Detected DICOM layout: {layout.value}")

        # Load or create configuration
        if config_file:
            config = load_config_from_toml(config_file, T2Config)
            logger.info(f"Loaded configuration from {config_file}")
        else:
            # Determine boundaries
            if boundary:
                # User-provided boundary string
                lower, upper = parse_boundary_string(boundary)
                boundary_tuple = ((lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))
                logger.info("Using user-provided boundaries")
            elif auto_boundary:
                # Automatic estimation from data
                boundary_tuple = estimate_boundaries_from_data(
                    dicom_folder, mask_file, modality="t2star"
                )
            else:
                # Default T2* boundaries (optimized for cartilage): S0, T2*, offset
                boundary_tuple = ((0.9, 0, -1.0), (2, 50, 1.0))
                logger.info("Using default boundaries")

            # Validate boundary
            boundary_tuple = validate_boundary(boundary_tuple, num_params=3)

            config = T2Config(
                boundary=boundary_tuple,
                normalize=normalize,
                min_r2=min_r2,
                pools=pools,
                dim=dim,
            )

        print_fitting_summary(dicom_folder, mask_file, output_dir, config)

        # Import legacy modules - add project root to path dynamically
        import sys
        from pathlib import Path as _Path
        # __file__ is: .../bMRI/src/bmri/cli/commands/fit.py
        # We need:     .../bMRI (project root)
        _project_root = _Path(__file__).resolve().parent.parent.parent.parent.parent
        logger.debug(f"Adding to sys.path: {_project_root}")
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from src.Fitting.T2_T2star import T2_T2star
        from src.Utilitis import load_nii

        # Create fitter instance
        fitter = T2_T2star(
            dim=config.dim,
            boundary=config.boundary,
            normalize=config.normalize,
        )

        # Run fitting with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Fitting T2* maps...",
                total=None,
            )

            fitter.run(
                dicom_folder=dicom_folder,
                mask_file=mask_file,
                pools=config.pools,
                min_r2=config.min_r2,
            )

            progress.update(task, completed=True)

        # Copy results to output directory and display statistics
        console.print("\n[bold green]✓[/bold green] Fitting complete!")
        _copy_results_and_display(
            source_dir=dicom_folder,
            output_dir=output_dir,
            param_names={"t2_t2star": "T2*", "S0": "S0", "offset": "Offset"},
        )

    except BMRIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e.message}")
        if e.details:
            console.print(f"[dim]{e.details}[/dim]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {str(e)}")
        logger.exception("Unexpected error during T2* fitting")
        raise typer.Exit(code=1)


@app.command(name="t2")
def fit_t2(
    dicom_folder: Annotated[
        Path,
        typer.Argument(
            help="Path to DICOM folder containing T2 data",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    mask: Annotated[
        Path,
        typer.Option(
            "--mask",
            "-m",
            help="Path to NIfTI mask file",
            exists=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
        ),
    ] = Path("./bmri_output"),
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="TOML configuration file",
        ),
    ] = None,
    boundary: Annotated[
        str | None,
        typer.Option(
            help="Parameter boundaries as 'S0_min,S0_max;T2_min,T2_max;offset_min,offset_max'",
        ),
    ] = None,
    normalize: Annotated[
        bool,
        typer.Option(
            help="Normalize signal intensities before fitting",
        ),
    ] = True,
    min_r2: Annotated[
        float,
        typer.Option(
            help="Minimum R² threshold for valid fits (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.7,
    pools: Annotated[
        int,
        typer.Option(
            help="Number of CPU cores for parallel processing (0=auto)",
            min=0,
        ),
    ] = 0,
    dim: Annotated[
        int,
        typer.Option(
            help="Dimensionality: 2D or 3D",
            min=2,
            max=3,
        ),
    ] = 3,
) -> None:
    """Fit T2 relaxation time maps from DICOM data.

    T2 is the transverse relaxation time measured in spin echo sequences,
    sensitive to tissue composition and pathology.

    Example:
        bmri fit t2 /data/patient01/t2 -m mask.nii.gz \\
            --boundary "0.9,3;5,40;-0.5,0.5" \\
            --min-r2 0.7
    """
    console.print(
        Panel.fit(
            "[bold cyan]T2 Relaxation Time Fitting[/bold cyan]\n"
            "[dim]Bio-sensitive MRI Analysis Framework[/dim]",
            border_style="cyan",
        )
    )

    try:
        # Validation
        dicom_folder, layout = validate_dicom_folder(dicom_folder)
        mask_file = validate_mask_file(mask)
        output_dir = validate_output_directory(output, create=True)

        # Load or create configuration
        if config_file:
            config = load_config_from_toml(config_file, T2Config)
        else:
            if boundary:
                lower, upper = parse_boundary_string(boundary)
                boundary_tuple = ((lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))
            else:
                # Default T2 boundaries
                boundary_tuple = ((0.9, 5, -0.5), (3, 40, 0.5))

            boundary_tuple = validate_boundary(boundary_tuple, num_params=3)

            config = T2Config(
                boundary=boundary_tuple,
                normalize=normalize,
                min_r2=min_r2,
                pools=pools,
                dim=dim,
            )

        print_fitting_summary(dicom_folder, mask_file, output_dir, config)

        # Import legacy modules - add project root to path dynamically
        import sys
        from pathlib import Path as _Path
        _project_root = _Path(__file__).resolve().parent.parent.parent.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from src.Fitting.T2_T2star import T2_T2star

        fitter = T2_T2star(
            dim=config.dim,
            boundary=config.boundary,
            normalize=config.normalize,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Fitting T2 maps...", total=None)

            fitter.run(
                dicom_folder=dicom_folder,
                mask_file=mask_file,
                pools=config.pools,
                min_r2=config.min_r2,
            )

            progress.update(task, completed=True)

        # Copy results to output directory and display statistics
        console.print("\n[bold green]✓[/bold green] Fitting complete!")
        _copy_results_and_display(
            source_dir=dicom_folder,
            output_dir=output_dir,
            param_names={"t2_t2star": "T2", "S0": "S0", "offset": "Offset"},
        )

    except BMRIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e.message}")
        if e.details:
            console.print(f"[dim]{e.details}[/dim]")
        raise typer.Exit(code=1)


@app.command(name="t1rho")
def fit_t1rho(
    dicom_folder: Annotated[
        Path,
        typer.Argument(
            help="Path to DICOM folder containing T1rho data (with TSL subfolders)",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    mask: Annotated[
        Path,
        typer.Option(
            "--mask",
            "-m",
            help="Path to NIfTI mask file",
            exists=True,
        ),
    ],
    tsl: Annotated[
        str,
        typer.Option(
            help="Spin-lock times (TSL) as comma-separated values, e.g., '0,10,40,70'",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
        ),
    ] = Path("./bmri_output"),
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="TOML configuration file",
        ),
    ] = None,
    model: Annotated[
        FittingModel,
        typer.Option(
            help="Fitting model: mono_exp, aronen, or rausch",
        ),
    ] = FittingModel.MONO_EXP,
    boundary: Annotated[
        str | None,
        typer.Option(
            help="Parameter boundaries as 'S0_min,S0_max;T1rho_min,T1rho_max;offset_min,offset_max'",
        ),
    ] = None,
    min_r2: Annotated[
        float,
        typer.Option(
            help="Minimum R² threshold for valid fits (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.3,
    pools: Annotated[
        int,
        typer.Option(
            help="Number of CPU cores for parallel processing (0=auto)",
            min=0,
        ),
    ] = 0,
) -> None:
    """Fit T1rho relaxation time maps from DICOM data.

    T1ρ (T1-rho) is the spin-lattice relaxation time in the rotating frame,
    sensitive to macromolecular content and proteoglycan changes.

    Example:
        # Basic mono-exponential fit
        bmri fit t1rho /data/patient01/t1rho -m mask.nii.gz \\
            --tsl "0,10,40,70"

        # Aronen model with config file
        bmri fit t1rho /data/patient01/t1rho -m mask.nii.gz \\
            --tsl "0,10,40,70" \\
            --model aronen \\
            --config t1rho.toml
    """
    console.print(
        Panel.fit(
            "[bold cyan]T1ρ Relaxation Time Fitting[/bold cyan]\n"
            "[dim]Bio-sensitive MRI Analysis Framework[/dim]",
            border_style="cyan",
        )
    )

    try:
        # Validation
        dicom_folder, layout = validate_dicom_folder(dicom_folder)
        mask_file = validate_mask_file(mask)
        output_dir = validate_output_directory(output, create=True)
        tsl_times = validate_tsl_times(tsl)

        logger.info(f"TSL times: {tsl_times}")

        # Load or create configuration
        if config_file:
            config = load_config_from_toml(config_file, T1rhoConfig)
        else:
            if boundary:
                lower, upper = parse_boundary_string(boundary)
                boundary_tuple = ((lower[0], lower[1], lower[2]), (upper[0], upper[1], upper[2]))
            else:
                # Default T1rho boundaries
                boundary_tuple = ((1, 1, -1000), (10000, 500, 1000))

            boundary_tuple = validate_boundary(boundary_tuple, num_params=3)

            config = T1rhoConfig(
                model=model,
                boundary=boundary_tuple,
                min_r2=min_r2,
                pools=pools,
                dim=3,
            )

        print_fitting_summary(dicom_folder, mask_file, output_dir, config)

        # Import legacy modules - add project root to path dynamically
        import sys
        from pathlib import Path as _Path
        _project_root = _Path(__file__).resolve().parent.parent.parent.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from src.Fitting.T1rho_T2prep import T1rho_T2prep

        # Prepare sequence config for advanced models
        sequence_config = None
        if config.model != FittingModel.MONO_EXP:
            if config.sequence is None:
                raise typer.BadParameter(
                    f"Model '{config.model.value}' requires sequence parameters.\n"
                    "Please provide a config file with TR, T1, alpha, etc."
                )
            sequence_config = {
                "TR": config.sequence.TR,
                "T1": config.sequence.T1,
                "alpha": config.sequence.alpha,
                "TE": config.sequence.TE,
                "T2star": config.sequence.T2star,
            }

        fitter = T1rho_T2prep(
            dim=config.dim,
            boundary=config.boundary,
            normalize=config.normalize,
            config=sequence_config,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Fitting T1rho maps...", total=None)

            fitter.run(
                dicom_folder=dicom_folder,
                mask_file=mask_file,
                tsl=tsl_times,
                pools=config.pools,
                min_r2=config.min_r2,
            )

            progress.update(task, completed=True)

        # Copy results to output directory and display statistics
        console.print("\n[bold green]✓[/bold green] Fitting complete!")
        _copy_results_and_display(
            source_dir=dicom_folder,
            output_dir=output_dir,
            param_names={"t1rho": "T1ρ", "S0": "S0", "offset": "Offset"},
        )

    except BMRIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e.message}")
        if e.details:
            console.print(f"[dim]{e.details}[/dim]")
        raise typer.Exit(code=1)
