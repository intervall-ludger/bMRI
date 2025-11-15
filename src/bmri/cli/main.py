"""Main CLI application for bMRI.

This module defines the primary entry point for the bMRI command-line interface
using Typer for modern, type-safe CLI design.
"""

import sys
from pathlib import Path

# Add src directory to Python path for legacy module imports
_src_path = Path(__file__).parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import typer
from rich.console import Console
from rich.panel import Panel

from bmri import __version__
from bmri.config import CLISettings
from bmri.logger import setup_logging

# Create main Typer app
app = typer.Typer(
    name="bmri",
    help="Bio-sensitive MRI Analysis Framework - Fit relaxation times from DICOM data",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Global console for rich output
console = Console()

# Global settings
settings = CLISettings()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold cyan]bMRI[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Minimal output",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Write logs to file",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Bio-sensitive MRI Analysis Framework.

    Fit relaxation times (T1, T2, T2*, T1ρ) from DICOM data with professional
    CLI tools and beautiful terminal output.

    Example:
        # Fit T2* from DICOM folder
        bmri fit t2star /data/patient01/t2star -m mask.nii.gz

        # Fit T1rho with configuration file
        bmri fit t1rho /data/patient01/t1rho -m mask.nii.gz -c t1rho.toml

        # Visualize results
        bmri visualize results/t2_t2star_map.nii.gz
    """
    # Update global settings
    settings.verbose = verbose
    settings.quiet = quiet
    settings.log_file = log_file

    # Setup logging based on flags
    log_level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    setup_logging(level=log_level, log_file=log_file)  # type: ignore


# Import and register subcommands
from bmri.cli.commands import fit

app.add_typer(fit.app, name="fit", help="Fit relaxation time maps")


def cli_entry_point() -> None:
    """Entry point for console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if settings.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    cli_entry_point()
