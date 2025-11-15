#!/usr/bin/env python
"""Helper script to run example fittings with beautiful output."""
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from example_code import t2star_fitting_example, t2_fitting_example, t1rho_fitting_example

console = Console()


def print_banner(example_type: str):
    """Print a beautiful banner for the fitting type."""
    titles = {
        "t2star": "T2* Relaxation Time Fitting",
        "t2": "T2 Relaxation Time Fitting",
        "t1rho": "T1ρ Relaxation Time Fitting"
    }

    console.print(Panel.fit(
        f"[bold cyan]{titles.get(example_type, 'MRI Fitting')}[/bold cyan]\n"
        f"[dim]bMRI - Bio-sensitive MRI Analysis Framework[/dim]",
        border_style="cyan"
    ))


def print_csv_results(csv_file: Path):
    """Print CSV results in a beautiful table."""
    import csv

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)

        if not rows:
            return

        # Get parameter name from filename
        param_name = csv_file.stem.split('_')[-1]

        # Create table
        table = Table(title=f"[bold cyan]{param_name.upper()} Results[/bold cyan]",
                     show_header=True, header_style="bold yellow")

        table.add_column("ROI", justify="right", style="cyan")
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Std", justify="right", style="blue")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Pixels", justify="right", style="dim")
        table.add_column("R²", justify="right", style="magenta")

        for row in rows:
            table.add_row(
                row.get('mask_index', '-'),
                row.get('mean', '-'),
                row.get('std', '-'),
                row.get('min', '-'),
                row.get('max', '-'),
                row.get('Pixels', '-'),
                row.get('Mean R^2', '-')
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[dim]Could not parse {csv_file.name}: {e}[/dim]")


def print_results_summary(example_type: str, output_dir: Path):
    """Print a summary of the fitting results."""
    console.print("\n")

    # Find output files in the data folder
    nifti_files = list(output_dir.glob("*.nii.gz"))

    # CSV files are in parent directory with prefix
    folder_name = output_dir.name
    parent_dir = output_dir.parent
    csv_files = list(parent_dir.glob(f"{folder_name}_*.csv"))

    # Create results table
    table = Table(title="[bold green]✓ Fitting Complete![/bold green]",
                  show_header=True, header_style="bold magenta")
    table.add_column("Output Type", style="cyan")
    table.add_column("Files Generated", justify="right", style="green")

    table.add_row("NIfTI Maps", str(len(nifti_files)))
    table.add_row("CSV Results", str(len(csv_files)))

    console.print(table)

    # List key files
    if nifti_files:
        console.print("\n[bold]NIfTI Files:[/bold]")
        for f in sorted(nifti_files):
            console.print(f"  [dim]→[/dim] {f.name}")

    # Show CSV results
    if csv_files:
        console.print(f"\n[bold]CSV Results:[/bold] ({len(csv_files)} files)")
        for f in sorted(csv_files):
            console.print(f"  [dim]→[/dim] {f.name}")

        console.print()

        # Print contents of each CSV
        for csv_file in sorted(csv_files):
            print_csv_results(csv_file)


def run_with_progress(example_type: str, func):
    """Run fitting function with progress display."""
    print_banner(example_type)

    # Show what we're about to do
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")

    data_info = {
        "t2star": ("1056 DICOM files", "7_T2-star_map_3D_cor_03445"),
        "t2": ("300 DICOM files", "10_T2_map_cor_10282"),
        "t1rho": ("4×64 DICOM files", "T1rho (4 TSL timepoints)")
    }

    files, folder = data_info.get(example_type, ("Unknown", "Unknown"))

    info_table.add_row("📁 Data:", folder)
    info_table.add_row("📊 Files:", files)

    console.print(info_table)
    console.print("")

    # Run with progress indicator
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Running {example_type.upper()} fitting...",
            total=None
        )

        try:
            func()
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
            raise

    elapsed = time.time() - start_time

    # Print results summary
    base_path = Path("test/resources/Mrtc-Studie_Cartilage_transplantation_05_GAPF97478/20211125_0925")
    folder_map = {
        "t2star": base_path / "7_T2-star_map_3D_cor_03445",
        "t2": base_path / "10_T2_map_cor_10282",
        "t1rho": base_path / "T1rho"
    }
    output_dir = folder_map.get(example_type, base_path)

    print_results_summary(example_type, output_dir)

    console.print(f"\n[dim]Completed in {elapsed:.1f} seconds[/dim]\n")


def main():
    """Run example based on command line argument."""
    if len(sys.argv) < 2:
        console.print("[bold red]Error:[/bold red] Missing argument\n")
        console.print("[bold]Usage:[/bold] python run_example.py [t2star|t2|t1rho|all]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        table.add_row("python run_example.py t2star", "Run T2* fitting (1056 DICOM files)")
        table.add_row("python run_example.py t2", "Run T2 fitting (300 DICOM files)")
        table.add_row("python run_example.py t1rho", "Run T1ρ fitting (64 DICOM files)")
        table.add_row("python run_example.py all", "Run all examples sequentially")

        console.print(table)
        sys.exit(1)

    example_type = sys.argv[1].lower()

    examples = {
        "t2star": t2star_fitting_example,
        "t2": t2_fitting_example,
        "t1rho": t1rho_fitting_example  # Note: T1rho has data ordering issues with current test data
    }

    if example_type == "all":
        # Run all examples sequentially
        console.print(Panel.fit(
            "[bold cyan]Running All Examples[/bold cyan]\n"
            "[dim]T2*, T2, and T1ρ fittings will run sequentially[/dim]",
            border_style="cyan"
        ))
        console.print()

        for name, func in examples.items():
            run_with_progress(name, func)
            console.print("\n" + "─" * 70 + "\n")

        console.print("[bold green]✓ All examples completed![/bold green]\n")

    elif example_type in examples:
        run_with_progress(example_type, examples[example_type])
    else:
        console.print(f"[bold red]✗ Unknown example type:[/bold red] {example_type}")
        console.print("[dim]Valid options: t2star, t2, t1rho, all[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
