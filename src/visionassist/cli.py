"""VisionAssist command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from visionassist.data.audit_visa import audit_visa
from visionassist.data.config import load_visa_config
from visionassist.data.phase1 import run_phase1
from visionassist.data.phase2 import run_phase2
from visionassist.data.phase3 import run_phase3
from visionassist.data.phase4 import run_phase4
from visionassist.data.phase5 import run_phase5

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()


def _config_option() -> Path:
    return Path("configs/data/visa.yaml")


@app.command("phase1-visa")
def phase1_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
    force_download: bool = typer.Option(False, help="Discard any archive and download again."),
    force_extract: bool = typer.Option(False, help="Discard extracted data and extract again."),
) -> None:
    """Download, verify, extract, audit, license, and document VisA."""

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading VisA", total=None)

        def update(completed: int, total: int | None) -> None:
            progress.update(task, completed=completed, total=total)

        result = run_phase1(
            load_visa_config(config),
            force_download=force_download,
            force_extract=force_extract,
            progress=update,
        )

    console.print("[bold green]Phase 1 completed successfully.[/bold green]")
    console.print(f"Images audited: {result.audit.record_count}")
    console.print(f"Archive SHA-256: {result.archive_sha256}")
    console.print(f"Receipt: {result.receipt_path}")
    console.print(f"Manifest: {result.audit.manifest_path}")
    console.print(f"Audit summary: {result.audit.summary_path}")
    console.print(f"License report: {result.license_report_path}")
    console.print(f"Dataset card: {result.dataset_card_path}")


@app.command("phase2-visa")
def phase2_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
) -> None:
    """Parse VisA CSVs and masks into validated canonical metadata."""

    result = run_phase2(load_visa_config(config))
    console.print("[bold green]Phase 2 completed successfully.[/bold green]")
    console.print(f"Canonical records: {result.records}")
    console.print(f"Errors: {result.errors}")
    console.print(f"Warnings: {result.warnings}")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"Validation report: {result.report_path}")
    console.print(f"Error report: {result.error_path}")


@app.command("phase3-visa")
def phase3_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
) -> None:
    """Derive bounding boxes, centroids, locations, area, and visual severity."""

    result = run_phase3(load_visa_config(config))
    console.print("[bold green]Phase 3 completed successfully.[/bold green]")
    console.print(f"Feature records: {result.records}")
    console.print(f"Anomalous records: {result.anomalous_records}")
    console.print(f"Errors: {result.errors}")
    console.print(f"Warnings: {result.warnings}")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"Validation report: {result.report_path}")
    console.print(f"Error report: {result.error_path}")


@app.command("phase4-visa")
def phase4_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
) -> None:
    """Create deterministic stratified splits and validate leakage."""

    result = run_phase4(load_visa_config(config))
    console.print("[bold green]Phase 4 completed successfully.[/bold green]")
    console.print(f"Total records: {result.records}")
    console.print(f"Train records: {result.train_records}")
    console.print(f"Validation records: {result.validation_records}")
    console.print(f"Test records: {result.test_records}")
    console.print(f"Errors: {result.errors}")
    console.print(f"Warnings: {result.warnings}")
    console.print(f"Split directory: {result.split_directory}")
    console.print(f"Assignment manifest: {result.assignment_path}")
    console.print(f"Validation report: {result.report_path}")
    console.print(f"Error report: {result.error_path}")


@app.command("phase5-visa")
def phase5_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
) -> None:
    """Generate grounded multimodal instruction records from Phase 4 splits."""

    result = run_phase5(load_visa_config(config))
    console.print("[bold green]Phase 5 completed successfully.[/bold green]")
    console.print(f"Instructions: {result.instructions}")
    console.print(f"Unique images: {result.unique_images}")
    console.print(f"Train instructions: {result.train_instructions}")
    console.print(f"Validation instructions: {result.validation_instructions}")
    console.print(f"Test instructions: {result.test_instructions}")
    console.print(f"Errors: {result.errors}")
    console.print(f"Warnings: {result.warnings}")
    console.print(f"Output directory: {result.output_directory}")
    console.print(f"Validation report: {result.report_path}")
    console.print(f"Error report: {result.error_path}")


@app.command("audit-visa")
def audit_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
) -> None:
    """Audit an already extracted VisA dataset and rebuild its raw manifest."""

    result = audit_visa(load_visa_config(config))
    console.print(f"[bold green]Audited {result.record_count} images.[/bold green]")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"Summary: {result.summary_path}")
    if result.error_count:
        console.print(f"[yellow]Warnings/errors: {result.error_count}[/yellow]")


if __name__ == "__main__":
    app()
