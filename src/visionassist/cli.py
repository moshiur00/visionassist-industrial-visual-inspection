"""VisionAssist command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from visionassist.benchmarks.build_visa_baseline import build_visa_baseline
from visionassist.benchmarks.schemas import load_benchmark_config
from visionassist.benchmarks.validate_benchmark import validate_baseline_benchmark
from visionassist.data.audit_visa import audit_visa
from visionassist.data.config import load_visa_config
from visionassist.data.phase1 import run_phase1
from visionassist.data.phase2 import run_phase2
from visionassist.data.phase3 import run_phase3
from visionassist.data.phase4 import run_phase4
from visionassist.data.phase5 import run_phase5
from visionassist.data.phase6 import run_phase6
from visionassist.evaluation.adapter import run_adapter_evaluation
from visionassist.evaluation.task_metrics import (
    evaluate_baseline_predictions,
    load_evaluation_config,
)
from visionassist.inference.generate import run_baseline_inference
from visionassist.inference.schemas import load_inference_config

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
console = Console()


def _config_option() -> Path:
    return Path("configs/data/visa.yaml")


def _benchmark_config_option() -> Path:
    return Path("configs/benchmark/visa_baseline_v1.yaml")


def _evaluation_config_option() -> Path:
    return Path("configs/evaluation/visa_baseline.yaml")


def _inference_config_option() -> Path:
    return Path("configs/inference/qwen25vl3b_direct.yaml")


def _adapter_evaluation_config_option() -> Path:
    return Path("configs/inference/qwen25vl3b_overfit_checkpoint50_validation.yaml")


@app.command("build-baseline-benchmark")
def build_baseline_benchmark_command(
    config: Path = typer.Option(
        _benchmark_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Build and freeze the deterministic Phase 7A benchmark."""

    result = build_visa_baseline(load_benchmark_config(config))
    console.print("[bold green]Phase 7A benchmark built successfully.[/bold green]")
    console.print(f"Benchmark: {result.benchmark_name}")
    console.print(f"Records: {result.records}")
    console.print(f"SHA-256: {result.benchmark_sha256}")
    console.print(f"Benchmark file: {result.benchmark_path}")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"Distribution: {result.distribution_path}")


@app.command("validate-baseline-benchmark")
def validate_baseline_benchmark_command(
    config: Path = typer.Option(
        _benchmark_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Validate the frozen Phase 7A benchmark and its images."""

    result = validate_baseline_benchmark(
        load_benchmark_config(config), project_root=project_root
    )
    console.print("[bold green]Phase 7A benchmark validation passed.[/bold green]")
    console.print(f"Records: {result.records}")
    console.print(f"Unique images: {result.unique_images}")
    console.print(f"Errors: {result.errors}")
    console.print(f"Warnings: {result.warnings}")
    console.print(f"Validation report: {result.report_path}")
    console.print(f"Statistics: {result.statistics_path}")
    console.print(f"Error report: {result.error_path}")


@app.command("evaluate-baseline")
def evaluate_baseline_command(
    benchmark: Path = typer.Option(
        Path("data/benchmarks/visa_baseline_v1/benchmark.jsonl"),
        "--benchmark",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    config: Path = typer.Option(
        _evaluation_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Evaluate baseline predictions with deterministic Phase 7B metrics."""

    result = evaluate_baseline_predictions(
        benchmark, predictions, load_evaluation_config(config)
    )
    console.print("[bold green]Phase 7B evaluation completed.[/bold green]")
    console.print(f"Benchmark records: {result.benchmark_records}")
    console.print(f"Predictions: {result.predictions}")
    console.print(f"Failure records: {result.failures}")
    console.print(f"Metrics: {result.metrics_path}")
    console.print(f"Per-task metrics: {result.per_task_path}")
    console.print(f"Per-category metrics: {result.per_category_path}")
    console.print(f"Failures: {result.failures_path}")




@app.command("baseline-inference")
def baseline_inference_command(
    config: Path = typer.Option(
        _inference_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Run resumable Phase 7C untouched-model inference."""

    result = run_baseline_inference(
        load_inference_config(config), project_root=project_root
    )
    status = "completed" if result.complete else "paused"
    console.print(f"[bold green]Phase 7C inference {status}.[/bold green]")
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Benchmark records: {result.benchmark_records}")
    console.print(f"Completed predictions: {result.completed_predictions}")
    console.print(f"New predictions this run: {result.new_predictions}")
    console.print(f"Errors this run: {result.errors}")
    console.print(f"Partial predictions: {result.partial_predictions_path}")
    if result.predictions_path is not None:
        console.print(f"Final predictions: {result.predictions_path}")
    console.print(f"Run manifest: {result.manifest_path}")


@app.command("evaluate-adapter")
def evaluate_adapter_command(
    config: Path = typer.Option(
        _adapter_evaluation_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Run resumable adapter inference and task-specific evaluation."""

    result = run_adapter_evaluation(
        load_inference_config(config), project_root=project_root
    )
    console.print(
        "[bold green]Post-training adapter evaluation completed.[/bold green]"
    )
    console.print(f"Records: {result.inference.benchmark_records}")
    console.print(f"Predictions: {result.inference.completed_predictions}")
    console.print(f"Failures: {result.evaluation.failures}")
    console.print(f"Summary: {result.summary_path}")
    console.print(f"Metrics: {result.evaluation.metrics_path}")


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


@app.command("phase6-visa")
def phase6_visa_command(
    config: Path = typer.Option(
        _config_option(), "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        exists=True,
        file_okay=False,
        readable=True,
        help="Project root used to resolve image paths.",
    ),
    processor_smoke_test: bool = typer.Option(
        False,
        "--processor-smoke-test",
        help="Load Qwen and run stratified token, vision-token, masking, and batch checks.",
    ),
    approve_gallery: bool = typer.Option(
        False,
        "--approve-gallery",
        help="Record that the generated gallery was manually reviewed and approved.",
    ),
    reviewer: str | None = typer.Option(
        None,
        "--reviewer",
        help="Name recorded with --approve-gallery.",
    ),
) -> None:
    """Validate Phase 5 data for VLM training and create readiness reports."""

    result = run_phase6(
        load_visa_config(config),
        project_root=project_root,
        processor_smoke_test=processor_smoke_test,
        approve_gallery=approve_gallery,
        reviewer=reviewer,
    )
    console.print("[bold green]Phase 6 completed successfully.[/bold green]")
    console.print(f"Instructions checked: {result.instructions}")
    console.print(f"Unique images: {result.unique_images}")
    console.print(f"Errors: {result.errors}")
    console.print(f"Warnings: {result.warnings}")
    console.print(f"Validation report: {result.report_path}")
    console.print(f"Statistics: {result.statistics_path}")
    console.print(f"Sample gallery: {result.gallery_path}")
    if result.processor_report_path is not None:
        console.print(f"Processor report: {result.processor_report_path}")
    if result.sequence_statistics_path is not None:
        console.print(f"Sequence statistics: {result.sequence_statistics_path}")
    console.print(f"Gallery review: {result.gallery_review_path}")
    console.print(f"Phase complete: {result.phase_complete}")
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


def _training_config_option() -> Path:
    return Path("configs/training/qwen25vl3b_qlora_overfit.yaml")


@app.command("training-environment")
def training_environment_command(
    config: Path = typer.Option(
        _training_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Inspect hardware and validate a Phase 8 training configuration."""

    from visionassist.training.config import load_training_config
    from visionassist.training.hardware import inspect_hardware, select_profile

    training_config = load_training_config(config)
    hardware = inspect_hardware(training_config.output_dir)
    profile = select_profile(hardware)
    console.print("[bold green]Phase 8 environment inspection completed.[/bold green]")
    console.print(f"Run ID: {training_config.run_id}")
    console.print(f"CUDA available: {hardware.cuda_available}")
    console.print(f"GPU: {hardware.gpu_name or 'none'}")
    console.print(f"VRAM: {hardware.total_vram_gib:.2f} GiB")
    console.print(f"BF16 supported: {hardware.bf16_supported}")
    console.print(f"Selected profile: {profile}")
    if not hardware.cuda_available:
        console.print(
            "[yellow]No local GPU detected. Configuration/tests are available locally; "
            "run smoke training and QLoRA in Google Colab Pro.[/yellow]"
        )


@app.command("training-data-audit")
def training_data_audit_command(
    config: Path = typer.Option(
        Path("configs/training/qwen25vl3b_qlora_pilot.yaml"),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional audit JSON path; defaults inside the run output directory.",
    ),
) -> None:
    """Audit and fingerprint deterministic training selections without a GPU."""

    from visionassist.training.config import load_training_config
    from visionassist.training.experiment import write_dataset_selection_audit

    training_config = load_training_config(config)
    audit_path = write_dataset_selection_audit(training_config, output)
    payload = __import__("json").loads(audit_path.read_text(encoding="utf-8"))
    console.print("[bold green]Training-data selection audit completed.[/bold green]")
    console.print(f"Train records: {payload['train']['records']}")
    console.print(f"Validation records: {payload['validation']['records']}")
    console.print(
        f"Instruction-ID SHA-256: {payload['train']['instruction_ids_sha256']}"
    )
    console.print(f"Audit: {audit_path}")


@app.command("training-smoke-test")
def training_smoke_test_command(
    config: Path = typer.Option(
        _training_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Run the Phase 8 one-batch QLoRA forward-pass validation."""

    from visionassist.training.config import load_training_config
    from visionassist.training.train import validate_one_batch

    result = validate_one_batch(
        load_training_config(config), project_root=project_root.resolve()
    )
    console.print("[bold green]Phase 8 one-batch smoke test passed.[/bold green]")
    console.print(f"Loss: {result['loss']}")
    console.print(f"Batch shape: {result['batch_shape']}")
    console.print(f"Trainable parameters: {result['trainable_parameters']}")


@app.command("train-qlora")
def train_qlora_command(
    config: Path = typer.Option(
        _training_config_option(),
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Override config with none, latest, best, or an explicit checkpoint path.",
    ),
    project_root: Path = typer.Option(
        Path.cwd(),
        "--project-root",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Run resumable Phase 8 QLoRA training with bounded checkpoints."""

    from visionassist.training.config import load_training_config
    from visionassist.training.train import run_qlora_training

    result = run_qlora_training(
        load_training_config(config),
        project_root=project_root.resolve(),
        resume_override=resume,
    )
    console.print("[bold green]Phase 8 QLoRA run completed.[/bold green]")
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Global step: {result.global_step}")
    console.print(f"Resumed from: {result.resumed_from or 'none'}")
    console.print(f"Best checkpoint: {result.best_checkpoint or 'none'}")
    console.print(f"Final adapter: {result.final_adapter}")
