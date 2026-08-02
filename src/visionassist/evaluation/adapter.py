"""End-to-end post-training adapter inference and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from visionassist.evaluation.task_metrics import (
    EvaluationConfig,
    EvaluationResult,
    evaluate_baseline_predictions,
)
from visionassist.inference.generate import (
    BaselineInferenceResult,
    run_baseline_inference,
)
from visionassist.inference.model_loader import LoadedInferenceModel, load_qwen25vl
from visionassist.inference.schemas import InferenceConfig


@dataclass(frozen=True)
class AdapterEvaluationResult:
    """Artifacts produced by one adapter assessment split."""

    inference: BaselineInferenceResult
    evaluation: EvaluationResult
    summary_path: Path


def _evaluation_config(output_dir: Path) -> EvaluationConfig:
    return EvaluationConfig(
        output_root=output_dir,
        metrics_path=output_dir / "metrics.json",
        per_task_path=output_dir / "per_task_metrics.csv",
        per_category_path=output_dir / "per_category_metrics.csv",
        failures_path=output_dir / "failures.jsonl",
        parsing_errors_path=output_dir / "parsing_errors.jsonl",
    )


def run_adapter_evaluation(
    config: InferenceConfig,
    *,
    project_root: Path = Path.cwd(),
    loader: Callable[[InferenceConfig], LoadedInferenceModel] = load_qwen25vl,
) -> AdapterEvaluationResult:
    """Generate adapter predictions and immediately score the exact selected records."""

    if config.adapter_path is None:
        raise ValueError("adapter_path is required for post-training evaluation.")
    if config.evaluation_records_path is None:
        raise ValueError("evaluation_records_path is required for adapter evaluation.")

    inference = run_baseline_inference(
        config, project_root=project_root, loader=loader
    )
    if not inference.complete or inference.predictions_path is None:
        raise RuntimeError("Adapter inference paused before evaluation was complete.")

    evaluation_dir = config.output_dir / "evaluation"
    evaluation = evaluate_baseline_predictions(
        config.evaluation_records_path,
        inference.predictions_path,
        _evaluation_config(evaluation_dir),
    )
    metrics = json.loads(evaluation.metrics_path.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "adapter_path": config.adapter_path.as_posix(),
        "dataset_split": config.expected_dataset_split,
        "records": inference.benchmark_records,
        "predictions": inference.completed_predictions,
        "inference_errors": inference.errors,
        "failure_records": evaluation.failures,
        "failure_rate": metrics.get("failure_rate"),
        "metrics_path": evaluation.metrics_path.as_posix(),
        "predictions_path": inference.predictions_path.as_posix(),
        "evaluation_records_path": config.evaluation_records_path.as_posix(),
    }
    summary_path = config.output_dir / "assessment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return AdapterEvaluationResult(inference, evaluation, summary_path)
