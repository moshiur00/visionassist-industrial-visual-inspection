from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visionassist.benchmarks.build_visa_baseline import build_visa_baseline
from visionassist.benchmarks.schemas import BenchmarkConfig, BenchmarkTaskTargets
from visionassist.benchmarks.validate_benchmark import validate_baseline_benchmark
from visionassist.schemas.instruction import InstructionRecord


def make_record(
    index: int,
    task: str,
    category: str,
    condition: str,
    image_path: str,
) -> InstructionRecord:
    anomalous = condition == "anomalous"
    answer = {
        "binary_inspection": "The item is defective." if anomalous else "The item is normal.",
        "product_identification": f"The product category is {category}.",
        "defect_identification": "The annotated defect is scratch.",
        "localization": "The anomaly centroid falls in the center region.",
        "evidence_explanation": "The inspection fails because scratch is annotated.",
        "structured_report": json.dumps(
            {
                "product": category,
                "condition": "defective" if anomalous else "normal",
                "defect_type": "scratch" if anomalous else None,
                "location": "center" if anomalous else None,
                "visual_severity": "minor" if anomalous else "none",
                "recommended_action": "Manual review" if anomalous else "Accept",
                "safety_note": "Root cause cannot be determined from this image alone.",
            }
        ),
        "technician_note": "QC note: reject for scratch.",
        "uncertainty": "The root cause cannot be determined from this image alone.",
    }[task]
    payload = {
        "schema_version": "1.0",
        "instruction_id": f"sample_{index}_{task}",
        "image_id": f"image_{index}",
        "dataset_split": "test",
        "task_family": task,
        "template_id": f"{task}_01",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Inspect the image."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "metadata": {
            "source": "visa",
            "category": category,
            "condition": condition,
            "defect_type": "scratch" if anomalous else None,
            "location": "center" if anomalous else None,
            "visual_severity": "minor" if anomalous else "none",
            "answer_format": "json" if task == "structured_report" else "text",
            "grounded_by": ["test"],
        },
    }
    return InstructionRecord.model_validate(payload)


def test_build_and_validate_small_benchmark(tmp_path: Path) -> None:
    image_dir = tmp_path / "data" / "raw"
    image_dir.mkdir(parents=True)
    source = tmp_path / "test.jsonl"
    tasks = list(BenchmarkTaskTargets.model_fields)
    records: list[InstructionRecord] = []
    for task_index, task in enumerate(tasks):
        for offset in range(4):
            image = image_dir / f"{task_index}_{offset}.jpg"
            Image.new("RGB", (16, 16)).save(image)
            records.append(
                make_record(
                    task_index * 10 + offset,
                    task,
                    "pcb1" if offset % 2 == 0 else "candle",
                    "anomalous" if offset % 2 == 0 else "normal",
                    image.relative_to(tmp_path).as_posix(),
                )
            )
    source.write_text("\n".join(record.model_dump_json() for record in records) + "\n")
    targets = BenchmarkTaskTargets(**{task: 2 for task in tasks})
    root = tmp_path / "benchmark"
    config = BenchmarkConfig(
        source_test_path=source,
        output_root=root,
        benchmark_path=root / "benchmark.jsonl",
        manifest_path=root / "manifest.json",
        distribution_path=root / "distribution.json",
        sha256_path=root / "sha256.txt",
        validation_report_path=tmp_path / "report.json",
        validation_error_path=tmp_path / "errors.jsonl",
        statistics_path=tmp_path / "stats.json",
        task_targets=targets,
    )

    first = build_visa_baseline(config)
    first_content = config.benchmark_path.read_bytes()
    second = build_visa_baseline(config)
    assert first.records == 16
    assert first.benchmark_sha256 == second.benchmark_sha256
    assert config.benchmark_path.read_bytes() == first_content

    validation = validate_baseline_benchmark(config, project_root=tmp_path)
    assert validation.passed
    assert validation.errors == 0
    assert validation.records == 16
