"""Deterministic, grounded VisA instruction generation for Phase 5."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from visionassist.data.config import VisaConfig
from visionassist.schemas.dataset import Condition, DatasetSplit, DerivedImageRecord
from visionassist.schemas.instruction import InstructionRecord


class InstructionGenerationError(RuntimeError):
    """Raised when Phase 5 cannot produce trustworthy instruction data."""


@dataclass(frozen=True)
class Phase5Result:
    """Counts and paths emitted by Phase 5."""

    instructions: int
    train_instructions: int
    validation_instructions: int
    test_instructions: int
    unique_images: int
    errors: int
    warnings: int
    output_directory: Path
    report_path: Path
    error_path: Path


@dataclass(frozen=True)
class Template:
    """One deterministic prompt/answer template."""

    family: str
    template_id: str
    prompt: str
    answer_format: str
    answer_builder: Callable[[DerivedImageRecord], str]


def _read_split(path: Path, split: DatasetSplit) -> list[DerivedImageRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase 4 split not found: {path}. Run phase4-visa first.")
    records: list[DerivedImageRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                payload.pop("dataset_split", None)
                record = DerivedImageRecord.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise InstructionGenerationError(
                    f"Invalid Phase 4 record at {path}:{row_number}: {exc}"
                ) from exc
            records.append(record)
    return records


def _humanize(value: str | None) -> str:
    if not value:
        return "unspecified visual anomaly"
    return value.replace("_", " ").replace(",", ", ")


def _status(record: DerivedImageRecord) -> str:
    return "defective" if record.condition is Condition.ANOMALOUS else "normal"


def _location(record: DerivedImageRecord) -> str:
    return _humanize(record.nine_grid_location.value if record.nine_grid_location else None)


def _binary_answer(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return (
            "The item is normal. The dataset annotation contains no anomaly mask for this "
            "image, so no defect is identified."
        )
    return (
        f"The item is defective. The annotated defect is {_humanize(record.defect_type)} "
        f"in the {_location(record)} region."
    )


def _product_answer(record: DerivedImageRecord) -> str:
    return f"The product category is {record.category}."


def _defect_answer(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return "No defect is annotated for this image."
    return f"The annotated defect is {_humanize(record.defect_type)}."


def _location_answer(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return "No anomaly location is available because the image is annotated as normal."
    return f"The anomaly centroid falls in the {_location(record)} region."


def _explanation_answer(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return (
            "This image is classified as normal because the source annotation provides no "
            "anomaly mask or defect label. This does not prove the absence of every possible "
            "real-world issue; it reflects the available annotation."
        )
    percentage = record.anomaly_area_ratio * 100.0
    return (
        f"The inspection fails because the source annotation marks {_humanize(record.defect_type)}. "
        f"The mask covers {percentage:.3f}% of the image and its centroid lies in the "
        f"{_location(record)} region. The project-defined visual severity is "
        f"{record.visual_severity.value}."
    )


def _json_answer(record: DerivedImageRecord) -> str:
    payload = {
        "product": record.category,
        "condition": _status(record),
        "defect_type": _humanize(record.defect_type)
        if record.condition is Condition.ANOMALOUS
        else None,
        "location": _location(record)
        if record.condition is Condition.ANOMALOUS
        else None,
        "visual_severity": record.visual_severity.value,
        "recommended_action": (
            "Send the item for manual verification and do not pass automatic quality control."
            if record.condition is Condition.ANOMALOUS
            else "Accept under the current visual inspection criteria."
        ),
        "safety_note": "Mechanical safety and root cause cannot be determined from this image alone.",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _technician_answer(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return (
            f"QC note: {record.category} sample appears normal under the supplied annotation. "
            "No anomaly region is marked."
        )
    return (
        f"QC note: Reject {record.category} sample for manual review. Annotated issue: "
        f"{_humanize(record.defect_type)}; location: {_location(record)}; visual severity: "
        f"{record.visual_severity.value}."
    )


def _uncertainty_answer(record: DerivedImageRecord) -> str:
    if record.condition is Condition.NORMAL:
        return (
            "No root cause can be determined from this image. The source only indicates that "
            "no visual anomaly was annotated."
        )
    return (
        f"The visible annotation supports describing {_humanize(record.defect_type)}, but the "
        "mechanical root cause and safety impact cannot be established from this image alone."
    )


def _templates() -> list[Template]:
    return [
        Template("binary_inspection", "binary_01", "Is this item normal or defective? Explain briefly.", "text", _binary_answer),
        Template("binary_inspection", "binary_02", "Should this product pass visual quality control?", "text", _binary_answer),
        Template("binary_inspection", "binary_03", "Inspect the image and state whether an annotated anomaly is present.", "text", _binary_answer),
        Template("product_identification", "product_01", "What product category is shown?", "text", _product_answer),
        Template("product_identification", "product_02", "Identify the industrial inspection category in this image.", "text", _product_answer),
        Template("defect_identification", "defect_01", "What defect is annotated in this image?", "text", _defect_answer),
        Template("defect_identification", "defect_02", "Name the visual anomaly, or state that none is annotated.", "text", _defect_answer),
        Template("defect_identification", "defect_03", "Describe the defect type without inventing a root cause.", "text", _defect_answer),
        Template("localization", "location_01", "Where is the annotated anomaly located?", "text", _location_answer),
        Template("localization", "location_02", "Which coarse image region should a technician inspect?", "text", _location_answer),
        Template("localization", "location_03", "Report the anomaly location using a nine-grid description.", "text", _location_answer),
        Template("evidence_explanation", "explain_01", "Explain the annotation evidence supporting the inspection decision.", "text", _explanation_answer),
        Template("evidence_explanation", "explain_02", "Why did this sample pass or fail visual inspection?", "text", _explanation_answer),
        Template("evidence_explanation", "explain_03", "Provide a grounded explanation using the defect label, location, and anomaly area when available.", "text", _explanation_answer),
        Template("structured_report", "json_01", "Return a concise quality-control report as valid JSON.", "json", _json_answer),
        Template("structured_report", "json_02", "Inspect this item and return only a JSON report with product, condition, defect, location, visual severity, action, and safety note.", "json", _json_answer),
        Template("structured_report", "json_03", "Generate a machine-readable inspection result in JSON. Do not add claims beyond the visual annotation.", "json", _json_answer),
        Template("technician_note", "tech_01", "Write a concise quality-control note for a technician.", "text", _technician_answer),
        Template("technician_note", "tech_02", "Summarize the inspection outcome and next action for a human reviewer.", "text", _technician_answer),
        Template("uncertainty", "uncertainty_01", "Can the mechanical root cause or safety impact be determined from this image alone?", "text", _uncertainty_answer),
    ]


def _selected_templates(record: DerivedImageRecord, config: VisaConfig) -> list[Template]:
    templates = _templates()
    count = (
        config.phase5_anomalous_instructions_per_image
        if record.condition is Condition.ANOMALOUS
        else config.phase5_normal_instructions_per_image
    )
    if count > len(templates):
        raise InstructionGenerationError(
            f"Requested {count} instructions per image but only {len(templates)} templates exist."
        )
    if record.condition is Condition.NORMAL:
        # Spread normal supervision across high-value families rather than taking adjacent templates.
        preferred_ids = ["binary_01", "product_01", "json_01", "explain_01", "tech_01", "uncertainty_01"]
        ordered = [next(t for t in templates if t.template_id == template_id) for template_id in preferred_ids]
        ordered.extend(t for t in templates if t.template_id not in preferred_ids)
        return ordered[:count]
    return templates[:count]


def _record_to_instruction(
    record: DerivedImageRecord,
    split: DatasetSplit,
    template: Template,
) -> InstructionRecord:
    image_path = record.image_path.as_posix()
    grounded_by = ["image_level_label", "product_category"]
    if record.condition is Condition.ANOMALOUS:
        grounded_by.extend(["defect_label", "segmentation_mask", "derived_spatial_features"])
    assistant_text = template.answer_builder(record)
    return InstructionRecord.model_validate(
        {
            "instruction_id": f"{record.image_id}__{template.template_id}",
            "image_id": record.image_id,
            "dataset_split": split.value,
            "task_family": template.family,
            "template_id": template.template_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": template.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ],
            "metadata": {
                "category": record.category,
                "condition": record.condition.value,
                "defect_type": record.defect_type,
                "location": record.nine_grid_location.value if record.nine_grid_location else None,
                "visual_severity": record.visual_severity.value,
                "answer_format": template.answer_format,
                "grounded_by": grounded_by,
            },
        }
    )


def _validate_json_answers(records: list[InstructionRecord]) -> list[str]:
    errors: list[str] = []
    for record in records:
        if record.metadata.answer_format != "json":
            continue
        text = record.messages[1].content[0].text or ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"{record.instruction_id}: invalid JSON answer: {exc}")
            continue
        required = {
            "product",
            "condition",
            "defect_type",
            "location",
            "visual_severity",
            "recommended_action",
            "safety_note",
        }
        missing = sorted(required - set(payload))
        if missing:
            errors.append(f"{record.instruction_id}: missing JSON fields {missing}")
    return errors


def generate_instructions(config: VisaConfig) -> Phase5Result:
    """Generate, validate, and write all Phase 5 instruction records."""

    config.phase5_output_root.mkdir(parents=True, exist_ok=True)
    config.phase5_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.phase5_error_path.parent.mkdir(parents=True, exist_ok=True)

    all_instructions: list[InstructionRecord] = []
    split_counts: dict[str, int] = {}
    source_image_counts: dict[str, int] = {}

    for split in DatasetSplit:
        source_path = config.phase4_split_root / f"{split.value}.jsonl"
        source_records = _read_split(source_path, split)
        source_image_counts[split.value] = len(source_records)
        generated = [
            _record_to_instruction(record, split, template)
            for record in source_records
            for template in _selected_templates(record, config)
        ]
        output_path = config.phase5_output_root / f"{split.value}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for instruction in generated:
                handle.write(instruction.model_dump_json(exclude_none=False) + "\n")
        all_instructions.extend(generated)
        split_counts[split.value] = len(generated)

    errors: list[str] = []
    instruction_ids = [record.instruction_id for record in all_instructions]
    duplicates = [value for value, count in Counter(instruction_ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate instruction IDs found: {duplicates[:10]}")

    image_splits: dict[str, set[str]] = defaultdict(set)
    image_task_ids: dict[str, set[str]] = defaultdict(set)
    for record in all_instructions:
        image_splits[record.image_id].add(record.dataset_split.value)
        image_task_ids[record.image_id].add(record.template_id)
    split_leaks = sorted(image_id for image_id, values in image_splits.items() if len(values) > 1)
    if split_leaks:
        errors.append(f"Images cross instruction splits: {split_leaks[:10]}")
    errors.extend(_validate_json_answers(all_instructions))

    family_counts = Counter(record.task_family for record in all_instructions)
    condition_counts = Counter(record.metadata.condition for record in all_instructions)
    category_counts = Counter(record.metadata.category for record in all_instructions)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": "visa",
        "source_version": config.version,
        "source_images": source_image_counts,
        "instruction_counts": split_counts,
        "total_instructions": len(all_instructions),
        "unique_images": len(image_splits),
        "instructions_per_image": {
            "normal": config.phase5_normal_instructions_per_image,
            "anomalous": config.phase5_anomalous_instructions_per_image,
        },
        "task_family_counts": dict(sorted(family_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "checks": {
            "all_source_images_preserved": len(image_splits) == config.expected_total_images,
            "instruction_ids_unique": not duplicates,
            "image_split_leakage_absent": not split_leaks,
            "json_answers_valid": not _validate_json_answers(all_instructions),
            "all_task_families_present": len(family_counts) == len({t.family for t in _templates()}),
        },
        "errors": errors,
        "warnings": [],
    }
    report["passed"] = all(report["checks"].values()) and not errors

    config.phase5_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with config.phase5_error_path.open("w", encoding="utf-8", newline="\n") as handle:
        for message in errors:
            handle.write(json.dumps({"error_type": "Phase5ValidationError", "message": message}) + "\n")

    if config.strict_phase5 and not report["passed"]:
        raise RuntimeError(
            "Phase 5 validation failed. See "
            f"{config.phase5_report_path} and {config.phase5_error_path}."
        )

    return Phase5Result(
        instructions=len(all_instructions),
        train_instructions=split_counts[DatasetSplit.TRAIN.value],
        validation_instructions=split_counts[DatasetSplit.VALIDATION.value],
        test_instructions=split_counts[DatasetSplit.TEST.value],
        unique_images=len(image_splits),
        errors=len(errors),
        warnings=0,
        output_directory=config.phase5_output_root,
        report_path=config.phase5_report_path,
        error_path=config.phase5_error_path,
    )
