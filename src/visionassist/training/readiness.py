"""Phase 6 training-readiness validation and completion reporting."""

from __future__ import annotations

import html
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from visionassist.data.config import VisaConfig
from visionassist.schemas.dataset import DatasetSplit
from visionassist.schemas.instruction import InstructionRecord
from visionassist.training.formatting import assistant_target, resolve_image_path, user_prompt

SUPPORTED_TASKS = {
    "binary_inspection",
    "product_identification",
    "defect_identification",
    "localization",
    "evidence_explanation",
    "structured_report",
    "technician_note",
    "uncertainty",
}
FORBIDDEN_ANSWER_MARKERS = (
    "mask_path",
    "segmentation_mask_path",
    "annotation_path",
    "image_anno.csv",
)
REQUIRED_JSON_FIELDS = {
    "product",
    "condition",
    "defect_type",
    "location",
    "visual_severity",
    "recommended_action",
    "safety_note",
}
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG"}


@dataclass(frozen=True)
class Phase6Result:
    """Counts and output paths produced by Phase 6."""

    instructions: int
    unique_images: int
    errors: int
    warnings: int
    report_path: Path
    error_path: Path
    statistics_path: Path
    gallery_path: Path
    processor_report_path: Path | None
    sequence_statistics_path: Path | None
    gallery_review_path: Path
    phase_complete: bool


def _percentiles(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        return {
            "minimum": 0,
            "median": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "maximum": 0,
            "mean": 0,
        }
    ordered = sorted(values)

    def nearest(percent: float) -> int | float:
        index = round((len(ordered) - 1) * percent)
        return ordered[index]

    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": nearest(0.90),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _read_records(path: Path) -> tuple[list[InstructionRecord], list[dict[str, Any]]]:
    records: list[InstructionRecord] = []
    errors: list[dict[str, Any]] = []
    if not path.is_file():
        return records, [{
            "path": str(path),
            "error_type": "MissingSplit",
            "message": "Instruction split does not exist.",
        }]
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(InstructionRecord.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                errors.append({
                    "path": str(path),
                    "row": row_number,
                    "error_type": "InstructionSchemaError",
                    "message": str(exc),
                })
    return records, errors


def _normalize_metadata_text(value: str | None) -> str | None:
    """Normalize human-readable metadata for semantic equality checks.

    Phase 5 intentionally renders comma-separated defect labels with a space after
    each comma for readability, while canonical metadata preserves the source
    comma formatting. These strings are semantically equivalent.
    """
    if value is None:
        return None
    normalized_parts = [
        " ".join(part.replace("_", " ").split())
        for part in value.split(",")
    ]
    return ", ".join(part for part in normalized_parts if part)


def _json_values_match(field: str, actual: Any, expected: Any) -> bool:
    """Compare JSON fields using field-appropriate normalization."""
    if field in {"defect_type", "location"}:
        actual_value = _normalize_metadata_text(actual) if isinstance(actual, str) else actual
        expected_value = _normalize_metadata_text(expected) if isinstance(expected, str) else expected
        return actual_value == expected_value
    return actual == expected


def _validate_json_target(record: InstructionRecord, answer: str) -> list[str]:
    if record.metadata.answer_format != "json":
        return []
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON target: {exc}"]
    if not isinstance(payload, dict):
        return ["structured-report target must be a JSON object"]
    missing = sorted(REQUIRED_JSON_FIELDS - set(payload))
    problems = [f"missing JSON fields: {missing}"] if missing else []
    expected_condition = "defective" if record.metadata.condition == "anomalous" else "normal"
    expected_location = (
        record.metadata.location.replace("_", " ") if record.metadata.location else None
    )
    expected_defect = (
        record.metadata.defect_type.replace("_", " ")
        if record.metadata.defect_type
        else None
    )
    comparisons = {
        "condition": expected_condition,
        "product": record.metadata.category,
        "defect_type": expected_defect,
        "location": expected_location,
        "visual_severity": record.metadata.visual_severity,
    }
    for field, expected in comparisons.items():
        actual = payload.get(field)
        if not _json_values_match(field, actual, expected):
            problems.append(
                f"JSON {field} {actual!r} does not match metadata {expected!r}"
            )
    return problems


def _validate_record(
    record: InstructionRecord,
    split: DatasetSplit,
    project_root: Path,
    image_cache: dict[Path, tuple[dict[str, Any] | None, str | None]],
) -> tuple[list[str], dict[str, Any] | None]:
    problems: list[str] = []
    if record.dataset_split is not split:
        problems.append(
            f"record split {record.dataset_split.value!r} does not match file {split.value!r}"
        )
    if record.task_family not in SUPPORTED_TASKS:
        problems.append(f"unsupported task family: {record.task_family}")

    # The Pydantic model enforces exactly one user message, one assistant message,
    # one image item, one user text item, and one assistant text target.
    prompt = user_prompt(record).strip()
    answer = assistant_target(record).strip()
    if not prompt:
        problems.append("empty user prompt")
    if not answer:
        problems.append("empty assistant target")
    answer_lower = answer.lower()
    for marker in FORBIDDEN_ANSWER_MARKERS:
        if marker in answer_lower:
            problems.append(f"answer exposes privileged marker: {marker}")
    problems.extend(_validate_json_target(record, answer))

    image_info: dict[str, Any] | None = None
    try:
        image_path = resolve_image_path(record, project_root)
        cached = image_cache.get(image_path)
        if cached is None:
            if not image_path.is_file():
                cached = (None, f"image does not exist: {image_path}")
            elif image_path.stat().st_size == 0:
                cached = (None, f"image is zero bytes: {image_path}")
            else:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        width, height = image.size
                        image_format = image.format
                    if image_format not in SUPPORTED_IMAGE_FORMATS:
                        cached = (
                            None,
                            f"unsupported image format {image_format!r}: {image_path}",
                        )
                    else:
                        cached = ({
                            "path": image_path,
                            "width": width,
                            "height": height,
                            "format": image_format,
                            "size_bytes": image_path.stat().st_size,
                        }, None)
                except (OSError, UnidentifiedImageError) as exc:
                    cached = (None, f"image validation failed: {exc}")
            image_cache[image_path] = cached
        image_info, image_error = cached
        if image_error is not None:
            problems.append(image_error)
    except ValueError as exc:
        problems.append(f"image validation failed: {exc}")
    return problems, image_info


def _duplicate_summary(values: Sequence[str], limit: int = 25) -> dict[str, Any]:
    counts = Counter(values)
    repeated = [(value, count) for value, count in counts.items() if count > 1]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    return {
        "unique_count": len(counts),
        "repeated_value_count": len(repeated),
        "records_in_repeated_values": sum(count for _, count in repeated),
        "maximum_frequency": repeated[0][1] if repeated else 1,
        "most_frequent": [
            {"value": value, "count": count} for value, count in repeated[:limit]
        ],
    }


def _quality_statistics(
    records: Sequence[InstructionRecord],
    image_dimensions: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    prompts = [user_prompt(record) for record in records]
    answers = [assistant_target(record) for record in records]
    prompt_words = [len(value.split()) for value in prompts]
    answer_words = [len(value.split()) for value in answers]
    widths = [value[0] for value in image_dimensions.values()]
    heights = [value[1] for value in image_dimensions.values()]
    instructions_per_image = Counter(record.image_id for record in records)
    prompt_template_pairs = {(user_prompt(r), r.template_id) for r in records}

    def optional_counter(attribute: str) -> dict[str, int]:
        values = [getattr(record.metadata, attribute) for record in records]
        return dict(sorted(Counter(str(value) for value in values if value is not None).items()))

    return {
        "schema_version": "1.1",
        "instruction_count": len(records),
        "unique_images": len(instructions_per_image),
        "unique_prompts": len(set(prompts)),
        "unique_answers": len(set(answers)),
        "unique_prompt_template_combinations": len(prompt_template_pairs),
        "prompt_word_counts": _percentiles(prompt_words),
        "answer_word_counts": _percentiles(answer_words),
        "image_widths": _percentiles(widths),
        "image_heights": _percentiles(heights),
        "instructions_per_image": _percentiles(list(instructions_per_image.values())),
        "task_family_counts": dict(sorted(Counter(r.task_family for r in records).items())),
        "split_counts": dict(sorted(Counter(r.dataset_split.value for r in records).items())),
        "condition_counts": dict(sorted(Counter(r.metadata.condition for r in records).items())),
        "category_counts": dict(sorted(Counter(r.metadata.category for r in records).items())),
        "defect_label_counts": optional_counter("defect_type"),
        "location_counts": optional_counter("location"),
        "severity_counts": optional_counter("visual_severity"),
        "template_counts": dict(sorted(Counter(r.template_id for r in records).items())),
        "repeated_prompts": _duplicate_summary(prompts),
        "repeated_answers": _duplicate_summary(answers),
    }


def _stratified_sample(
    records: Sequence[InstructionRecord], sample_size: int, seed: int
) -> list[InstructionRecord]:
    """Return a deterministic round-robin sample across task/category/condition strata."""

    grouped: dict[tuple[str, str, str], list[InstructionRecord]] = defaultdict(list)
    for record in records:
        key = (record.task_family, record.metadata.category, record.metadata.condition)
        grouped[key].append(record)
    rng = random.Random(seed)
    for values in grouped.values():
        values.sort(key=lambda item: item.instruction_id)
        rng.shuffle(values)

    keys = sorted(grouped)
    selected: list[InstructionRecord] = []
    index = 0
    while keys and len(selected) < min(sample_size, len(records)):
        key = keys[index % len(keys)]
        values = grouped[key]
        if values:
            selected.append(values.pop())
        if not values:
            keys.remove(key)
            if not keys:
                break
            index %= len(keys)
        else:
            index += 1
    return selected


def _gallery_sample(
    records: Sequence[InstructionRecord],
    per_family: int,
    seed: int,
) -> list[InstructionRecord]:
    """Select a deterministic balanced gallery with every task family represented."""

    selected: list[InstructionRecord] = []
    for family_index, family in enumerate(sorted(SUPPORTED_TASKS)):
        family_records = [record for record in records if record.task_family == family]
        if not family_records:
            continue
        family_sample = _stratified_sample(
            family_records,
            min(per_family, len(family_records)),
            seed + family_index,
        )
        selected.extend(family_sample)
    return selected


def _gallery(
    records: Sequence[InstructionRecord],
    project_root: Path,
    output_path: Path,
    per_family: int,
    seed: int,
) -> dict[str, Any]:
    selected = _gallery_sample(records, per_family=per_family, seed=seed)
    cards: list[str] = []
    coverage = {
        "task_families": Counter(),
        "categories": Counter(),
        "conditions": Counter(),
        "splits": Counter(),
    }
    for record in selected:
        coverage["task_families"][record.task_family] += 1
        coverage["categories"][record.metadata.category] += 1
        coverage["conditions"][record.metadata.condition] += 1
        coverage["splits"][record.dataset_split.value] += 1
        image_path = resolve_image_path(record, project_root)
        relative = Path(os.path.relpath(image_path, output_path.parent))
        cards.append(
            "<article class='card'>"
            f"<img loading='lazy' src='{html.escape(relative.as_posix())}' "
            f"alt='{html.escape(record.image_id)}'>"
            f"<h2>{html.escape(record.task_family)}</h2>"
            f"<p><b>ID:</b> {html.escape(record.instruction_id)}</p>"
            f"<p><b>Image ID:</b> {html.escape(record.image_id)}</p>"
            f"<p><b>Split:</b> {html.escape(record.dataset_split.value)}</p>"
            f"<p><b>Category:</b> {html.escape(record.metadata.category)}</p>"
            f"<p><b>Condition:</b> {html.escape(record.metadata.condition)}</p>"
            f"<p><b>Defect:</b> {html.escape(str(record.metadata.defect_type))}</p>"
            f"<p><b>Location:</b> {html.escape(str(record.metadata.location))}</p>"
            f"<p><b>Severity:</b> {html.escape(record.metadata.visual_severity)}</p>"
            f"<p><b>Prompt:</b> {html.escape(user_prompt(record))}</p>"
            f"<p><b>Target:</b> {html.escape(assistant_target(record))}</p>"
            "</article>"
        )
    summary = "".join(
        f"<li><b>{html.escape(name)}:</b> {html.escape(str(dict(sorted(values.items()))))}</li>"
        for name, values in coverage.items()
    )
    style = """
<style>
body { font-family: system-ui; margin: 2rem; background: #f5f5f5; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }
.card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 1px 5px #bbb; }
.card img { width: 100%; height: 260px; object-fit: contain; background: #111; }
.card p { overflow-wrap: anywhere; }
.summary { background: white; padding: 1rem; margin-bottom: 1rem; border-radius: 10px; }
</style>
"""
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>VisionAssist Phase 6 Sample Gallery</title>" + style
        + "</head><body><h1>VisionAssist training-readiness sample gallery</h1>"
        + f"<section class='summary'><h2>Coverage</h2><ul>{summary}</ul></section>"
        + "<div class='grid'>" + "".join(cards) + "</div></body></html>"
    )
    output_path.write_text(document, encoding="utf-8")
    return {
        "sample_count": len(selected),
        **{name: dict(sorted(values.items())) for name, values in coverage.items()},
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("<|im_end|>", " ").split()).strip().lower()


def _processor_analysis(
    records: Sequence[InstructionRecord],
    config: VisaConfig,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install training dependencies with `uv sync --extra training`."
        ) from exc

    from visionassist.training.collator import QwenAssistantOnlyCollator

    processor = AutoProcessor.from_pretrained(
        config.phase6_processor_model_id,
        trust_remote_code=config.phase6_trust_remote_code,
    )
    sample = _stratified_sample(records, config.phase6_processor_sample_size, config.phase6_seed)
    collator = QwenAssistantOnlyCollator(
        processor,
        project_root,
        max_length=config.phase6_max_sequence_length,
    )
    token_lengths: list[int] = []
    target_lengths: list[int] = []
    raw_visual_tokens: list[int] = []
    estimated_merged_visual_tokens: list[int] = []
    masking_failures: list[dict[str, Any]] = []
    exceeded: list[dict[str, Any]] = []
    per_task_lengths: dict[str, list[int]] = defaultdict(list)
    merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 2))
    sequence_limit = config.phase6_analysis_sequence_limit

    for record in sample:
        encoded = collator._encode(record)
        input_ids = encoded["input_ids"][0]
        labels = encoded["labels"][0]
        length = int(input_ids.numel())
        target_length = int((labels != -100).sum().item())
        token_lengths.append(length)
        target_lengths.append(target_length)
        per_task_lengths[record.task_family].append(length)
        if length > sequence_limit:
            exceeded.append({
                "instruction_id": record.instruction_id,
                "task_family": record.task_family,
                "sequence_length": length,
                "limit": sequence_limit,
            })

        prefix_length = int(encoded["assistant_prefix_length"])
        prefix_masked = bool((labels[:prefix_length] == -100).all().item())
        target_unmasked = bool((labels[prefix_length:] != -100).any().item())
        trainable_ids = labels[labels != -100].tolist()
        decoded = processor.tokenizer.decode(trainable_ids, skip_special_tokens=True)
        target = assistant_target(record)
        decoded_matches = _normalize_text(target) in _normalize_text(decoded)
        if not (prefix_masked and target_unmasked and decoded_matches):
            masking_failures.append({
                "instruction_id": record.instruction_id,
                "prefix_masked": prefix_masked,
                "target_unmasked": target_unmasked,
                "decoded_matches_target": decoded_matches,
                "decoded_trainable_labels": decoded[:500],
                "expected_target": target[:500],
            })

        if "image_grid_thw" in encoded:
            grid = encoded["image_grid_thw"]
            for row in grid.reshape(-1, 3).tolist():
                raw = int(row[0] * row[1] * row[2])
                raw_visual_tokens.append(raw)
                estimated_merged_visual_tokens.append(
                    max(1, raw // max(1, merge_size * merge_size))
                )

    batch_size = min(2, len(sample))
    batch = collator(sample[:batch_size]) if batch_size else {}
    padding_masked = True
    if batch:
        padding_positions = batch["attention_mask"] == 0
        if bool(padding_positions.any().item()):
            padding_masked = bool((batch["labels"][padding_positions] == -100).all().item())

    processor_report = {
        "model_id": config.phase6_processor_model_id,
        "sampling": {
            "method": "deterministic_round_robin_task_category_condition",
            "requested_size": config.phase6_processor_sample_size,
            "actual_size": len(sample),
            "seed": config.phase6_seed,
            "task_family_counts": dict(sorted(Counter(r.task_family for r in sample).items())),
            "category_counts": dict(sorted(Counter(r.metadata.category for r in sample).items())),
            "condition_counts": dict(sorted(Counter(r.metadata.condition for r in sample).items())),
        },
        "assistant_only_masking": {
            "records_checked": len(sample),
            "failures": len(masking_failures),
            "padding_tokens_masked": padding_masked,
            "passed": not masking_failures and padding_masked,
            "failure_examples": masking_failures[:20],
        },
        "batch_smoke_test": {
            "batch_size": batch_size,
            "input_shape": list(batch["input_ids"].shape) if batch else None,
            "label_shape": list(batch["labels"].shape) if batch else None,
            "has_pixel_values": "pixel_values" in batch,
            "has_image_grid_thw": "image_grid_thw" in batch,
        },
    }
    processor_report["passed"] = (
        bool(sample)
        and not masking_failures
        and padding_masked
        and all(length > 0 for length in target_lengths)
        and bool(batch)
    )

    sequence_report = {
        "schema_version": "1.0",
        "model_id": config.phase6_processor_model_id,
        "sample_size": len(sample),
        "sampling_method": "deterministic stratified round-robin",
        "sequence_limit": sequence_limit,
        "token_lengths": _percentiles(token_lengths),
        "assistant_target_lengths": _percentiles(target_lengths),
        "raw_visual_grid_tokens": _percentiles(raw_visual_tokens),
        "estimated_post_merge_visual_tokens": _percentiles(estimated_merged_visual_tokens),
        "vision_merge_size": merge_size,
        "per_task_token_lengths": {
            key: _percentiles(values) for key, values in sorted(per_task_lengths.items())
        },
        "examples_exceeding_limit": exceeded,
        "exceeding_limit_count": len(exceeded),
    }
    return processor_report, sequence_report


def _write_gallery_review(
    path: Path,
    *,
    approved: bool,
    reviewer: str | None,
    gallery_path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if approved:
        payload = {
            "schema_version": "1.0",
            "gallery": str(gallery_path),
            "review_status": "approved",
            "reviewed_by": reviewer or "unspecified reviewer",
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "statement": (
                "The reviewer inspected the gallery images, prompts, targets, and metadata "
                "and approved them for the next development phase."
            ),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("review_status") == "approved":
                return payload
        except (json.JSONDecodeError, OSError):
            pass
    payload = {
        "schema_version": "1.0",
        "gallery": str(gallery_path),
        "review_status": "pending",
        "reviewed_by": None,
        "reviewed_at_utc": None,
        "statement": "Open the gallery, inspect the visual/target alignment, then rerun with --approve-gallery.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_training_readiness(
    config: VisaConfig,
    *,
    project_root: Path = Path.cwd(),
    processor_smoke_test: bool = False,
    approve_gallery: bool = False,
    reviewer: str | None = None,
) -> Phase6Result:
    """Validate Phase 5 records and produce complete Phase 6 evidence."""

    project_root = project_root.resolve()
    config.phase6_report_root.mkdir(parents=True, exist_ok=True)
    all_records: list[InstructionRecord] = []
    errors: list[dict[str, Any]] = []
    image_dimensions: dict[str, tuple[int, int]] = {}
    image_cache: dict[Path, tuple[dict[str, Any] | None, str | None]] = {}
    image_splits: dict[str, set[str]] = defaultdict(set)
    instruction_ids: set[str] = set()
    duplicate_instruction_ids: list[str] = []

    for split in DatasetSplit:
        path = config.phase5_output_root / f"{split.value}.jsonl"
        records, schema_errors = _read_records(path)
        errors.extend(schema_errors)
        for record in records:
            if record.instruction_id in instruction_ids:
                duplicate_instruction_ids.append(record.instruction_id)
            instruction_ids.add(record.instruction_id)
            image_splits[record.image_id].add(split.value)
            problems, image_info = _validate_record(record, split, project_root, image_cache)
            for problem in problems:
                errors.append({
                    "split": split.value,
                    "instruction_id": record.instruction_id,
                    "image_id": record.image_id,
                    "error_type": "TrainingReadinessError",
                    "message": problem,
                })
            if image_info is not None:
                image_dimensions[record.image_id] = (
                    int(image_info["width"]), int(image_info["height"])
                )
        all_records.extend(records)

    split_leaks = sorted(
        image_id for image_id, values in image_splits.items() if len(values) > 1
    )
    if split_leaks:
        errors.append({
            "error_type": "SplitLeakage",
            "message": f"Images appear in multiple splits: {split_leaks[:20]}",
        })
    if duplicate_instruction_ids:
        errors.append({
            "error_type": "DuplicateInstructionId",
            "message": f"Duplicate instruction IDs: {sorted(set(duplicate_instruction_ids))[:20]}",
        })

    statistics_report = _quality_statistics(all_records, image_dimensions)
    config.phase6_statistics_path.write_text(
        json.dumps(statistics_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gallery_coverage: dict[str, Any] = {}
    if all_records:
        gallery_coverage = _gallery(
            all_records,
            project_root,
            config.phase6_gallery_path,
            config.phase6_gallery_samples_per_family,
            config.phase6_seed,
        )
    gallery_review = _write_gallery_review(
        config.phase6_gallery_review_path,
        approved=approve_gallery,
        reviewer=reviewer,
        gallery_path=config.phase6_gallery_path,
    )

    processor_report: dict[str, Any] | None = None
    sequence_report: dict[str, Any] | None = None
    if processor_smoke_test and all_records:
        processor_report, sequence_report = _processor_analysis(
            all_records, config, project_root
        )
        config.phase6_processor_report_path.write_text(
            json.dumps(processor_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        config.phase6_sequence_statistics_path.write_text(
            json.dumps(sequence_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not processor_report["passed"]:
            errors.append({
                "error_type": "ProcessorSmokeTest",
                "message": "Processor or assistant-only masking test failed.",
            })

    semantic_checks = {
        "exactly_one_user_and_assistant_message": True,
        "exactly_one_user_image_and_text": True,
        "assistant_targets_non_empty": all(bool(assistant_target(r).strip()) for r in all_records),
        "structured_json_valid_and_consistent": not any(
            _validate_json_target(r, assistant_target(r)) for r in all_records
        ),
        "privileged_answer_leakage_absent": not any(
            marker in assistant_target(r).lower()
            for r in all_records
            for marker in FORBIDDEN_ANSWER_MARKERS
        ),
    }
    automated_checks = {
        "expected_instruction_count": len(all_records) == config.phase6_expected_instructions,
        "expected_unique_images": len(image_splits) == config.expected_total_images,
        "all_images_readable": len(image_dimensions) == len(image_splits),
        "instruction_ids_unique": not duplicate_instruction_ids,
        "image_split_leakage_absent": not split_leaks,
        "all_task_families_present": SUPPORTED_TASKS.issubset({r.task_family for r in all_records}),
        "quality_statistics_complete": all(
            key in statistics_report
            for key in (
                "defect_label_counts", "location_counts", "severity_counts",
                "repeated_prompts", "repeated_answers", "instructions_per_image",
                "unique_prompt_template_combinations",
            )
        ),
        **semantic_checks,
        "processor_smoke_test": processor_report["passed"] if processor_report else None,
        "assistant_only_masking_verified": (
            processor_report["assistant_only_masking"]["passed"]
            if processor_report else None
        ),
        "sequence_statistics_available": sequence_report is not None,
    }
    automated_passed = (
        all(value for value in automated_checks.values() if value is not None) and not errors
    )
    gallery_approved = gallery_review.get("review_status") == "approved"
    phase_complete = automated_passed and processor_report is not None and gallery_approved
    report = {
        "schema_version": "1.1",
        "dataset": "visa",
        "source_version": config.version,
        "instructions": len(all_records),
        "unique_images": len(image_splits),
        "readable_images": len(image_dimensions),
        "errors": len(errors),
        "warnings": 0,
        "automated_checks": automated_checks,
        "automated_passed": automated_passed,
        "processor_smoke_test_requested": processor_smoke_test,
        "gallery": {
            "coverage": gallery_coverage,
            "review_status": gallery_review.get("review_status"),
            "reviewed_by": gallery_review.get("reviewed_by"),
        },
        "phase_complete": phase_complete,
        "passed": automated_passed,
        "outputs": {
            "statistics": str(config.phase6_statistics_path),
            "gallery": str(config.phase6_gallery_path),
            "gallery_review": str(config.phase6_gallery_review_path),
            "processor_report": str(config.phase6_processor_report_path) if processor_report else None,
            "sequence_statistics": (
                str(config.phase6_sequence_statistics_path) if sequence_report else None
            ),
        },
    }
    config.phase6_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with config.phase6_error_path.open("w", encoding="utf-8", newline="\n") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    if config.strict_phase6 and not automated_passed:
        raise RuntimeError(
            "Phase 6 validation failed. See "
            f"{config.phase6_report_path} and {config.phase6_error_path}."
        )

    return Phase6Result(
        instructions=len(all_records),
        unique_images=len(image_splits),
        errors=len(errors),
        warnings=0,
        report_path=config.phase6_report_path,
        error_path=config.phase6_error_path,
        statistics_path=config.phase6_statistics_path,
        gallery_path=config.phase6_gallery_path,
        processor_report_path=config.phase6_processor_report_path if processor_report else None,
        sequence_statistics_path=(
            config.phase6_sequence_statistics_path if sequence_report else None
        ),
        gallery_review_path=config.phase6_gallery_review_path,
        phase_complete=phase_complete,
    )
