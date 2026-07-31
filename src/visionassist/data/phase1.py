"""Orchestrate Phase 1: acquisition, verification, licensing, and dataset card."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from visionassist.data.audit_visa import AuditResult, audit_visa
from visionassist.data.checksum import sha256_file
from visionassist.data.config import VisaConfig
from visionassist.data.dataset_card import write_dataset_card
from visionassist.data.download import DownloadResult, download_file
from visionassist.data.extract import safe_extract_tar
from visionassist.data.license_report import write_license_report


@dataclass(frozen=True)
class Phase1Result:
    """Artifacts produced by a complete Phase 1 run."""

    download: DownloadResult
    archive_sha256: str
    audit: AuditResult
    receipt_path: Path
    license_report_path: Path
    dataset_card_path: Path


def _place_extracted_dataset(extracted_root: Path, configured_root: Path) -> None:
    configured_root.parent.mkdir(parents=True, exist_ok=True)
    if configured_root.resolve() == extracted_root.resolve():
        return
    if configured_root.exists():
        return
    shutil.move(str(extracted_root), str(configured_root))


def run_phase1(
    config: VisaConfig,
    *,
    force_download: bool = False,
    force_extract: bool = False,
    progress: object | None = None,
) -> Phase1Result:
    """Execute all Phase 1 steps and fail on structural expectation mismatches."""

    progress_callback = progress if callable(progress) else None
    download = download_file(
        str(config.download_url),
        config.archive_path,
        timeout=config.request_timeout_seconds,
        chunk_size=config.chunk_size_bytes,
        force=force_download,
        progress=progress_callback,
    )
    archive_sha256 = sha256_file(config.archive_path, config.chunk_size_bytes)

    if force_extract and config.raw_root.exists():
        shutil.rmtree(config.raw_root)

    if not config.raw_root.is_dir():
        extraction_parent = config.raw_root.parent / ".visa_extract"
        if extraction_parent.exists() and force_extract:
            shutil.rmtree(extraction_parent)
        extracted_root = safe_extract_tar(
            config.archive_path,
            extraction_parent,
            force=force_extract,
        )
        _place_extracted_dataset(extracted_root, config.raw_root)
        if extraction_parent.exists() and extraction_parent != config.raw_root:
            shutil.rmtree(extraction_parent, ignore_errors=True)

    audit = audit_visa(config)
    summary = json.loads(audit.summary_path.read_text(encoding="utf-8"))

    observed_total = int(summary["total_records"])
    observed_conditions = summary["condition_counts"]
    observed_normal = int(observed_conditions.get("normal", 0))
    observed_anomalous = int(observed_conditions.get("anomalous", 0))
    mismatches: list[str] = []
    if observed_total != config.expected_total_images:
        mismatches.append(f"total images: expected {config.expected_total_images}, got {observed_total}")
    if observed_normal != config.expected_normal_images:
        mismatches.append(f"normal images: expected {config.expected_normal_images}, got {observed_normal}")
    if observed_anomalous != config.expected_anomalous_images:
        mismatches.append(
            f"anomalous images: expected {config.expected_anomalous_images}, got {observed_anomalous}"
        )
    if summary.get("missing_expected_categories"):
        mismatches.append(f"missing categories: {summary['missing_expected_categories']}")

    receipt = {
        "dataset": config.dataset_name,
        "version": config.version,
        "source_url": str(config.download_url),
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "archive_path": config.archive_path.as_posix(),
        "archive_size_bytes": config.archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "remote_content_length": download.content_length,
        "remote_etag": download.etag,
        "remote_last_modified": download.last_modified,
        "download_resumed": download.resumed,
        "download_skipped": download.skipped,
        "structural_verification": {
            "passed": not mismatches and audit.error_count == 0,
            "mismatches": mismatches,
            "audit_error_count": audit.error_count,
        },
        "checksum_note": (
            "The official source does not publish a SHA-256 value in its registry entry. "
            "This digest fingerprints the locally acquired official archive and enables "
            "repeat-run integrity checks; structural counts provide an additional verification layer."
        ),
    }
    config.archive_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    config.archive_receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    license_path = write_license_report(config.license_report_path, version=config.version)
    card_path = write_dataset_card(config.dataset_card_path, audit.summary_path, version=config.version)

    if mismatches or audit.error_count:
        details = "; ".join(mismatches) or f"audit errors: {audit.error_count}"
        raise RuntimeError(
            "VisA acquisition completed, but verification failed. "
            f"Review {audit.summary_path}: {details}"
        )

    return Phase1Result(download, archive_sha256, audit, config.archive_receipt_path, license_path, card_path)
