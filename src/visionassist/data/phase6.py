"""Phase 6 orchestration."""

from __future__ import annotations

from pathlib import Path

from visionassist.data.config import VisaConfig
from visionassist.training.readiness import Phase6Result, validate_training_readiness


def run_phase6(
    config: VisaConfig,
    *,
    project_root: Path = Path.cwd(),
    processor_smoke_test: bool = False,
    approve_gallery: bool = False,
    reviewer: str | None = None,
) -> Phase6Result:
    """Run training-readiness validation and optional Qwen processor checks."""

    return validate_training_readiness(
        config,
        project_root=project_root,
        processor_smoke_test=processor_smoke_test,
        approve_gallery=approve_gallery,
        reviewer=reviewer,
    )
