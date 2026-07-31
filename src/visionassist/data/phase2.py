"""Phase 2 orchestration for VisA annotation and mask parsing."""

from __future__ import annotations

from visionassist.data.config import VisaConfig
from visionassist.data.parse_visa import Phase2Result, parse_visa


def run_phase2(config: VisaConfig) -> Phase2Result:
    """Build and validate the canonical VisA metadata manifest."""

    return parse_visa(config)
