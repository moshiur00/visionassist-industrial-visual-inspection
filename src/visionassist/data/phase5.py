"""Phase 5 orchestration."""

from __future__ import annotations

from visionassist.data.config import VisaConfig
from visionassist.data.generate_instructions import Phase5Result, generate_instructions


def run_phase5(config: VisaConfig) -> Phase5Result:
    """Generate grounded multimodal instruction data from Phase 4 splits."""

    return generate_instructions(config)
