"""Phase 4 orchestration."""

from visionassist.data.config import VisaConfig
from visionassist.data.split_visa import Phase4Result, split_visa


def run_phase4(config: VisaConfig) -> Phase4Result:
    """Generate deterministic supervised splits and leakage reports."""

    return split_visa(config)
