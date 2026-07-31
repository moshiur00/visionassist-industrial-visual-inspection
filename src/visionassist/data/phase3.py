"""Phase 3 orchestration."""

from __future__ import annotations

from visionassist.data.config import VisaConfig
from visionassist.data.derive_features import Phase3Result, derive_visa_features


def run_phase3(config: VisaConfig) -> Phase3Result:
    """Derive and validate spatial features from all canonical VisA records."""

    return derive_visa_features(config)
