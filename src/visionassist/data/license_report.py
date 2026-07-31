"""Generate a source-grounded VisA license report."""

from __future__ import annotations

from datetime import date
from pathlib import Path


OFFICIAL_DATASET_URL = "https://registry.opendata.aws/visa/"
OFFICIAL_REPOSITORY_URL = "https://github.com/amazon-science/spot-diff"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def write_license_report(path: Path, *, version: str, accessed_on: date | None = None) -> Path:
    """Write the Phase 1 license report."""

    accessed = accessed_on or date.today()
    text = f"""# VisA Dataset License Report

## Dataset identity

- **Dataset:** Visual Anomaly (VisA)
- **Release used:** {version}
- **Maintainer/source:** Amazon Science / AWS Registry of Open Data
- **Official dataset registry:** {OFFICIAL_DATASET_URL}
- **Official project repository:** {OFFICIAL_REPOSITORY_URL}
- **Access date:** {accessed.isoformat()}

## License

The official project repository and AWS registry identify the dataset license as
**Creative Commons Attribution 4.0 International (CC BY 4.0)**:
{LICENSE_URL}

CC BY 4.0 permits sharing and adaptation, including commercial use, provided
appropriate attribution is supplied and changes are indicated. This report is a
project record, not legal advice. The repository's own source code remains under
its separately declared software license.

## Attribution requirement

Any publication, model card, dataset card, demo, or redistributed derivative
must credit the VisA creators, identify the source, link or refer to CC BY 4.0,
and indicate material modifications.

## Recommended citation

```bibtex
@article{{zou2022spot,
  title={{SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation}},
  author={{Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar}},
  journal={{arXiv preprint arXiv:2207.14315}},
  year={{2022}}
}}
```

## Project policy

- Raw VisA files are not committed to Git.
- The dataset is downloaded only from the official AWS-hosted object.
- Derived manifests retain source and release metadata.
- Any future combination with another dataset must receive a separate license
  compatibility review.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
