import io
import tarfile
from pathlib import Path

import pytest

from visionassist.data.extract import safe_extract_tar


def test_safe_extract_tar_extracts_regular_file(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("VisA/readme.txt")
        payload = b"ok"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    extracted = safe_extract_tar(archive_path, tmp_path / "out")
    assert (extracted / "readme.txt").read_bytes() == b"ok"


def test_safe_extract_tar_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"bad"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract_tar(archive_path, tmp_path / "out")
