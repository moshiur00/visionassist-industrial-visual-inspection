from pathlib import Path

from visionassist.data.download import download_file


def test_download_file_skips_existing_without_remote_metadata(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "archive.tar"
    destination.write_bytes(b"existing")
    monkeypatch.setattr(
        "visionassist.data.download._remote_metadata",
        lambda _url, _timeout: (None, None, None),
    )
    result = download_file("https://example.invalid/archive.tar", destination)
    assert result.skipped is True
    assert destination.read_bytes() == b"existing"
