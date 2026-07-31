"""Resumable downloader for the official VisA archive."""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DownloadResult:
    """Metadata about one download attempt."""

    path: Path
    downloaded_bytes: int
    resumed: bool
    skipped: bool
    content_length: int | None
    etag: str | None
    last_modified: str | None


def _remote_metadata(url: str, timeout: int) -> tuple[int | None, str | None, str | None]:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return (
                int(length) if length is not None else None,
                response.headers.get("ETag"),
                response.headers.get("Last-Modified"),
            )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        # Some object stores/proxies reject HEAD. GET still works, so metadata is optional.
        return None, None, None


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
    force: bool = False,
    progress: Callable[[int, int | None], None] | None = None,
) -> DownloadResult:
    """Download a file with best-effort HTTP range resume support.

    A partial file is stored as ``<destination>.part`` and atomically renamed only
    after the transfer finishes.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    content_length, etag, last_modified = _remote_metadata(url, timeout)

    if destination.is_file() and not force:
        size = destination.stat().st_size
        if content_length is None or size == content_length:
            return DownloadResult(destination, 0, False, True, content_length, etag, last_modified)

    partial = destination.with_suffix(destination.suffix + ".part")
    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "VisionAssist/0.1"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and content_length is not None and start == content_length:
            partial.replace(destination)
            return DownloadResult(destination, 0, True, False, content_length, etag, last_modified)
        raise RuntimeError(f"Download failed with HTTP {exc.code}: {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    with response:
        status = getattr(response, "status", 200)
        resumed = start > 0 and status == 206
        if start > 0 and not resumed:
            # Server ignored Range; restart safely instead of appending duplicate bytes.
            start = 0
            partial.unlink(missing_ok=True)

        mode = "ab" if resumed else "wb"
        transferred = 0
        with partial.open(mode) as target:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                target.write(chunk)
                transferred += len(chunk)
                if progress is not None:
                    progress(start + transferred, content_length)
            target.flush()

    final_size = partial.stat().st_size
    if content_length is not None and final_size != content_length:
        raise RuntimeError(
            f"Incomplete download: expected {content_length} bytes, received {final_size}. "
            "Run the command again to resume."
        )

    shutil.move(str(partial), str(destination))
    return DownloadResult(
        destination,
        transferred,
        resumed,
        False,
        content_length,
        etag,
        last_modified,
    )
