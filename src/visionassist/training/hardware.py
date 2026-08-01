"""Hardware inspection and adaptive profile selection for Colab training."""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HardwareInfo:
    """Serializable runtime hardware summary."""

    cuda_available: bool
    gpu_name: str | None
    total_vram_gib: float
    free_vram_gib: float
    bf16_supported: bool
    compute_capability: str | None
    cuda_version: str | None
    torch_version: str | None
    system_ram_gib: float
    free_disk_gib: float
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _nearest_existing_path(path: Path) -> Path:
    """Return the nearest existing path for filesystem capacity inspection.

    ``shutil.disk_usage`` requires its argument to exist on Windows. Training
    output directories are often created later, so hardware inspection walks
    upward until it finds an existing parent.
    """

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve(strict=False)

    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return Path.cwd().resolve()
        candidate = parent

    return candidate


def inspect_hardware(path_for_disk: Path | None = None) -> HardwareInfo:
    """Inspect CPU/disk everywhere and CUDA details when available."""

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu_name: str | None = None
    total_vram = 0.0
    free_vram = 0.0
    bf16 = False
    capability: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None

    if torch is not None:
        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
    if cuda_available and torch is not None:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_vram = free_bytes / (1024**3)
        bf16 = bool(torch.cuda.is_bf16_supported())
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"{major}.{minor}"

    system_ram = 0.0
    try:
        import psutil

        system_ram = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            system_ram = pages * page_size / (1024**3)
        except (AttributeError, ValueError, OSError):
            system_ram = 0.0

    requested_disk_path = path_for_disk or Path.cwd()
    disk_root = _nearest_existing_path(requested_disk_path)
    free_disk = shutil.disk_usage(disk_root).free / (1024**3)
    return HardwareInfo(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        total_vram_gib=round(total_vram, 3),
        free_vram_gib=round(free_vram, 3),
        bf16_supported=bf16,
        compute_capability=capability,
        cuda_version=cuda_version,
        torch_version=torch_version,
        system_ram_gib=round(system_ram, 3),
        free_disk_gib=round(free_disk, 3),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def select_profile(
    hardware: HardwareInfo,
) -> Literal["cpu_only", "low_vram", "standard_vram", "high_vram"]:
    """Choose a conservative profile from available VRAM."""

    if not hardware.cuda_available:
        return "cpu_only"
    if hardware.total_vram_gib < 16:
        return "low_vram"
    if hardware.total_vram_gib < 32:
        return "standard_vram"
    return "high_vram"


def write_hardware_report(path: Path, hardware: HardwareInfo) -> None:
    """Write hardware details as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hardware.to_dict(), indent=2), encoding="utf-8")
