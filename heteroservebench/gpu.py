"""NVIDIA GPU discovery helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GpuInfo:
    """One NVIDIA GPU reported by nvidia-smi."""

    index: int
    name: str
    uuid: Optional[str]
    memory_total_mb: Optional[int]
    driver_version: Optional[str]
    cuda_version: Optional[str]


def parse_nvidia_smi_csv(text: str, driver_version: Optional[str] = None, cuda_version: Optional[str] = None) -> list[GpuInfo]:
    """Parse a nvidia-smi CSV query response."""
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 4:
            continue
        memory_text = parts[3].replace("MiB", "").strip()
        try:
            memory_total_mb = int(memory_text)
        except ValueError:
            memory_total_mb = None
        gpus.append(
            GpuInfo(
                index=int(parts[0]),
                name=parts[1],
                uuid=parts[2] or None,
                memory_total_mb=memory_total_mb,
                driver_version=driver_version,
                cuda_version=cuda_version,
            )
        )
    return gpus


def _driver_cuda_versions() -> tuple[Optional[str], Optional[str]]:
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None, None
    driver = None
    cuda = None
    for token in completed.stdout.replace("|", " ").split():
        if token.count(".") >= 1:
            continue
    marker_driver = "Driver Version:"
    marker_cuda = "CUDA Version:"
    if marker_driver in completed.stdout:
        driver = completed.stdout.split(marker_driver, 1)[1].split()[0]
    if marker_cuda in completed.stdout:
        cuda = completed.stdout.split(marker_cuda, 1)[1].split()[0]
    return driver, cuda


def discover_nvidia_gpus() -> dict:
    """Discover NVIDIA GPUs with nvidia-smi without requiring CUDA imports."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "error": "nvidia-smi not found",
            "visible_gpu_count": 0,
            "selected_cuda_device": visible,
            "gpus": [],
        }
    driver, cuda = _driver_cuda_versions()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpus = parse_nvidia_smi_csv(completed.stdout, driver_version=driver, cuda_version=cuda)
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "visible_gpu_count": 0,
            "selected_cuda_device": visible,
            "gpus": [],
            "driver_version": driver,
            "cuda_version": cuda,
        }
    return {
        "available": bool(gpus),
        "error": None,
        "visible_gpu_count": len(gpus),
        "selected_cuda_device": visible,
        "driver_version": driver,
        "cuda_version": cuda,
        "gpus": [gpu.__dict__ for gpu in gpus],
    }


def primary_gpu_profile(discovery: dict) -> dict:
    """Flatten the first discovered GPU into manifest hardware fields."""
    gpus = discovery.get("gpus") or []
    first = gpus[0] if gpus else {}
    memory_mb = first.get("memory_total_mb")
    return {
        "gpu_name": first.get("name"),
        "gpu_uuid": first.get("uuid"),
        "gpu_memory_bytes": None if memory_mb is None else memory_mb * 1024 * 1024,
        "cuda_version": discovery.get("cuda_version"),
        "driver_version": discovery.get("driver_version"),
        "visible_gpu_count": discovery.get("visible_gpu_count", 0),
        "selected_cuda_device": discovery.get("selected_cuda_device"),
        "gpu_discovery_error": discovery.get("error"),
    }
