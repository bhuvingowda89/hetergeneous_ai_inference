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


def parse_cuda_visible_devices(value: Optional[str]) -> tuple[Optional[list[str]], Optional[str]]:
    """Parse CUDA_VISIBLE_DEVICES without guessing malformed values."""
    if value is None:
        return None, None
    stripped = value.strip()
    if stripped == "":
        return [], None
    devices = [item.strip() for item in stripped.split(",")]
    if any(item == "" for item in devices):
        return None, f"malformed CUDA_VISIBLE_DEVICES: {value!r}"
    return devices, None


def _gpu_matches_identifier(gpu: dict, identifier: str) -> bool:
    return str(gpu.get("index")) == identifier or gpu.get("uuid") == identifier


def benchmark_visible_gpus(gpus: list[dict], cuda_visible_devices: Optional[str]) -> tuple[list[dict], Optional[str]]:
    """Return GPUs visible to the benchmark after CUDA_VISIBLE_DEVICES scoping."""
    parsed, error = parse_cuda_visible_devices(cuda_visible_devices)
    if error:
        return [], error
    if parsed is None:
        return list(gpus), None
    visible = [gpu for gpu in gpus if any(_gpu_matches_identifier(gpu, item) for item in parsed)]
    missing = [item for item in parsed if not any(_gpu_matches_identifier(gpu, item) for gpu in gpus)]
    if missing:
        return visible, f"CUDA_VISIBLE_DEVICES references undiscovered GPU(s): {','.join(missing)}"
    return visible, None


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
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "error": "nvidia-smi not found",
            "physical_gpu_count": 0,
            "cuda_visible_devices": cuda_visible_devices,
            "benchmark_visible_gpu_count": 0,
            "selected_cuda_device": None,
            "gpus": [],
            "benchmark_visible_gpus": [],
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
            "physical_gpu_count": 0,
            "cuda_visible_devices": cuda_visible_devices,
            "benchmark_visible_gpu_count": 0,
            "selected_cuda_device": None,
            "gpus": [],
            "benchmark_visible_gpus": [],
            "driver_version": driver,
            "cuda_version": cuda,
        }
    gpu_dicts = [gpu.__dict__ for gpu in gpus]
    visible_gpus, visibility_error = benchmark_visible_gpus(gpu_dicts, cuda_visible_devices)
    parsed_visible, parse_error = parse_cuda_visible_devices(cuda_visible_devices)
    selected_cuda_device = None
    if parsed_visible:
        selected_cuda_device = parsed_visible[0]
    return {
        "available": bool(gpus),
        "error": visibility_error or parse_error,
        "physical_gpu_count": len(gpus),
        "cuda_visible_devices": cuda_visible_devices,
        "benchmark_visible_gpu_count": len(visible_gpus) if not visibility_error else None,
        "selected_cuda_device": selected_cuda_device,
        "driver_version": driver,
        "cuda_version": cuda,
        "gpus": gpu_dicts,
        "benchmark_visible_gpus": visible_gpus,
    }


def selected_gpu(discovery: dict, selected_cuda_device: Optional[str] = None) -> Optional[dict]:
    """Return the selected GPU from discovery, preferring an explicit identifier."""
    gpus = discovery.get("gpus") or []
    if selected_cuda_device is not None:
        for gpu in gpus:
            if _gpu_matches_identifier(gpu, selected_cuda_device):
                return gpu
        return None
    visible = discovery.get("benchmark_visible_gpus") or []
    if visible:
        return visible[0]
    return gpus[0] if gpus else None


def primary_gpu_profile(discovery: dict, selected_cuda_device: Optional[str] = None) -> dict:
    """Flatten the selected benchmark GPU into manifest hardware fields."""
    chosen = selected_gpu(discovery, selected_cuda_device) or {}
    memory_mb = chosen.get("memory_total_mb")
    return {
        "gpu_name": chosen.get("name"),
        "gpu_uuid": chosen.get("uuid"),
        "gpu_index": chosen.get("index"),
        "gpu_memory_bytes": None if memory_mb is None else memory_mb * 1024 * 1024,
        "cuda_version": discovery.get("cuda_version"),
        "driver_version": discovery.get("driver_version"),
        "physical_gpu_count": discovery.get("physical_gpu_count", 0),
        "cuda_visible_devices": discovery.get("cuda_visible_devices"),
        "benchmark_visible_gpu_count": discovery.get("benchmark_visible_gpu_count", 0),
        "selected_cuda_device": selected_cuda_device or discovery.get("selected_cuda_device"),
        "gpu_discovery_error": discovery.get("error"),
    }
