"""Lightweight runtime telemetry sampling."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


def _host_memory_utilization_percent() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            values = {}
            for line in handle:
                key, rest = line.split(":", 1)
                values[key] = float(rest.strip().split()[0])
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total and available is not None:
            return (total - available) / total * 100.0
    except Exception:
        return None
    return None


def host_metrics() -> dict:
    """Return best-effort host telemetry without psutil dependency."""
    load_percent = None
    try:
        load1 = os.getloadavg()[0]
        cpus = os.cpu_count() or 1
        load_percent = min(100.0, load1 / cpus * 100.0)
    except Exception:
        pass
    return {
        "host_cpu_utilization_percent": load_percent,
        "host_memory_utilization_percent": _host_memory_utilization_percent(),
    }


def sample_gpu_telemetry() -> dict:
    """Sample one point of GPU telemetry via nvidia-smi."""
    sample = {
        "timestamp_ns": time.time_ns(),
        "timestamp_monotonic_ns": time.monotonic_ns(),
        "gpu_utilization_percent": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_c": None,
        "gpu_power_draw_w": None,
        "telemetry_error": None,
    }
    sample.update(host_metrics())
    if shutil.which("nvidia-smi") is None:
        sample["telemetry_error"] = "nvidia-smi not found"
        return sample
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = completed.stdout.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        sample["gpu_utilization_percent"] = float(parts[0])
        sample["gpu_memory_used_mb"] = float(parts[1])
        sample["gpu_memory_total_mb"] = float(parts[2])
        sample["gpu_temperature_c"] = float(parts[3])
        sample["gpu_power_draw_w"] = float(parts[4])
    except Exception as exc:
        sample["telemetry_error"] = f"{type(exc).__name__}: {exc}"
    return sample


class TelemetrySampler:
    """Async best-effort telemetry sampler that never owns request results."""

    def __init__(self, path: Path, interval_s: float) -> None:
        self.path = path
        self.interval_s = interval_s
        self._stop = asyncio.Event()
        self.errors: list[str] = []
        self.samples_written = 0

    async def run(self) -> None:
        """Sample until stopped."""
        while not self._stop.is_set():
            try:
                sample = await asyncio.to_thread(sample_gpu_telemetry)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                self.samples_written += 1
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        """Stop sampling."""
        self._stop.set()
