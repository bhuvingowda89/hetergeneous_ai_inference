"""Run manifest construction and provenance capture."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from heteroservebench import __version__
from heteroservebench.config import ExperimentConfig
from heteroservebench.gpu import discover_nvidia_gpus, primary_gpu_profile
from heteroservebench.serialization import stable_hash
from heteroservebench.tokenizer_prompt import resolve_hf_snapshot_revision, tokenizer_identifier
from heteroservebench.workload import WorkloadTrace


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id(campaign_id: str) -> str:
    """Create a unique run ID suitable for directory names."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{campaign_id}-{stamp}-{uuid4().hex[:12]}"


def git_commit_sha(cwd: Path) -> str | None:
    """Return the current git commit SHA when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def package_metadata() -> dict[str, str | None]:
    """Capture versions of runtime packages used by the benchmark."""
    packages = ["heteroservebench", "pydantic", "yaml", "vllm"]
    values: dict[str, str | None] = {}
    for package in packages:
        lookup = "PyYAML" if package == "yaml" else package
        try:
            values[package] = metadata.version(lookup)
        except metadata.PackageNotFoundError:
            values[package] = __version__ if package == "heteroservebench" else None
    return values


def hardware_profile() -> dict[str, Any]:
    """Capture CPU-safe hardware profile fields, with GPU fields left null."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "gpu_name": None,
        "gpu_uuid": None,
        "gpu_index": None,
        "gpu_memory_bytes": None,
        "cuda_version": None,
        "driver_version": None,
        "vllm_version": None,
        "model_id": None,
        "model_revision_hash": None,
        "container_digest": None,
        "physical_gpu_count": 0,
        "cuda_visible_devices": None,
        "benchmark_visible_gpu_count": 0,
        "selected_cuda_device": None,
        "gpu_discovery_error": None,
    }


def model_provenance(config: ExperimentConfig) -> dict[str, Any]:
    """Return model provenance fields, leaving unknown values null."""
    if config.backend.type != "vllm":
        return {
            "model_id": None,
            "requested_model_revision": None,
            "resolved_model_revision_hash": None,
            "tokenizer_identifier": None,
            "tokenizer_resolved_revision_hash": None,
            "dtype": None,
            "quantization": None,
            "tensor_parallel_size": None,
            "serving_engine": None,
            "serving_engine_version": None,
            "serving_mode": None,
            "serving_engine_command": None,
            "max_model_len": None,
            "warnings": [],
        }
    warnings = []
    if config.backend.requested_model_revision is None:
        warnings.append("requested model revision is not specified")
    tokenizer_id = tokenizer_identifier(config.backend.model, config.backend.tokenizer)
    resolved_model_revision = resolve_hf_snapshot_revision(config.backend.model, config.backend.requested_model_revision)
    resolved_tokenizer_revision = resolve_hf_snapshot_revision(tokenizer_id, config.backend.requested_model_revision)
    if resolved_model_revision is None:
        warnings.append("resolved model revision/hash is unavailable")
    return {
        "model_id": config.backend.model,
        "requested_model_revision": config.backend.requested_model_revision,
        "resolved_model_revision_hash": resolved_model_revision,
        "tokenizer_identifier": tokenizer_id,
        "tokenizer_resolved_revision_hash": resolved_tokenizer_revision,
        "dtype": config.backend.dtype,
        "quantization": config.backend.quantization,
        "tensor_parallel_size": config.backend.tensor_parallel_size,
        "serving_engine": "vllm",
        "serving_engine_version": package_metadata().get("vllm"),
        "serving_mode": config.backend.serving_mode,
        "serving_engine_command": config.backend.serving_engine_command,
        "max_model_len": config.backend.max_model_len,
        "warnings": warnings,
    }


def initial_manifest(
    config: ExperimentConfig,
    trace: WorkloadTrace,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Create a manifest before request execution starts."""
    canonical_config = config.canonical()
    profile = hardware_profile()
    gpu_discovery = None
    provenance = model_provenance(config)
    if config.backend.type == "vllm":
        gpu_discovery = discover_nvidia_gpus()
        profile.update(primary_gpu_profile(gpu_discovery, config.backend.selected_cuda_device))
        profile["model_id"] = config.backend.model
        profile["vllm_version"] = package_metadata().get("vllm")
        profile["model_revision_hash"] = provenance.get("resolved_model_revision_hash")
    return {
        "schema_version": "1.0",
        "validation_mode": config.validation_mode,
        "campaign_id": config.campaign_id,
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "benchmark_version": __version__,
        "git_commit_sha": git_commit_sha(Path.cwd()),
        "config_hash": config.config_hash(),
        "canonical_configuration": canonical_config,
        "seed": config.seed.value,
        "workload": config.workload.model_dump(mode="json"),
        "workload_trace_hash": trace.hash(),
        "backend": config.backend.model_dump(mode="json"),
        "hardware_profile": profile,
        "gpu_discovery": gpu_discovery,
        "model_provenance": provenance,
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "hostname": socket.gethostname(),
        "execution_identifier": socket.gethostname(),
        "package_metadata": package_metadata(),
        "run_status": "running",
        "start_time": utc_now_iso(),
        "end_time": None,
        "expected_requests": len(trace.requests),
        "warmup_expected_requests": config.warmup.count,
        "warmup_completed_requests": 0,
        "warmup_failed_requests": 0,
        "warmup_result_location": str(run_dir / "warmup_results.jsonl"),
        "completed_requests": 0,
        "failed_requests": 0,
        "raw_result_location": str(run_dir / "raw_results.jsonl"),
        "telemetry_enabled": config.telemetry.enabled,
        "telemetry_location": str(run_dir / "telemetry.jsonl"),
        "telemetry_samples": 0,
        "telemetry_complete": None,
        "telemetry_errors": [],
        "run_directory": str(run_dir),
        "terminal_error": None,
    }


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Return a stable hash of manifest content excluding its own hash if present."""
    copy = dict(manifest)
    copy.pop("manifest_hash", None)
    return stable_hash(copy)
