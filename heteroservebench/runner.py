"""Asynchronous load generator and run orchestration."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from heteroservebench.backend import Backend, create_backend
from heteroservebench.config import ExperimentConfig
from heteroservebench.io import (
    MANIFEST_FILENAME,
    RAW_RESULTS_FILENAME,
    TELEMETRY_FILENAME,
    TRACE_FILENAME,
    WARMUP_RESULTS_FILENAME,
    append_raw_result,
    create_run_dir,
    refuse_to_overwrite_raw,
)
from heteroservebench.manifest import initial_manifest, manifest_hash, new_run_id, utc_now_iso
from heteroservebench.results import RequestResult
from heteroservebench.serialization import write_json
from heteroservebench.telemetry import TelemetrySampler
from heteroservebench.workload import BenchmarkRequest, WorkloadTrace, generate_workload


async def _execute_one(
    backend: Backend,
    request: BenchmarkRequest,
    base_time_ns: int,
    raw_path: Path,
    slo_latency_ms: float | None,
) -> RequestResult:
    target_ns = base_time_ns + int(request.scheduled_arrival_time_s * 1_000_000_000)
    delay_s = max(0.0, (target_ns - time.monotonic_ns()) / 1_000_000_000)
    if delay_s:
        await asyncio.sleep(delay_s)
    dispatch_ns = time.monotonic_ns()
    try:
        result = await backend.infer(request)
    except Exception as exc:
        result = RequestResult(
            request_id=request.request_id,
            workload_id=request.workload_id,
            scheduled_arrival_time_s=request.scheduled_arrival_time_s,
            actual_dispatch_time_ns=dispatch_ns,
            completion_time_ns=time.monotonic_ns(),
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_tokens=request.input_tokens,
            requested_input_tokens=request.input_tokens,
            requested_output_tokens=request.requested_output_tokens,
            failure_classification="backend_error",
            backend_metadata=backend.metadata(),
        )
    else:
        result.actual_dispatch_time_ns = dispatch_ns

    if result.backend_start_time_ns is not None and result.actual_dispatch_time_ns is not None:
        result.queue_latency_s = (result.backend_start_time_ns - result.actual_dispatch_time_ns) / 1_000_000_000
    if result.completion_time_ns is not None and result.actual_dispatch_time_ns is not None:
        result.end_to_end_latency_s = (result.completion_time_ns - result.actual_dispatch_time_ns) / 1_000_000_000
    if result.completion_time_ns is not None and result.backend_start_time_ns is not None:
        result.service_latency_s = (result.completion_time_ns - result.backend_start_time_ns) / 1_000_000_000
    if slo_latency_ms is not None:
        result.slo_latency_ms = slo_latency_ms
        result.slo_met = (
            result.success
            and result.end_to_end_latency_s is not None
            and result.end_to_end_latency_s * 1000.0 <= slo_latency_ms
        )
    append_raw_result(raw_path, result)
    return result


async def replay_schedule(
    trace: WorkloadTrace,
    backend: Backend,
    raw_path: Path,
    slo_latency_ms: float | None,
) -> list[RequestResult]:
    """Replay a planned request schedule against a backend."""
    refuse_to_overwrite_raw(raw_path)
    base_time_ns = time.monotonic_ns()
    tasks = [
        asyncio.create_task(_execute_one(backend, request, base_time_ns, raw_path, slo_latency_ms))
        for request in trace.requests
    ]
    return await asyncio.gather(*tasks)


def make_warmup_trace(trace: WorkloadTrace, count: int) -> WorkloadTrace:
    """Create identifiable warm-up requests based on the measured workload shape."""
    if count <= 0:
        return WorkloadTrace([])
    template = trace.requests[0]
    requests = [
        BenchmarkRequest(
            request_id=f"warmup-{index:08d}",
            workload_id=template.workload_id,
            input_tokens=template.input_tokens,
            requested_output_tokens=template.requested_output_tokens,
            scheduled_arrival_time_s=0.0,
            seed_metadata={"warmup": True, "sequence_index": index},
        )
        for index in range(count)
    ]
    return WorkloadTrace(requests)


def validate_context_capacity(config: ExperimentConfig, trace: WorkloadTrace) -> None:
    """Reject vLLM requests that cannot fit within configured model context."""
    if config.backend.type != "vllm":
        return
    max_model_len = config.backend.max_model_len
    for request in trace.requests:
        required_context_length = request.input_tokens + request.requested_output_tokens
        if required_context_length > max_model_len:
            raise ValueError(
                "request context length exceeds configured max_model_len: "
                f"request_id={request.request_id}, "
                f"workload_id={request.workload_id}, "
                f"requested_input_tokens={request.input_tokens}, "
                f"requested_output_tokens={request.requested_output_tokens}, "
                f"required_context_length={required_context_length}, "
                f"configured_max_model_len={max_model_len}"
            )


async def _run_with_optional_telemetry(
    trace: WorkloadTrace,
    backend: Backend,
    raw_path: Path,
    telemetry_path: Path,
    telemetry_enabled: bool,
    telemetry_interval_s: float,
    selected_cuda_device: str | None,
    slo_latency_ms: float | None,
) -> tuple[list[RequestResult], dict]:
    sampler = None
    sampler_task = None
    telemetry_status = {"samples": 0, "complete": None, "errors": []}
    if telemetry_enabled:
        sampler = TelemetrySampler(telemetry_path, telemetry_interval_s, selected_cuda_device=selected_cuda_device)
        sampler_task = asyncio.create_task(sampler.run())
    try:
        results = await replay_schedule(trace, backend, raw_path, slo_latency_ms)
    finally:
        if sampler is not None and sampler_task is not None:
            sampler.stop()
            await sampler_task
            telemetry_status = {
                "samples": sampler.samples_written,
                "complete": not sampler.errors,
                "errors": sampler.errors,
            }
    return results, telemetry_status


async def _execute_run(
    config: ExperimentConfig,
    trace: WorkloadTrace,
    backend: Backend,
    run_dir: Path,
) -> tuple[list[RequestResult], list[RequestResult], dict]:
    warmup_trace = make_warmup_trace(trace, config.warmup.count)
    warmup_results: list[RequestResult] = []
    if warmup_trace.requests:
        warmup_results = await replay_schedule(
            trace=warmup_trace,
            backend=backend,
            raw_path=run_dir / WARMUP_RESULTS_FILENAME,
            slo_latency_ms=None,
        )
    measured_results, telemetry_status = await _run_with_optional_telemetry(
        trace=trace,
        backend=backend,
        raw_path=run_dir / RAW_RESULTS_FILENAME,
        telemetry_path=run_dir / TELEMETRY_FILENAME,
        telemetry_enabled=config.telemetry.enabled,
        telemetry_interval_s=config.telemetry.sampling_interval_s,
        selected_cuda_device=config.backend.selected_cuda_device if config.backend.type == "vllm" else None,
        slo_latency_ms=None if config.slo is None else config.slo.latency_ms,
    )
    return warmup_results, measured_results, telemetry_status


def run_experiment(config: ExperimentConfig) -> Path:
    """Execute an experiment and return its run directory."""
    trace = generate_workload(config)
    validate_context_capacity(config, trace)
    run_id = new_run_id(config.campaign_id)
    run_dir = create_run_dir(config.output_dir, run_id)
    raw_path = run_dir / RAW_RESULTS_FILENAME
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = initial_manifest(config, trace, run_id, run_dir)
    write_json(run_dir / TRACE_FILENAME, trace.canonical())
    write_json(manifest_path, manifest)
    backend = create_backend(config.backend)

    try:
        warmup_results, results, telemetry_status = asyncio.run(_execute_run(config, trace, backend, run_dir))
    except BaseException as exc:
        manifest["run_status"] = "failed"
        manifest["terminal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["end_time"] = utc_now_iso()
        if raw_path.exists():
            lines = [line for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest["completed_requests"] = len(lines)
            manifest["failed_requests"] = None
        manifest["manifest_hash"] = manifest_hash(manifest)
        write_json(manifest_path, manifest)
        raise

    manifest["warmup_completed_requests"] = sum(1 for result in warmup_results if result.success)
    manifest["warmup_failed_requests"] = sum(1 for result in warmup_results if not result.success)
    manifest["completed_requests"] = sum(1 for result in results if result.success)
    manifest["failed_requests"] = sum(1 for result in results if not result.success)
    manifest["telemetry_samples"] = telemetry_status["samples"]
    manifest["telemetry_complete"] = telemetry_status["complete"]
    manifest["telemetry_errors"] = telemetry_status["errors"]
    manifest["run_status"] = "completed" if manifest["failed_requests"] == 0 else "completed_with_failures"
    manifest["end_time"] = utc_now_iso()
    manifest["manifest_hash"] = manifest_hash(manifest)
    write_json(manifest_path, manifest)
    return run_dir
