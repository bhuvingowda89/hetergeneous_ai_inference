"""Structured validation for completed or partial benchmark runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from heteroservebench.config import ExperimentConfig
from heteroservebench.io import MANIFEST_FILENAME, RAW_RESULTS_FILENAME, TELEMETRY_FILENAME, TRACE_FILENAME, read_raw_results
from heteroservebench.results import RequestResult
from heteroservebench.serialization import read_json, stable_hash
from heteroservebench.workload import BenchmarkRequest


REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "campaign_id",
    "run_id",
    "timestamp",
    "benchmark_version",
    "git_commit_sha",
    "config_hash",
    "canonical_configuration",
    "seed",
    "workload",
    "backend",
    "hardware_profile",
    "python_version",
    "operating_system",
    "hostname",
    "package_metadata",
    "run_status",
    "start_time",
    "end_time",
    "expected_requests",
    "completed_requests",
    "failed_requests",
    "raw_result_location",
    "model_provenance",
    "telemetry_enabled",
    "warmup_expected_requests",
}


def _issue(code: str, message: str, severity: str = "error", context: dict | None = None) -> dict:
    return {"code": code, "severity": severity, "message": message, "context": context or {}}


def _valid_timestamp(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _gpu_matches_identifier(gpu: dict, identifier: str) -> bool:
    return str(gpu.get("index")) == identifier or gpu.get("uuid") == identifier


def validate_run(run_dir: Path) -> dict:
    """Validate run artifacts and return a structured report."""
    issues: list[dict] = []
    manifest_path = run_dir / MANIFEST_FILENAME
    trace_path = run_dir / TRACE_FILENAME
    raw_path = run_dir / RAW_RESULTS_FILENAME
    telemetry_path = run_dir / TELEMETRY_FILENAME

    manifest: dict[str, Any] = {}
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return {
            "valid": False,
            "issues": [_issue("corrupt_manifest", f"cannot read manifest: {exc}")],
            "run_dir": str(run_dir),
        }

    missing_manifest_fields = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing_manifest_fields:
        issues.append(_issue("missing_manifest_fields", "manifest is missing required fields", context={"fields": missing_manifest_fields}))

    for field in ("timestamp", "start_time", "end_time"):
        if field in manifest and not _valid_timestamp(manifest[field]):
            issues.append(_issue("invalid_timestamp", f"manifest field {field} is not a valid ISO timestamp"))

    canonical_config = manifest.get("canonical_configuration")
    if isinstance(canonical_config, dict):
        actual_hash = stable_hash(canonical_config)
        if manifest.get("config_hash") != actual_hash:
            issues.append(_issue("config_hash_mismatch", "manifest config_hash does not match canonical configuration"))
        try:
            ExperimentConfig.model_validate(canonical_config)
        except ValidationError as exc:
            issues.append(_issue("invalid_canonical_config", "canonical configuration no longer validates", context={"errors": exc.errors()}))
    else:
        issues.append(_issue("missing_canonical_config", "manifest canonical_configuration is missing or invalid"))

    planned: list[BenchmarkRequest] = []
    try:
        planned_json = read_json(trace_path)
        planned = [BenchmarkRequest.model_validate(item) for item in planned_json]
    except Exception as exc:
        issues.append(_issue("corrupt_workload_trace", f"cannot read planned workload trace: {exc}"))

    results: list[RequestResult] = []
    try:
        if raw_path.exists():
            results = read_raw_results(raw_path)
        else:
            issues.append(_issue("missing_raw_results", "raw result file is missing"))
    except Exception as exc:
        issues.append(_issue("corrupt_raw_results", f"cannot read raw results: {exc}"))

    planned_by_id = {request.request_id: request for request in planned}
    planned_ids = [request.request_id for request in planned]
    result_ids = [result.request_id for result in results]
    backend_type = None
    if isinstance(canonical_config, dict) and isinstance(canonical_config.get("backend"), dict):
        backend_type = canonical_config["backend"].get("type")

    if len(planned_ids) != len(set(planned_ids)):
        issues.append(_issue("duplicate_planned_request_ids", "planned trace contains duplicate request IDs"))
    duplicate_result_ids = sorted({request_id for request_id in result_ids if result_ids.count(request_id) > 1})
    if duplicate_result_ids:
        issues.append(_issue("duplicate_result_request_ids", "raw results contain duplicate request IDs", context={"request_ids": duplicate_result_ids}))
    missing_result_ids = sorted(set(planned_ids) - set(result_ids))
    terminal = manifest.get("run_status") in {"completed", "completed_with_failures"}
    if terminal and missing_result_ids:
        issues.append(_issue("missing_request_ids", "terminal run is missing planned request results", context={"request_ids": missing_result_ids}))
    extra_result_ids = sorted(set(result_ids) - set(planned_ids))
    if extra_result_ids:
        issues.append(_issue("more_results_than_planned", "raw results include request IDs absent from the plan", context={"request_ids": extra_result_ids}))
    if len(results) > len(planned):
        issues.append(_issue("more_results_than_planned_count", "raw result count exceeds planned request count"))
    warmup_in_measured = sorted(request_id for request_id in result_ids if request_id.startswith("warmup-"))
    if warmup_in_measured:
        issues.append(_issue("measured_warmup_observations", "warm-up observations are present in measured raw results", context={"request_ids": warmup_in_measured}))

    for result in results:
        planned_request = planned_by_id.get(result.request_id)
        if planned_request and result.workload_id != planned_request.workload_id:
            issues.append(
                _issue(
                    "inconsistent_workload_identifier",
                    "result workload_id does not match planned workload_id",
                    context={"request_id": result.request_id},
                )
            )
        if result.end_to_end_latency_s is not None and result.end_to_end_latency_s < 0:
            issues.append(_issue("negative_latency", "end-to-end latency is negative", context={"request_id": result.request_id}))
        if result.queue_latency_s is not None and result.queue_latency_s < 0:
            issues.append(_issue("negative_latency", "queue latency is negative", context={"request_id": result.request_id}))
        if result.service_latency_s is not None and result.service_latency_s < 0:
            issues.append(_issue("negative_latency", "service latency is negative", context={"request_id": result.request_id}))
        if (
            result.actual_dispatch_time_ns is not None
            and result.completion_time_ns is not None
            and result.completion_time_ns < result.actual_dispatch_time_ns
        ):
            issues.append(_issue("completion_preceding_dispatch", "completion precedes dispatch", context={"request_id": result.request_id}))
        if result.ttft_s is not None and result.ttft_s < 0:
            issues.append(_issue("negative_ttft", "TTFT is negative", context={"request_id": result.request_id}))
        if result.ttft_s is not None and result.end_to_end_latency_s is not None and result.ttft_s > result.end_to_end_latency_s:
            issues.append(_issue("ttft_exceeds_latency", "TTFT exceeds end-to-end latency", context={"request_id": result.request_id}))
        if result.success:
            missing = [
                field
                for field in ("actual_dispatch_time_ns", "backend_start_time_ns", "completion_time_ns", "end_to_end_latency_s")
                if getattr(result, field) is None
            ]
            if missing:
                issues.append(
                    _issue(
                        "success_missing_required_timing",
                        "success record is missing required timing fields",
                        context={"request_id": result.request_id, "fields": missing},
                    )
                )
            if backend_type == "vllm":
                if result.input_tokens is None or result.requested_output_tokens is None:
                    issues.append(_issue("success_missing_token_counts", "GPU success record is missing configured token counts", context={"request_id": result.request_id}))
                if result.generated_tokens is None:
                    issues.append(_issue("generated_token_count_unavailable", "generated token count was not exposed by backend", severity="warning", context={"request_id": result.request_id}))

    if backend_type == "vllm":
        hardware = manifest.get("hardware_profile") or {}
        discovery = manifest.get("gpu_discovery") or {}
        model = manifest.get("model_provenance") or {}
        config_backend = canonical_config.get("backend", {}) if isinstance(canonical_config, dict) else {}
        if not (hardware.get("gpu_name") or hardware.get("gpu_uuid")):
            issues.append(_issue("missing_gpu_identity", "GPU run does not contain a discovered GPU identity"))
        expected_gpu_count = config_backend.get("expected_gpu_count")
        actual_benchmark_visible = hardware.get("benchmark_visible_gpu_count")
        if actual_benchmark_visible is None:
            actual_benchmark_visible = discovery.get("benchmark_visible_gpu_count")
        if expected_gpu_count is not None and actual_benchmark_visible != expected_gpu_count:
            issues.append(
                _issue(
                    "gpu_count_mismatch",
                    "benchmark-visible GPU count does not match configuration",
                    context={"expected": expected_gpu_count, "actual": actual_benchmark_visible},
                )
            )
        selected_device = config_backend.get("selected_cuda_device") or hardware.get("selected_cuda_device")
        discovered_gpus = discovery.get("gpus") or []
        if selected_device is not None and not any(_gpu_matches_identifier(gpu, str(selected_device)) for gpu in discovered_gpus):
            issues.append(
                _issue(
                    "selected_gpu_absent",
                    "configured selected CUDA device is absent from GPU discovery results",
                    context={"selected_cuda_device": selected_device},
                )
            )
        if not model.get("model_id"):
            issues.append(_issue("missing_model_identity", "model provenance is missing model_id"))
        if not model.get("serving_engine"):
            issues.append(_issue("missing_serving_engine", "model provenance is missing serving engine"))
        if model.get("serving_engine_version") is None:
            issues.append(_issue("missing_serving_engine_version", "serving engine version is unavailable", severity="warning"))
        if "dtype" not in model or model.get("dtype") in (None, ""):
            issues.append(_issue("missing_dtype", "model provenance is missing explicit dtype"))
        if "quantization" not in model:
            issues.append(_issue("missing_quantization_status", "model provenance is missing explicit quantization status"))
        if model.get("tensor_parallel_size") is None:
            issues.append(_issue("missing_tensor_parallel_size", "model provenance is missing tensor parallel size"))
        if model.get("resolved_model_revision_hash") is None:
            issues.append(_issue("unidentified_model_revision", "resolved model revision/hash is unavailable", severity="warning"))

        if manifest.get("telemetry_enabled"):
            telemetry_rows = []
            if telemetry_path.exists():
                try:
                    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            import json

                            telemetry_rows.append(json.loads(line))
                except Exception as exc:
                    issues.append(_issue("corrupt_telemetry", f"cannot read telemetry: {exc}"))
            if not telemetry_rows:
                issues.append(_issue("missing_telemetry", "telemetry was enabled but no samples were recorded"))
            else:
                starts = [result.actual_dispatch_time_ns for result in results if result.actual_dispatch_time_ns is not None]
                ends = [result.completion_time_ns for result in results if result.completion_time_ns is not None]
                sample_times = [row.get("timestamp_monotonic_ns") for row in telemetry_rows if row.get("timestamp_monotonic_ns") is not None]
                if starts and ends and sample_times:
                    lower = min(starts)
                    upper = max(ends)
                    if not any(lower <= sample <= upper for sample in sample_times):
                        issues.append(_issue("telemetry_no_overlap", "telemetry timestamps do not overlap measured experiment time"))

    report = {
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "issues": issues,
        "run_dir": str(run_dir),
        "planned_requests": len(planned),
        "raw_results": len(results),
    }
    return report
