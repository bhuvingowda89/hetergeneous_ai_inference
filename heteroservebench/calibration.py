"""Adaptive offered-load calibration for vLLM GPU runs."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, Optional

from heteroservebench.config import ExperimentConfig, FixedIntervalArrivalConfig, LoadConfig, WarmupConfig
from heteroservebench.io import MANIFEST_FILENAME, RAW_RESULTS_FILENAME, TELEMETRY_FILENAME, read_raw_results
from heteroservebench.manifest import git_commit_sha, new_run_id, package_metadata, utc_now_iso
from heteroservebench.runner import run_experiment, validate_context_capacity
from heteroservebench.serialization import read_json, write_json
from heteroservebench.tokenizer_prompt import construct_exact_prompt, load_tokenizer, tokenizer_identifier
from heteroservebench.validation import validate_run
from heteroservebench.vllm_http import VllmHttpClient
from heteroservebench.workload import CANONICAL_WORKLOADS, generate_workload


MIN_PROBE_RPS = 0.01
INITIAL_MAX_RPS = 4.0
MAX_PROBE_RPS = 16.0
MAX_OPEN_LOOP_PROBES = 10
MAX_REFINEMENT_PROBES = 3
BRACKET_RATIO_STOP = 1.25
DEFAULT_COOLDOWN_S = 5.0
BASELINE_REPETITIONS = 3
BASELINE_MEASURED_REQUESTS = 1
PROBE_WARMUP_REQUESTS = 1
PROBE_MEASURED_REQUESTS = 8


@dataclass(frozen=True)
class ProbeOutcome:
    offered_rps: float
    strict_validation_valid: bool
    success_count: int
    failure_count: int
    measured_request_count: int
    achieved_throughput_rps: Optional[float]
    throughput_ratio: Optional[float]


@dataclass(frozen=True)
class SearchResult:
    highest_sustainable_rps: Optional[float]
    first_non_sustainable_rps: Optional[float]
    capacity_rps: Optional[float]
    capacity_is_lower_bound: bool
    calibration_failed: bool
    probes: list[ProbeOutcome]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def initial_offered_rps(l0_seconds: float) -> float:
    if l0_seconds <= 0:
        raise ValueError(f"L0 must be positive: {l0_seconds}")
    return clamp(2.0 / l0_seconds, MIN_PROBE_RPS, INITIAL_MAX_RPS)


def interval_ms_for_rate(offered_rps: float) -> float:
    if offered_rps <= 0:
        raise ValueError(f"offered_rps must be positive: {offered_rps}")
    return 1000.0 / offered_rps


def is_sustainable_probe(outcome: ProbeOutcome) -> bool:
    if not outcome.strict_validation_valid:
        return False
    if outcome.failure_count != 0:
        return False
    if outcome.success_count != outcome.measured_request_count:
        return False
    return outcome.throughput_ratio is not None and outcome.throughput_ratio >= 0.95


def adaptive_search(
    initial_rps: float,
    probe: Callable[[float, str], ProbeOutcome],
    *,
    minimum_rps: float = MIN_PROBE_RPS,
    maximum_rps: float = MAX_PROBE_RPS,
    max_open_loop_probes: int = MAX_OPEN_LOOP_PROBES,
    max_refinement_probes: int = MAX_REFINEMENT_PROBES,
    bracket_ratio_stop: float = BRACKET_RATIO_STOP,
) -> SearchResult:
    """Run pure adaptive search over an injected probe function."""
    probes: list[ProbeOutcome] = []
    rate = clamp(initial_rps, minimum_rps, maximum_rps)

    def run(rate_value: float, probe_type: str) -> ProbeOutcome:
        outcome = probe(rate_value, probe_type)
        probes.append(outcome)
        return outcome

    first = run(rate, "initial")
    if is_sustainable_probe(first):
        low = rate
        high: Optional[float] = None
        while len(probes) < max_open_loop_probes and low < maximum_rps:
            candidate = min(low * 2.0, maximum_rps)
            if candidate == low:
                break
            outcome = run(candidate, "expansion")
            if is_sustainable_probe(outcome):
                low = candidate
                if low >= maximum_rps:
                    break
            else:
                high = candidate
                break
        if high is None:
            return SearchResult(low, None, low, True, False, probes)
    else:
        high = rate
        low = None
        while len(probes) < max_open_loop_probes and high > minimum_rps:
            candidate = max(high / 2.0, minimum_rps)
            if candidate == high:
                break
            outcome = run(candidate, "halving")
            if is_sustainable_probe(outcome):
                low = candidate
                break
            high = candidate
        if low is None:
            return SearchResult(None, high, None, False, True, probes)

    assert low is not None
    assert high is not None
    refinements = 0
    while high / low > bracket_ratio_stop and refinements < max_refinement_probes and len(probes) < max_open_loop_probes:
        midpoint = math.sqrt(low * high)
        outcome = run(midpoint, "refinement")
        refinements += 1
        if is_sustainable_probe(outcome):
            low = midpoint
        else:
            high = midpoint
    return SearchResult(low, high, low, False, False, probes)


def derived_probe_config(
    base_config: ExperimentConfig,
    workload_id: str,
    *,
    offered_rps: float,
    measured_requests: int,
    warmup_requests: int,
    output_dir: Path,
    experiment_suffix: str,
) -> ExperimentConfig:
    """Return a fresh config for one calibration probe without mutating the base."""
    config = base_config.model_copy(deep=True)
    config.validation_mode = "scientific"
    config.workload.id = workload_id
    config.workload.probabilities = None
    config.arrival = FixedIntervalArrivalConfig(interval_ms=interval_ms_for_rate(offered_rps))
    config.load = LoadConfig(request_count=measured_requests)
    config.warmup = WarmupConfig(count=warmup_requests)
    config.output_dir = output_dir
    config.experiment_id = f"{base_config.experiment_id}-{experiment_suffix}-{workload_id.lower()}"
    config.campaign_id = f"{base_config.campaign_id}-calibration-{workload_id.lower()}"
    return config


def check_server_health(base_url: str, timeout_s: float) -> None:
    VllmHttpClient(base_url, timeout_s).models()


def check_scientific_calibration_config(base_config: ExperimentConfig, workload_ids: Iterable[str], *, check_server: bool = True) -> None:
    if base_config.backend.type != "vllm":
        raise ValueError("calibration requires a vllm backend")
    if base_config.validation_mode != "scientific":
        raise ValueError("calibration requires validation_mode=scientific")
    if base_config.backend.requested_model_revision is None:
        raise ValueError("calibration requires backend.requested_model_revision")
    if not base_config.backend.exact_output_tokens:
        raise ValueError("calibration requires exact_output_tokens=true")
    tokenizer_id = tokenizer_identifier(base_config.backend.model, base_config.backend.tokenizer)
    tokenizer = load_tokenizer(tokenizer_id, base_config.backend.requested_model_revision)
    for workload_id in workload_ids:
        input_tokens, output_tokens = CANONICAL_WORKLOADS[workload_id]
        construct_exact_prompt(tokenizer, input_tokens)
        required_context_length = input_tokens + output_tokens
        if required_context_length > base_config.backend.max_model_len:
            raise ValueError(
                f"{workload_id} requires context length {required_context_length}, "
                f"configured max_model_len={base_config.backend.max_model_len}"
            )
        if output_tokens > base_config.backend.max_tokens:
            raise ValueError(
                f"{workload_id} requested_output_tokens={output_tokens} exceeds "
                f"configured max_tokens={base_config.backend.max_tokens}"
            )
    if check_server:
        check_server_health(base_config.backend.base_url, base_config.backend.request_timeout_s)


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _mean(values: list[float]) -> Optional[float]:
    return None if not values else mean(values)


def _read_telemetry(run_dir: Path) -> list[dict]:
    path = run_dir / TELEMETRY_FILENAME
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def probe_record(
    *,
    workload_id: str,
    probe_type: str,
    run_dir: Path,
    offered_rps: float,
    measured_request_count: int,
    validation_report: dict,
) -> dict:
    manifest = read_json(run_dir / MANIFEST_FILENAME)
    results = read_raw_results(run_dir / RAW_RESULTS_FILENAME)
    successes = [result for result in results if result.success]
    starts = [result.actual_dispatch_time_ns for result in results if result.actual_dispatch_time_ns is not None]
    ends = [result.completion_time_ns for result in results if result.completion_time_ns is not None]
    duration_s = None
    if starts and ends and max(ends) >= min(starts):
        duration_s = (max(ends) - min(starts)) / 1_000_000_000
    achieved = None if not duration_s or duration_s <= 0 else len(successes) / duration_s
    throughput_ratio = None if achieved is None else achieved / offered_rps
    latencies = [result.end_to_end_latency_s for result in successes if result.end_to_end_latency_s is not None]
    ttfts = [result.ttft_s for result in successes if result.ttft_s is not None]
    services = [result.service_latency_s for result in successes if result.service_latency_s is not None]
    queues = [result.queue_latency_s for result in successes if result.queue_latency_s is not None]
    telemetry_rows = _read_telemetry(run_dir)
    gpu_utils = [row["gpu_utilization_percent"] for row in telemetry_rows if row.get("gpu_utilization_percent") is not None]
    gpu_memory = [row["gpu_memory_used_mb"] for row in telemetry_rows if row.get("gpu_memory_used_mb") is not None]
    first_success = successes[0] if successes else None
    model = manifest.get("model_provenance") or {}
    hardware = manifest.get("hardware_profile") or {}
    return {
        "workload_id": workload_id,
        "probe_type": probe_type,
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir),
        "offered_rps": offered_rps,
        "interval_ms": interval_ms_for_rate(offered_rps),
        "measured_request_count": measured_request_count,
        "strict_validation_valid": validation_report.get("valid"),
        "validation_issues": validation_report.get("issues", []),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "success_rate": None if not results else len(successes) / len(results),
        "achieved_throughput_rps": achieved,
        "throughput_ratio": throughput_ratio,
        "e2e_latency_mean_s": _mean(latencies),
        "e2e_latency_p50_s": _percentile(latencies, 50),
        "e2e_latency_p95_s": _percentile(latencies, 95),
        "e2e_latency_p99_s": _percentile(latencies, 99),
        "ttft_mean_s": _mean(ttfts),
        "ttft_p50_s": _percentile(ttfts, 50),
        "ttft_p95_s": _percentile(ttfts, 95),
        "service_latency_mean_s": _mean(services),
        "service_latency_p95_s": _percentile(services, 95),
        "queue_latency_p95_s": _percentile(queues, 95),
        "requested_input_tokens": None if first_success is None else first_success.requested_input_tokens,
        "actual_prompt_tokens": None if first_success is None else first_success.actual_prompt_tokens,
        "provider_prompt_tokens": None if first_success is None else first_success.provider_prompt_tokens,
        "requested_output_tokens": None if first_success is None else first_success.requested_output_tokens,
        "generated_output_tokens": None if first_success is None else first_success.generated_tokens,
        "gpu_utilization_mean_pct": _mean(gpu_utils),
        "gpu_utilization_p95_pct": _percentile(gpu_utils, 95),
        "gpu_memory_used_max_mb": None if not gpu_memory else max(gpu_memory),
        "telemetry_sample_count": len(telemetry_rows),
        "git_commit_sha": manifest.get("git_commit_sha"),
        "model_id": model.get("model_id"),
        "resolved_model_revision": model.get("resolved_model_revision_hash"),
        "vllm_version": model.get("serving_engine_version"),
        "gpu_name": hardware.get("gpu_name"),
        "gpu_uuid": hardware.get("gpu_uuid"),
    }


def outcome_from_record(record: dict) -> ProbeOutcome:
    return ProbeOutcome(
        offered_rps=record["offered_rps"],
        strict_validation_valid=bool(record["strict_validation_valid"]),
        success_count=record["success_count"],
        failure_count=record["failure_count"],
        measured_request_count=record["measured_request_count"],
        achieved_throughput_rps=record["achieved_throughput_rps"],
        throughput_ratio=record["throughput_ratio"],
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    fieldnames = [
        "workload_id",
        "unloaded_median_service_latency_s",
        "unloaded_median_ttFT_s",
        "highest_sustainable_offered_rps",
        "first_non_sustainable_offered_rps",
        "estimated_capacity_rps",
        "capacity_is_lower_bound",
        "number_of_probes",
        "run_directories",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})


def dry_run_plan(base_config: ExperimentConfig, workload_ids: list[str], *, assumed_l0_s: float = 1.0) -> dict:
    output_root = Path("calibration_runs") / "dry-run"
    workloads = []
    for workload_id in workload_ids:
        rate = initial_offered_rps(assumed_l0_s)
        config = derived_probe_config(
            base_config,
            workload_id,
            offered_rps=rate,
            measured_requests=PROBE_MEASURED_REQUESTS,
            warmup_requests=PROBE_WARMUP_REQUESTS,
            output_dir=output_root / workload_id.lower(),
            experiment_suffix="dry-run",
        )
        workloads.append(
            {
                "workload_id": workload_id,
                "input_tokens": CANONICAL_WORKLOADS[workload_id][0],
                "requested_output_tokens": CANONICAL_WORKLOADS[workload_id][1],
                "assumed_l0_seconds": assumed_l0_s,
                "proposed_initial_offered_rps": rate,
                "interval_ms": config.arrival.interval_ms,
                "derived_configuration": config.canonical(),
            }
        )
    return {"dry_run": True, "workloads": workloads}


def run_calibration(
    base_config: ExperimentConfig,
    workload_ids: list[str],
    *,
    output_root: Path = Path("calibration_runs"),
    cooldown_s: float = DEFAULT_COOLDOWN_S,
) -> Path:
    base_config = base_config.model_copy(deep=True)
    base_config.validation_mode = "scientific"
    check_scientific_calibration_config(base_config, workload_ids)
    calibration_id = new_run_id("phase3d-calibration")
    calibration_dir = output_root / calibration_id
    calibration_dir.mkdir(parents=True, exist_ok=False)
    all_records: list[dict] = []
    summaries: list[dict] = []
    manifest = {
        "calibration_id": calibration_id,
        "start_time": utc_now_iso(),
        "base_config": base_config.canonical(),
        "workloads": workload_ids,
        "git_commit_sha": git_commit_sha(Path.cwd()),
        "package_metadata": package_metadata(),
    }
    write_json(calibration_dir / "calibration_manifest.json", manifest)

    for workload_id in workload_ids:
        baseline_records = []
        baseline_services = []
        baseline_ttfts = []
        baseline_e2e = []
        workload_output = calibration_dir / workload_id.lower()
        for index in range(BASELINE_REPETITIONS):
            check_server_health(base_config.backend.base_url, base_config.backend.request_timeout_s)
            config = derived_probe_config(
                base_config,
                workload_id,
                offered_rps=MIN_PROBE_RPS,
                measured_requests=BASELINE_MEASURED_REQUESTS,
                warmup_requests=0,
                output_dir=workload_output / "baseline",
                experiment_suffix=f"baseline-{index}",
            )
            validate_context_capacity(config, generate_workload(config))
            run_dir = run_experiment(config)
            report = validate_run(run_dir, strict_scientific=True)
            record = probe_record(
                workload_id=workload_id,
                probe_type="baseline",
                run_dir=run_dir,
                offered_rps=MIN_PROBE_RPS,
                measured_request_count=BASELINE_MEASURED_REQUESTS,
                validation_report=report,
            )
            baseline_records.append(record)
            all_records.append(record)
            if not report.get("valid"):
                raise RuntimeError(f"baseline strict validation failed for {workload_id}: {report}")
            baseline_services.extend([value for value in [record["service_latency_mean_s"]] if value is not None])
            baseline_ttfts.extend([value for value in [record["ttft_mean_s"]] if value is not None])
            baseline_e2e.extend([value for value in [record["e2e_latency_mean_s"]] if value is not None])
            if cooldown_s:
                time.sleep(cooldown_s)
        l0 = median(baseline_services)
        initial_rate = initial_offered_rps(l0)

        probe_records_by_rate: dict[float, dict] = {}

        def execute_probe(rate: float, probe_type: str) -> ProbeOutcome:
            check_server_health(base_config.backend.base_url, base_config.backend.request_timeout_s)
            config = derived_probe_config(
                base_config,
                workload_id,
                offered_rps=rate,
                measured_requests=PROBE_MEASURED_REQUESTS,
                warmup_requests=PROBE_WARMUP_REQUESTS,
                output_dir=workload_output / "probes",
                experiment_suffix=probe_type,
            )
            validate_context_capacity(config, generate_workload(config))
            run_dir = run_experiment(config)
            report = validate_run(run_dir, strict_scientific=True)
            record = probe_record(
                workload_id=workload_id,
                probe_type=probe_type,
                run_dir=run_dir,
                offered_rps=rate,
                measured_request_count=PROBE_MEASURED_REQUESTS,
                validation_report=report,
            )
            probe_records_by_rate[rate] = record
            all_records.append(record)
            if cooldown_s:
                time.sleep(cooldown_s)
            return outcome_from_record(record)

        search = adaptive_search(initial_rate, execute_probe)
        if search.calibration_failed:
            raise RuntimeError(f"no sustainable calibration point found for {workload_id} at minimum rate")
        summaries.append(
            {
                "workload_id": workload_id,
                "unloaded_median_service_latency_s": l0,
                "unloaded_median_ttFT_s": None if not baseline_ttfts else median(baseline_ttfts),
                "unloaded_median_e2e_latency_s": None if not baseline_e2e else median(baseline_e2e),
                "highest_sustainable_offered_rps": search.highest_sustainable_rps,
                "first_non_sustainable_offered_rps": search.first_non_sustainable_rps,
                "estimated_capacity_rps": search.capacity_rps,
                "capacity_is_lower_bound": search.capacity_is_lower_bound,
                "number_of_probes": len(search.probes),
                "run_directories": [record["run_dir"] for record in baseline_records + list(probe_records_by_rate.values())],
            }
        )

    write_jsonl(calibration_dir / "calibration_probes.jsonl", all_records)
    write_json(calibration_dir / "calibration_summary.json", summaries)
    write_summary_csv(calibration_dir / "calibration_summary.csv", summaries)
    manifest["end_time"] = utc_now_iso()
    write_json(calibration_dir / "calibration_manifest.json", manifest)
    return calibration_dir


def parse_workload_ids(text: str) -> list[str]:
    workload_ids = [item.strip() for item in text.split(",") if item.strip()]
    valid = set(CANONICAL_WORKLOADS)
    invalid = sorted(set(workload_ids) - valid)
    if invalid:
        raise ValueError(f"unknown workload IDs: {', '.join(invalid)}")
    if not workload_ids:
        raise ValueError("at least one workload is required")
    return workload_ids
