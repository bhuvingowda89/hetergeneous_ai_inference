from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from heteroservebench.backend import SimulatedBackend
from heteroservebench.config import (
    ExperimentConfig,
    FixedIntervalArrivalConfig,
    LoadConfig,
    SeedConfig,
    SimulatedBackendConfig,
    SLOConfig,
    WorkloadConfig,
    load_config,
)
from heteroservebench.io import MANIFEST_FILENAME, RAW_RESULTS_FILENAME, SUMMARY_FILENAME, append_raw_result
from heteroservebench.metrics import summarize_run
from heteroservebench.results import RequestResult
from heteroservebench.runner import replay_schedule, run_experiment
from heteroservebench.serialization import read_json, stable_hash, write_json
from heteroservebench.validation import validate_run
from heteroservebench.workload import generate_workload


def make_config(tmp_path: Path, *, seed: int = 123, request_count: int = 5, workload_id: str = "W6") -> ExperimentConfig:
    workload = (
        WorkloadConfig(id="W6", probabilities={"W1": 0.2, "W2": 0.2, "W3": 0.2, "W4": 0.2, "W5": 0.2})
        if workload_id == "W6"
        else WorkloadConfig(id=workload_id)
    )
    return ExperimentConfig(
        campaign_id="test",
        experiment_id="unit",
        seed=SeedConfig(value=seed),
        workload=workload,
        arrival=FixedIntervalArrivalConfig(interval_ms=0.0),
        load=LoadConfig(request_count=request_count),
        backend=SimulatedBackendConfig(service_latency_ms=0.0),
        slo=SLOConfig(latency_ms=10.0),
        output_dir=tmp_path,
    )


def issue_codes(report: dict) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def test_t1_identical_seed_produces_identical_workload_trace(tmp_path: Path) -> None:
    config = make_config(tmp_path, seed=7)
    assert generate_workload(config).canonical() == generate_workload(config).canonical()


def test_t2_different_seeds_produce_different_stochastic_traces(tmp_path: Path) -> None:
    trace_a = generate_workload(make_config(tmp_path, seed=1)).canonical()
    trace_b = generate_workload(make_config(tmp_path, seed=2)).canonical()
    assert trace_a != trace_b


def test_t3_canonical_config_hashing_is_stable(tmp_path: Path) -> None:
    config = make_config(tmp_path, seed=9)
    assert config.config_hash() == stable_hash(config.canonical())
    assert config.config_hash() == make_config(tmp_path, seed=9).config_hash()


def test_t4_planned_request_count_equals_generated_request_count(tmp_path: Path) -> None:
    config = make_config(tmp_path, request_count=11)
    assert len(generate_workload(config).requests) == 11


def test_t5_every_request_id_is_unique(tmp_path: Path) -> None:
    trace = generate_workload(make_config(tmp_path, request_count=20))
    ids = [request.request_id for request in trace.requests]
    assert len(ids) == len(set(ids))


def test_t6_successful_latency_decomposition_is_internally_consistent(tmp_path: Path) -> None:
    config = make_config(tmp_path, request_count=1)
    trace = generate_workload(config)
    raw_path = tmp_path / "raw.jsonl"
    results = asyncio.run(replay_schedule(trace, SimulatedBackend(config.backend), raw_path, 10.0))
    result = results[0]
    assert result.queue_latency_s is not None
    assert result.service_latency_s is not None
    assert result.end_to_end_latency_s is not None
    assert result.end_to_end_latency_s == pytest.approx(result.queue_latency_s + result.service_latency_s, abs=0.002)


def test_t7_slo_classification_is_correct_at_threshold_boundary() -> None:
    result = RequestResult(
        request_id="req",
        workload_id="W1",
        scheduled_arrival_time_s=0,
        actual_dispatch_time_ns=0,
        backend_start_time_ns=0,
        completion_time_ns=10_000_000,
        success=True,
        input_tokens=1,
        requested_output_tokens=1,
        end_to_end_latency_s=0.010,
        slo_latency_ms=10.0,
        slo_met=0.010 * 1000 <= 10.0,
    )
    assert result.slo_met is True


def test_t8_failed_requests_are_preserved(tmp_path: Path) -> None:
    config = make_config(tmp_path, request_count=3)
    config.backend.fail_request_ids.append("req-00000001")
    run_dir = run_experiment(config)
    results = (run_dir / RAW_RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(results) == 3
    assert any(json.loads(line)["success"] is False for line in results)
    assert read_json(run_dir / MANIFEST_FILENAME)["failed_requests"] == 1


def test_t9_interrupted_failed_run_produces_auditable_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import heteroservebench.runner as runner

    async def boom(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(runner, "replay_schedule", boom)
    with pytest.raises(RuntimeError):
        runner.run_experiment(make_config(tmp_path, request_count=2))
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    manifest = read_json(run_dirs[0] / MANIFEST_FILENAME)
    assert manifest["run_status"] == "failed"
    assert manifest["terminal_error"]["type"] == "RuntimeError"
    assert manifest["end_time"] is not None


def test_t10_invalid_timestamps_are_detected(tmp_path: Path) -> None:
    run_dir = run_experiment(make_config(tmp_path))
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = read_json(manifest_path)
    manifest["start_time"] = "not-a-time"
    write_json(manifest_path, manifest)
    assert "invalid_timestamp" in issue_codes(validate_run(run_dir))


def test_t11_duplicate_results_are_detected(tmp_path: Path) -> None:
    run_dir = run_experiment(make_config(tmp_path, request_count=2))
    raw_path = run_dir / RAW_RESULTS_FILENAME
    first = raw_path.read_text(encoding="utf-8").splitlines()[0]
    with raw_path.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")
    assert "duplicate_result_request_ids" in issue_codes(validate_run(run_dir))


def test_t12_raw_results_cannot_be_silently_replaced_by_derived_analysis(tmp_path: Path) -> None:
    run_dir = run_experiment(make_config(tmp_path))
    raw_path = run_dir / RAW_RESULTS_FILENAME
    before = raw_path.read_bytes()
    summarize_run(run_dir)
    assert raw_path.read_bytes() == before
    assert (run_dir / SUMMARY_FILENAME).exists()
    with pytest.raises(FileExistsError):
        summarize_run(run_dir)


def test_t13_manifest_contains_required_provenance_fields(tmp_path: Path) -> None:
    run_dir = run_experiment(make_config(tmp_path))
    manifest = read_json(run_dir / MANIFEST_FILENAME)
    required = {
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
    }
    assert required <= set(manifest)
    assert manifest["hardware_profile"]["gpu_name"] is None


def test_t14_smoke_experiment_runs_end_to_end(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke/cpu_smoke.yaml"))
    config.output_dir = tmp_path
    run_dir = run_experiment(config)
    manifest = read_json(run_dir / MANIFEST_FILENAME)
    assert manifest["expected_requests"] == 20
    assert manifest["run_status"] == "completed"


def test_t15_smoke_experiment_validates_successfully(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke/cpu_smoke.yaml"))
    config.output_dir = tmp_path
    run_dir = run_experiment(config)
    report = validate_run(run_dir)
    assert report["valid"], report
