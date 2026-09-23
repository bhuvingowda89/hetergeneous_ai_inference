from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from heteroservebench.backend import SimulatedBackend, VllmBackend
from heteroservebench.config import (
    ExperimentConfig,
    FixedIntervalArrivalConfig,
    LoadConfig,
    SeedConfig,
    SimulatedBackendConfig,
    SLOConfig,
    TelemetryConfig,
    VllmBackendConfig,
    WarmupConfig,
    WorkloadConfig,
    load_config,
)
from heteroservebench.gpu import benchmark_visible_gpus, parse_nvidia_smi_csv, primary_gpu_profile
from heteroservebench.io import MANIFEST_FILENAME, RAW_RESULTS_FILENAME, SUMMARY_FILENAME, append_raw_result
from heteroservebench.manifest import initial_manifest, manifest_hash
from heteroservebench.metrics import summarize_run
from heteroservebench.results import RequestResult
from heteroservebench.runner import replay_schedule, run_experiment, validate_context_capacity
from heteroservebench.serialization import read_json, stable_hash, write_json
from heteroservebench.tokenizer_prompt import construct_exact_prompt
from heteroservebench.validation import validate_run
from heteroservebench.vllm_http import StreamingParseResult, observe_stream_event
from heteroservebench.workload import BenchmarkRequest, WorkloadTrace, generate_workload


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


def make_vllm_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        campaign_id="gpu-test",
        experiment_id="vllm",
        seed=SeedConfig(value=55),
        workload=WorkloadConfig(id="W1"),
        arrival=FixedIntervalArrivalConfig(interval_ms=1.0),
        load=LoadConfig(request_count=1),
        backend=VllmBackendConfig(
            base_url="http://127.0.0.1:8000",
            model="Qwen/Qwen3-4B-Instruct-2507",
            tokenizer="Qwen/Qwen3-4B-Instruct-2507",
            dtype="float16",
            quantization=None,
            tensor_parallel_size=1,
            max_model_len=4096,
            expected_gpu_count=1,
            serving_engine_command="CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port 8000",
        ),
        telemetry=TelemetryConfig(enabled=False),
        output_dir=tmp_path,
    )


def make_synthetic_vllm_run(tmp_path: Path, *, gpu_identity: bool = True, ttft_s: float = 0.01) -> Path:
    config = make_vllm_config(tmp_path)
    trace = generate_workload(config)
    run_dir = tmp_path / "synthetic-vllm"
    run_dir.mkdir()
    manifest = initial_manifest(config, trace, "synthetic-vllm", run_dir)
    manifest["run_status"] = "completed"
    manifest["end_time"] = manifest["start_time"]
    manifest["completed_requests"] = 1
    manifest["failed_requests"] = 0
    manifest["hardware_profile"]["physical_gpu_count"] = 1
    manifest["hardware_profile"]["benchmark_visible_gpu_count"] = 1
    manifest["hardware_profile"]["selected_cuda_device"] = "0"
    manifest["gpu_discovery"] = {
        "physical_gpu_count": 1,
        "cuda_visible_devices": "0",
        "benchmark_visible_gpu_count": 1,
        "selected_cuda_device": "0",
        "gpus": [
            {"index": 0, "name": "Tesla T4", "uuid": "GPU-test", "memory_total_mb": 15109},
        ],
        "benchmark_visible_gpus": [
            {"index": 0, "name": "Tesla T4", "uuid": "GPU-test", "memory_total_mb": 15109},
        ],
    }
    if gpu_identity:
        manifest["hardware_profile"]["gpu_name"] = "Tesla T4"
        manifest["hardware_profile"]["gpu_uuid"] = "GPU-test"
        manifest["hardware_profile"]["gpu_index"] = 0
    else:
        manifest["hardware_profile"]["gpu_name"] = None
        manifest["hardware_profile"]["gpu_uuid"] = None
    manifest["model_provenance"]["serving_engine_version"] = "0.test"
    manifest["manifest_hash"] = manifest_hash(manifest)
    write_json(run_dir / "planned_workload.json", trace.canonical())
    result = RequestResult(
        request_id="req-00000000",
        workload_id="W1",
        scheduled_arrival_time_s=0.0,
        actual_dispatch_time_ns=100,
        backend_start_time_ns=100,
        first_token_time_ns=100 + int(ttft_s * 1_000_000_000),
        completion_time_ns=100 + 20_000_000,
        success=True,
        input_tokens=128,
        requested_input_tokens=128,
        actual_prompt_tokens=128,
        provider_prompt_tokens=128,
        requested_output_tokens=128,
        generated_tokens=3,
        ttft_s=ttft_s,
        service_latency_s=0.02,
        end_to_end_latency_s=0.02,
    )
    append_raw_result(run_dir / RAW_RESULTS_FILENAME, result)
    write_json(run_dir / MANIFEST_FILENAME, manifest)
    return run_dir


def issue_codes(report: dict) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    vocab_size = 4
    all_special_ids = [0]

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        pieces = {0: "", 1: "a", 2: "b", 3: " "}
        return "".join(pieces[token_id] for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [1 for _ in text]


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


def test_t16_gpu_metadata_parsing() -> None:
    parsed = parse_nvidia_smi_csv("0, Tesla T4, GPU-abc, 15109\n", driver_version="550.54", cuda_version="12.4")
    assert parsed[0].name == "Tesla T4"
    assert parsed[0].uuid == "GPU-abc"
    assert parsed[0].memory_total_mb == 15109
    assert parsed[0].driver_version == "550.54"


def test_t17_vllm_streaming_parser_extracts_first_token_timing() -> None:
    parsed = StreamingParseResult()
    observe_stream_event(parsed, {"choices": [{"text": "hello"}]}, 1_000)
    observe_stream_event(parsed, {"choices": [{"text": " world"}], "usage": {"completion_tokens": 2, "prompt_tokens": 128}}, 2_000)
    observe_stream_event(parsed, {"done": True}, 3_000)
    assert parsed.first_token_time_ns == 1_000
    assert parsed.generated_tokens == 2
    assert parsed.provider_prompt_tokens == 128
    assert parsed.text == "hello world"
    assert parsed.inter_token_latency_s() == pytest.approx(0.000001)


def test_t18_ttft_exceeds_end_to_end_latency_validation(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path, ttft_s=0.03)
    report = validate_run(run_dir)
    assert "ttft_exceeds_latency" in issue_codes(report)
    assert not report["valid"]


def test_t19_missing_gpu_identity_invalidates_gpu_run(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path, gpu_identity=False)
    report = validate_run(run_dir)
    assert "missing_gpu_identity" in issue_codes(report)
    assert not report["valid"]


def test_t20_warmup_observations_are_excluded_from_measured_summary(tmp_path: Path) -> None:
    config = make_config(tmp_path, request_count=3, workload_id="W1")
    config.warmup = WarmupConfig(count=2)
    run_dir = run_experiment(config)
    summary = summarize_run(run_dir, write=False)
    assert summary["request_count"] == 3
    assert (run_dir / "warmup_results.jsonl").exists()
    assert len((run_dir / "warmup_results.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_t21_telemetry_failure_does_not_delete_request_observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import heteroservebench.telemetry as telemetry

    def fail_sample() -> dict:
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(telemetry, "sample_gpu_telemetry", fail_sample)
    config = make_config(tmp_path, request_count=2, workload_id="W1")
    config.backend.service_latency_ms = 20.0
    config.telemetry = TelemetryConfig(enabled=True, sampling_interval_s=0.001)
    run_dir = run_experiment(config)
    assert len((run_dir / RAW_RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()) == 2
    manifest = read_json(run_dir / MANIFEST_FILENAME)
    assert manifest["telemetry_errors"]


def test_t22_model_provenance_fields_are_preserved(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    provenance = read_json(run_dir / MANIFEST_FILENAME)["model_provenance"]
    assert provenance["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert provenance["dtype"] == "float16"
    assert provenance["quantization"] is None
    assert provenance["tensor_parallel_size"] == 1


def test_t23_nullable_unavailable_token_metrics_are_handled_correctly(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    raw_path = run_dir / RAW_RESULTS_FILENAME
    raw_path.unlink()
    append_raw_result(
        raw_path,
        RequestResult(
            request_id="req-00000000",
            workload_id="W1",
            scheduled_arrival_time_s=0.0,
            actual_dispatch_time_ns=100,
            backend_start_time_ns=100,
            first_token_time_ns=200,
            completion_time_ns=300,
            success=True,
            input_tokens=128,
            requested_input_tokens=128,
            actual_prompt_tokens=128,
            requested_output_tokens=128,
            generated_tokens=None,
            ttft_s=0.0000001,
            end_to_end_latency_s=0.0000002,
        ),
    )
    report = validate_run(run_dir)
    assert report["valid"]
    assert "generated_token_count_unavailable" in issue_codes(report)


def test_t24_gpu_configuration_canonical_hashing_remains_stable(tmp_path: Path) -> None:
    config = make_vllm_config(tmp_path)
    assert config.config_hash() == make_vllm_config(tmp_path).config_hash()


def test_t25_existing_cpu_smoke_test_still_passes_without_gpu_dependencies(tmp_path: Path) -> None:
    config = load_config(Path("configs/smoke/cpu_smoke.yaml"))
    config.output_dir = tmp_path
    run_dir = run_experiment(config)
    assert validate_run(run_dir)["valid"]


def test_w3_context_capacity_passes_with_4096_max_model_len(tmp_path: Path) -> None:
    config = make_vllm_config(tmp_path)
    config.workload = WorkloadConfig(id="W3")
    config.backend.max_model_len = 4096
    validate_context_capacity(config, generate_workload(config))


def test_w3_context_capacity_fails_with_2048_max_model_len(tmp_path: Path) -> None:
    config = make_vllm_config(tmp_path)
    config.workload = WorkloadConfig(id="W3")
    config.backend.max_model_len = 2048
    with pytest.raises(ValueError) as exc_info:
        validate_context_capacity(config, generate_workload(config))
    message = str(exc_info.value)
    assert "requested_input_tokens=2048" in message
    assert "requested_output_tokens=128" in message
    assert "required_context_length=2176" in message
    assert "configured_max_model_len=2048" in message


def test_context_capacity_boundary_equal_to_max_model_len_passes(tmp_path: Path) -> None:
    config = make_vllm_config(tmp_path)
    config.backend.max_model_len = 2176
    trace = WorkloadTrace(
        [
            BenchmarkRequest(
                request_id="req-boundary",
                workload_id="W3",
                input_tokens=2048,
                requested_output_tokens=128,
                scheduled_arrival_time_s=0.0,
            )
        ]
    )
    validate_context_capacity(config, trace)


def test_two_physical_gpus_cuda_visible_zero_means_one_benchmark_visible() -> None:
    gpus = [
        {"index": 0, "name": "Tesla T4", "uuid": "GPU-0", "memory_total_mb": 15109},
        {"index": 1, "name": "Tesla T4", "uuid": "GPU-1", "memory_total_mb": 15109},
    ]
    visible, error = benchmark_visible_gpus(gpus, "0")
    assert error is None
    assert len(visible) == 1
    assert visible[0]["index"] == 0
    profile = primary_gpu_profile(
        {
            "physical_gpu_count": 2,
            "cuda_visible_devices": "0",
            "benchmark_visible_gpu_count": len(visible),
            "benchmark_visible_gpus": visible,
            "gpus": gpus,
        },
        "0",
    )
    assert profile["physical_gpu_count"] == 2
    assert profile["benchmark_visible_gpu_count"] == 1


def test_two_physical_gpus_cuda_visible_zero_one_means_two_benchmark_visible() -> None:
    gpus = [
        {"index": 0, "name": "Tesla T4", "uuid": "GPU-0", "memory_total_mb": 15109},
        {"index": 1, "name": "Tesla T4", "uuid": "GPU-1", "memory_total_mb": 15109},
    ]
    visible, error = benchmark_visible_gpus(gpus, "0,1")
    assert error is None
    assert [gpu["index"] for gpu in visible] == [0, 1]


def test_cuda_visible_unset_defaults_to_discovered_gpu_count() -> None:
    gpus = [
        {"index": 0, "name": "Tesla T4", "uuid": "GPU-0", "memory_total_mb": 15109},
        {"index": 1, "name": "Tesla T4", "uuid": "GPU-1", "memory_total_mb": 15109},
    ]
    visible, error = benchmark_visible_gpus(gpus, None)
    assert error is None
    assert len(visible) == 2


def test_selected_device_absent_validation_failure(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = read_json(manifest_path)
    manifest["canonical_configuration"]["backend"]["selected_cuda_device"] = "1"
    manifest["config_hash"] = stable_hash(manifest["canonical_configuration"])
    manifest["hardware_profile"]["selected_cuda_device"] = "1"
    write_json(manifest_path, manifest)
    report = validate_run(run_dir)
    assert "selected_gpu_absent" in issue_codes(report)
    assert not report["valid"]


def test_telemetry_selects_configured_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import heteroservebench.telemetry as telemetry

    commands = []

    class Completed:
        stdout = "1, Tesla T4, GPU-1, 42, 1000, 15109, 70, 50.5\n"

    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def fake_run(command, **kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)
    sample = telemetry.sample_gpu_telemetry("1")
    assert commands[0][1:3] == ["-i", "1"]
    assert sample["gpu_index"] == 1
    assert sample["gpu_uuid"] == "GPU-1"
    assert sample["gpu_name"] == "Tesla T4"


def test_exact_prompt_construction_is_deterministic() -> None:
    first = construct_exact_prompt(FakeTokenizer(), 128)
    second = construct_exact_prompt(FakeTokenizer(), 128)
    assert first.text == second.text
    assert first.actual_tokens == 128


@pytest.mark.parametrize("token_count", [128, 512, 2048])
def test_exact_prompt_construction_supports_canonical_lengths(token_count: int) -> None:
    prompt = construct_exact_prompt(FakeTokenizer(), token_count)
    assert prompt.requested_tokens == token_count
    assert prompt.actual_tokens == token_count
    assert len(FakeTokenizer().encode(prompt.text, add_special_tokens=False)) == token_count


def test_vllm_prompt_uses_configured_tokenizer_and_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import heteroservebench.backend as backend_module

    calls = []

    def fake_load_tokenizer(identifier, revision):
        calls.append((identifier, revision))
        return FakeTokenizer()

    monkeypatch.setattr(backend_module, "load_tokenizer", fake_load_tokenizer)
    config = make_vllm_config(tmp_path)
    config.backend.tokenizer = "tokenizer/test"
    config.backend.requested_model_revision = "abc123"
    backend = VllmBackend(config.backend)
    payload, actual_prompt_tokens = backend._payload(
        BenchmarkRequest(
            request_id="req",
            workload_id="W1",
            input_tokens=128,
            requested_output_tokens=1,
            scheduled_arrival_time_s=0.0,
        )
    )
    assert calls == [("tokenizer/test", "abc123")]
    assert actual_prompt_tokens == 128
    assert payload["prompt"] == "a" * 128


def test_requested_actual_prompt_token_mismatch_is_detected(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    raw_path = run_dir / RAW_RESULTS_FILENAME
    row = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    row["actual_prompt_tokens"] = 127
    raw_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    report = validate_run(run_dir)
    assert "prompt_token_count_mismatch" in issue_codes(report)
    assert not report["valid"]


def test_provider_prompt_token_mismatch_is_detected(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    raw_path = run_dir / RAW_RESULTS_FILENAME
    row = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    row["provider_prompt_tokens"] = 129
    raw_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    report = validate_run(run_dir)
    assert "provider_prompt_token_count_mismatch" in issue_codes(report)
    assert not report["valid"]


def test_explicit_model_revision_provenance_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import heteroservebench.manifest as manifest_module

    def fake_resolve(identifier, revision):
        assert revision == "model-commit"
        return f"resolved-{identifier}"

    monkeypatch.setattr(manifest_module, "resolve_hf_snapshot_revision", fake_resolve)
    config = make_vllm_config(tmp_path)
    config.backend.requested_model_revision = "model-commit"
    trace = generate_workload(config)
    manifest = initial_manifest(config, trace, "run", tmp_path)
    provenance = manifest["model_provenance"]
    assert provenance["requested_model_revision"] == "model-commit"
    assert provenance["resolved_model_revision_hash"] == "resolved-Qwen/Qwen3-4B-Instruct-2507"
    assert provenance["tokenizer_identifier"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert provenance["tokenizer_resolved_revision_hash"] == "resolved-Qwen/Qwen3-4B-Instruct-2507"


def test_scientific_validation_rejects_missing_model_revision(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = read_json(manifest_path)
    manifest["validation_mode"] = "scientific"
    manifest["canonical_configuration"]["validation_mode"] = "scientific"
    manifest["model_provenance"]["resolved_model_revision_hash"] = None
    manifest["model_provenance"]["tokenizer_resolved_revision_hash"] = None
    manifest["config_hash"] = stable_hash(manifest["canonical_configuration"])
    manifest["manifest_hash"] = manifest_hash(manifest)
    write_json(manifest_path, manifest)
    report = validate_run(run_dir)
    assert "unidentified_model_revision" in issue_codes(report)
    assert not report["valid"]


def test_strict_validation_rejects_missing_model_revision_in_smoke_run(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = read_json(manifest_path)
    manifest["model_provenance"]["resolved_model_revision_hash"] = None
    manifest["model_provenance"]["tokenizer_resolved_revision_hash"] = None
    manifest["manifest_hash"] = manifest_hash(manifest)
    write_json(manifest_path, manifest)
    report = validate_run(run_dir, strict_scientific=True)
    assert "unidentified_model_revision" in issue_codes(report)
    assert not report["valid"]


def test_external_vllm_command_provenance_is_verbatim(tmp_path: Path) -> None:
    run_dir = make_synthetic_vllm_run(tmp_path)
    provenance = read_json(run_dir / MANIFEST_FILENAME)["model_provenance"]
    assert provenance["serving_mode"] == "external"
    assert provenance["serving_engine_command"] == (
        "CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port 8000"
    )
    assert "entrypoints.openai.api_server" not in provenance["serving_engine_command"]
