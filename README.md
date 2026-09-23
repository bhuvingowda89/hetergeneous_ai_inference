# HeteroServeBench

HeteroServeBench is research infrastructure for the paper **“HeteroServeBench: Characterizing Heterogeneity Break-Even Regimes for SLO-Constrained LLM Inference.”**

This repository currently implements Phase 2 infrastructure only. It does not implement heterogeneity break-even analysis, GPU scheduling policy comparisons, routing algorithms, or paper conclusions. No benchmark result in this phase should be presented as LLM serving performance evidence.

## What Phase 2 Implements

- Validated, canonical, hashable experiment configuration.
- Deterministic workload traces for W1-W6 token-length metadata.
- Fixed-interval and Poisson arrival processes.
- A backend abstraction with a CPU-safe deterministic simulated backend.
- A vLLM adapter stub that does not require vLLM installation.
- Asynchronous schedule replay with per-request raw observations.
- Immutable raw results separated from derived summaries.
- Run manifests with provenance and null GPU fields for CPU runs.
- Structured validation and basic summary metrics.

## What Is Simulated

The `simulated` backend sleeps for a configured service latency and emits request-level timing records. It is only an infrastructure test backend. CPU simulated measurements must never be presented as paper performance results, GPU measurements, or model-serving measurements.

## Setup

Python 3.11+ is recommended; the infrastructure is kept compatible with Python 3.9+ for CPU-only artifact portability.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run The CPU Smoke Benchmark

```bash
python -m heteroservebench run --config configs/smoke/cpu_smoke.yaml
```

The command prints the created run directory under `runs/`. Raw request observations are written to `raw_results.jsonl`; manifests and planned traces are separate JSON files.

## Validate A Run

```bash
python -m heteroservebench validate --run-dir runs/<run_id>
```

Validation exits nonzero when required provenance, trace, timing, or consistency checks fail.

## Summarize A Run

```bash
python -m heteroservebench summarize --run-dir runs/<run_id>
```

Summaries include only basic derived metrics: counts, throughput, latency percentiles, mean latency, and SLO attainment when configured. Analysis code refuses to overwrite existing summary files and never rewrites raw results.

## Phase 3C GPU Smoke And Reproducibility

GPU/vLLM support is optional and uses vLLM's OpenAI-compatible HTTP server. The base package still runs CPU tests without CUDA, PyTorch, or vLLM.

```bash
pip install -e ".[gpu]"
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --host 127.0.0.1 \
  --port 8000
python -m heteroservebench run --config configs/gpu/t4_qwen3_4b_smoke.yaml
```

Phase 3C pins the GPU extra to the verified serving engine `vLLM 0.27.1`. On Tesla T4, vLLM falls back to TRITON_ATTN because FlashAttention-2 is unavailable on compute capability 7.5. Use `CUDA_VISIBLE_DEVICES=0` to isolate one benchmark-visible GPU even when Kaggle exposes two physical T4s.

Scientific GPU runs require exact tokenizer-level prompts. Each request records `requested_input_tokens`, `actual_prompt_tokens`, and, when vLLM reports it, `provider_prompt_tokens`; validation fails if these disagree. Scientific runs must also pin a real Hugging Face model revision and resolve a snapshot hash in the manifest. Smoke/exploratory runs may leave the revision unresolved, but validation reports a warning.

See `docs/KAGGLE_T4.md` for the full single-T4 Kaggle procedure. Smoke runs validate infrastructure only, not scientific benchmarking.

## Repository Organization

- `heteroservebench/config.py` - validated configuration schemas.
- `heteroservebench/workload.py` - deterministic request trace generation.
- `heteroservebench/backend.py` - backend interface, simulator, and vLLM stub.
- `heteroservebench/gpu.py` - NVIDIA GPU discovery helpers.
- `heteroservebench/telemetry.py` - best-effort GPU/host telemetry sampling.
- `heteroservebench/vllm_http.py` - OpenAI-compatible streaming vLLM HTTP client.
- `heteroservebench/runner.py` - asynchronous load generation and run orchestration.
- `heteroservebench/results.py` - raw request observation schema.
- `heteroservebench/manifest.py` - provenance manifest generation.
- `heteroservebench/metrics.py` - basic derived summaries.
- `heteroservebench/validation.py` - structured run validation.
- `configs/smoke/cpu_smoke.yaml` - small deterministic CPU smoke run.
- `tests/` - acceptance and regression tests.
