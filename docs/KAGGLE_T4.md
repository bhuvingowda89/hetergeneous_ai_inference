# Kaggle Single-T4 Phase 3C Smoke Run

These commands validate infrastructure on one NVIDIA T4 only. They do not produce paper benchmark results or heterogeneity conclusions.

## 1. Clone And Install

```bash
git clone https://github.com/bhuvingowda89/hetergeneous_ai_inference.git
cd hetergeneous_ai_inference
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## 2. Verify One GPU

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
```

Kaggle may expose two T4 GPUs. Phase 3A intentionally uses only device `0`.

## 3. Install GPU Dependencies

```bash
python -m pip install -e ".[gpu]"
```

The GPU extra pins the verified serving engine version: `vLLM 0.27.1`.

## 4. Verify vLLM

```bash
vllm serve --help
```

## 5. Start One-GPU vLLM Server

```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --host 127.0.0.1 \
  --port 8000
```

The first launch downloads/loads `Qwen/Qwen3-4B-Instruct-2507`. Keep this server running.
On Tesla T4, vLLM uses the TRITON_ATTN fallback because FlashAttention-2 is not available for compute capability 7.5.

## 6. Verify The Server

In another notebook cell or terminal:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

## 7. Run The GPU Smoke Benchmark

```bash
export CUDA_VISIBLE_DEVICES=0
python -m heteroservebench run --config configs/gpu/t4_qwen3_4b_smoke.yaml
```

The command prints a `runs/<run_id>` directory.
The config records the externally managed `vllm serve ...` command verbatim in `model_provenance.serving_engine_command`; HeteroServeBench is only the benchmark client and does not claim to launch the server.

## 8. Validate And Summarize

```bash
RUN_DIR=runs/<run_id>
python -m heteroservebench validate --run-dir "$RUN_DIR"
python -m heteroservebench summarize --run-dir "$RUN_DIR"
```

Smoke/exploratory validation permits unresolved model revision hashes with a warning. For scientific runs, set `validation_mode: "scientific"` and configure a real `backend.requested_model_revision`; validation fails if `resolved_model_revision_hash` is absent.

Every GPU request must use an exact tokenizer-level prompt length. Scientific vLLM workloads also use fixed completion lengths with `exact_output_tokens: true`; the client sends each request's canonical output length as `max_tokens` and `min_tokens` with `ignore_eos` enabled. Raw results record `requested_input_tokens`, `actual_prompt_tokens`, and `provider_prompt_tokens` when vLLM reports usage. Scientific/GPU validation fails token-count mismatches instead of silently accepting approximations.

## 9. Archive Artifacts

```bash
tar -czf phase3a_t4_smoke_artifacts.tgz "$RUN_DIR"
```

Preserve `manifest.json`, `planned_workload.json`, `raw_results.jsonl`, `warmup_results.jsonl`, `telemetry.jsonl`, and `summary.json` when present.
