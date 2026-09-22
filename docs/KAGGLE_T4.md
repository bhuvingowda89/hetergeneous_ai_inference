# Kaggle Single-T4 Phase 3A Smoke Run

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

## 4. Verify vLLM

```bash
python -m vllm.entrypoints.openai.api_server --help
```

## 5. Start One-GPU vLLM Server

```bash
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --port 8000
```

The first launch downloads/loads `Qwen/Qwen3-4B-Instruct-2507`. Keep this server running.

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

## 8. Validate And Summarize

```bash
RUN_DIR=runs/<run_id>
python -m heteroservebench validate --run-dir "$RUN_DIR"
python -m heteroservebench summarize --run-dir "$RUN_DIR"
```

## 9. Archive Artifacts

```bash
tar -czf phase3a_t4_smoke_artifacts.tgz "$RUN_DIR"
```

Preserve `manifest.json`, `planned_workload.json`, `raw_results.jsonl`, `warmup_results.jsonl`, `telemetry.jsonl`, and `summary.json` when present.
