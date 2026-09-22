"""Run directory and raw observation IO helpers."""

from __future__ import annotations

import json
from pathlib import Path

from heteroservebench.results import RequestResult


RAW_RESULTS_FILENAME = "raw_results.jsonl"
MANIFEST_FILENAME = "manifest.json"
TRACE_FILENAME = "planned_workload.json"
SUMMARY_FILENAME = "summary.json"


def create_run_dir(output_dir: Path, run_id: str) -> Path:
    """Create a unique run directory without overwriting an existing run."""
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_raw_result(path: Path, result: RequestResult) -> None:
    """Append one raw result record as JSON Lines."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")


def read_raw_results(path: Path) -> list[RequestResult]:
    """Read raw JSONL results."""
    results: list[RequestResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(RequestResult.model_validate_json(line))
    return results


def refuse_to_overwrite_raw(path: Path) -> None:
    """Raise if a raw results file already exists."""
    if path.exists():
        raise FileExistsError(f"raw results already exist and will not be overwritten: {path}")
