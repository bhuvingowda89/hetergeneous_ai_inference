"""Derived summary metrics computed from raw observations."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from heteroservebench.io import MANIFEST_FILENAME, RAW_RESULTS_FILENAME, SUMMARY_FILENAME, read_raw_results
from heteroservebench.serialization import read_json, write_json


def _percentile(values: list[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_run(run_dir: Path, write: bool = True) -> dict:
    """Compute basic derived metrics without modifying raw observations."""
    manifest = read_json(run_dir / MANIFEST_FILENAME)
    results = read_raw_results(run_dir / RAW_RESULTS_FILENAME)
    successes = [result for result in results if result.success]
    latencies = [result.end_to_end_latency_s for result in successes if result.end_to_end_latency_s is not None]
    duration_s = None
    if results:
        starts = [r.actual_dispatch_time_ns for r in results if r.actual_dispatch_time_ns is not None]
        ends = [r.completion_time_ns for r in results if r.completion_time_ns is not None]
        if starts and ends and max(ends) >= min(starts):
            duration_s = (max(ends) - min(starts)) / 1_000_000_000
    summary = {
        "run_id": manifest.get("run_id"),
        "request_count": len(results),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "throughput_requests_per_second": None
        if not duration_s or duration_s <= 0
        else len(successes) / duration_s,
        "latency_s": {
            "mean": None if not latencies else mean(latencies),
            "p50": None if not latencies else _percentile(latencies, 50),
            "p95": None if not latencies else _percentile(latencies, 95),
            "p99": None if not latencies else _percentile(latencies, 99),
        },
        "slo_attainment": None
        if not any(result.slo_met is not None for result in results)
        else sum(1 for result in results if result.slo_met) / len(results),
    }
    if write:
        summary_path = run_dir / SUMMARY_FILENAME
        if summary_path.exists():
            raise FileExistsError(f"summary already exists and will not be overwritten: {summary_path}")
        write_json(summary_path, summary)
    return summary
