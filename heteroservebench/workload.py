"""Deterministic benchmark workload trace generation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from heteroservebench.config import ExperimentConfig
from heteroservebench.serialization import stable_hash


CANONICAL_WORKLOADS: dict[str, tuple[int, int]] = {
    "W1": (128, 128),
    "W2": (512, 128),
    "W3": (2048, 128),
    "W4": (128, 512),
    "W5": (512, 512),
}


class BenchmarkRequest(BaseModel):
    """Planned request metadata independent of any tokenizer or backend."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    workload_id: str
    input_tokens: int = Field(ge=0)
    requested_output_tokens: int = Field(ge=0)
    scheduled_arrival_time_s: float = Field(ge=0)
    seed_metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadTrace:
    """A generated request schedule and its stable hash."""

    requests: list[BenchmarkRequest]

    def canonical(self) -> list[dict[str, Any]]:
        return [request.model_dump(mode="json") for request in self.requests]

    def hash(self) -> str:
        return stable_hash(self.canonical())


def _arrival_times(config: ExperimentConfig, rng: random.Random) -> list[float]:
    count = config.load.request_count
    if config.arrival.type == "fixed_interval":
        interval_s = config.arrival.interval_ms / 1000.0
        return [i * interval_s for i in range(count)]
    if config.arrival.type == "poisson":
        elapsed = 0.0
        arrivals: list[float] = []
        for index in range(count):
            if index > 0:
                elapsed += rng.expovariate(config.arrival.rate_per_second)
            arrivals.append(round(elapsed, 12))
        return arrivals
    raise ValueError(f"unsupported arrival process: {config.arrival.type}")


def _workload_ids(config: ExperimentConfig, rng: random.Random) -> list[str]:
    count = config.load.request_count
    if config.workload.id != "W6":
        return [config.workload.id] * count
    assert config.workload.probabilities is not None
    ids = ["W1", "W2", "W3", "W4", "W5"]
    weights = [config.workload.probabilities[item] for item in ids]
    return rng.choices(ids, weights=weights, k=count)


def generate_workload(config: ExperimentConfig) -> WorkloadTrace:
    """Generate a deterministic workload trace from configuration and seed."""
    rng = random.Random(config.seed.value)
    arrivals = _arrival_times(config, rng)
    workload_ids = _workload_ids(config, rng)
    requests: list[BenchmarkRequest] = []
    for index, (workload_id, arrival_s) in enumerate(zip(workload_ids, arrivals)):
        input_tokens, output_tokens = CANONICAL_WORKLOADS[workload_id]
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{index:08d}",
                workload_id=workload_id,
                input_tokens=input_tokens,
                requested_output_tokens=output_tokens,
                scheduled_arrival_time_s=arrival_s,
                seed_metadata={"seed": config.seed.value, "sequence_index": index},
            )
        )
    return WorkloadTrace(requests)
