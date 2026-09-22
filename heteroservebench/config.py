"""Validated configuration models for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from heteroservebench.serialization import stable_hash


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


class SeedConfig(StrictModel):
    value: int = Field(ge=0)


class SLOConfig(StrictModel):
    latency_ms: float = Field(gt=0)


class LoadConfig(StrictModel):
    request_count: int = Field(gt=0)


class FixedIntervalArrivalConfig(StrictModel):
    type: Literal["fixed_interval"] = "fixed_interval"
    interval_ms: float = Field(ge=0)


class PoissonArrivalConfig(StrictModel):
    type: Literal["poisson"] = "poisson"
    rate_per_second: float = Field(gt=0)


ArrivalConfig = Union[FixedIntervalArrivalConfig, PoissonArrivalConfig]


class BatchingConfig(StrictModel):
    max_batch_size: int = Field(default=1, gt=0)
    max_wait_ms: float = Field(default=0.0, ge=0)


class WorkloadConfig(StrictModel):
    id: Literal["W1", "W2", "W3", "W4", "W5", "W6"]
    probabilities: Optional[dict[Literal["W1", "W2", "W3", "W4", "W5"], float]] = None

    @model_validator(mode="after")
    def validate_probabilities(self) -> "WorkloadConfig":
        if self.id == "W6":
            if not self.probabilities:
                raise ValueError("W6 requires probabilities over W1-W5")
            expected = {"W1", "W2", "W3", "W4", "W5"}
            if set(self.probabilities) != expected:
                raise ValueError("W6 probabilities must specify exactly W1-W5")
            total = sum(self.probabilities.values())
            if any(v < 0 for v in self.probabilities.values()):
                raise ValueError("W6 probabilities must be non-negative")
            if abs(total - 1.0) > 1e-9:
                raise ValueError("W6 probabilities must sum to 1.0")
        elif self.probabilities is not None:
            raise ValueError("probabilities are only valid for W6")
        return self


class SimulatedBackendConfig(StrictModel):
    type: Literal["simulated"] = "simulated"
    service_latency_ms: float = Field(default=10.0, ge=0)
    ttft_ms: Optional[float] = Field(default=None, ge=0)
    inter_token_latency_ms: Optional[float] = Field(default=None, ge=0)
    generated_tokens: Optional[int] = Field(default=None, ge=0)
    fail_request_ids: list[str] = Field(default_factory=list)


class VllmBackendConfig(StrictModel):
    type: Literal["vllm"] = "vllm"
    endpoint: str
    model_id: str


BackendConfig = Union[SimulatedBackendConfig, VllmBackendConfig]


class ExperimentConfig(StrictModel):
    schema_version: str = "1.0"
    campaign_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    seed: SeedConfig
    workload: WorkloadConfig
    arrival: ArrivalConfig = Field(discriminator="type")
    load: LoadConfig
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    backend: BackendConfig = Field(discriminator="type")
    slo: Optional[SLOConfig] = None
    output_dir: Path

    @field_validator("output_dir")
    @classmethod
    def output_dir_not_empty(cls, value: Path) -> Path:
        if str(value).strip() == "":
            raise ValueError("output_dir cannot be empty")
        return value

    def canonical(self) -> dict:
        """Return JSON-compatible canonical configuration content."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """Return a stable hash of the canonical configuration."""
        return stable_hash(self.canonical())


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration file must contain a mapping")
    return ExperimentConfig.model_validate(data)
