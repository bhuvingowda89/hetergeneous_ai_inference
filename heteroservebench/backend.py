"""Backend abstraction and CPU-only simulated implementation."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from heteroservebench.config import BackendConfig, SimulatedBackendConfig, VllmBackendConfig
from heteroservebench.results import RequestResult
from heteroservebench.workload import BenchmarkRequest


class BackendError(RuntimeError):
    """Backend request failure."""


class Backend(ABC):
    """Generic inference backend interface."""

    backend_type: str
    is_simulated: bool

    @abstractmethod
    async def infer(self, request: BenchmarkRequest) -> RequestResult:
        """Execute one request and return a raw observation."""

    def metadata(self) -> dict:
        """Return backend metadata for manifests."""
        return {"type": self.backend_type, "is_simulated": self.is_simulated}


class SimulatedBackend(Backend):
    """Deterministic CPU-safe backend for infrastructure testing only."""

    backend_type = "simulated"
    is_simulated = True

    def __init__(self, config: SimulatedBackendConfig) -> None:
        self.config = config
        self._fail_request_ids = set(config.fail_request_ids)

    async def infer(self, request: BenchmarkRequest) -> RequestResult:
        start_ns = time.monotonic_ns()
        if request.request_id in self._fail_request_ids:
            await asyncio.sleep(0)
            completion_ns = time.monotonic_ns()
            return RequestResult(
                request_id=request.request_id,
                workload_id=request.workload_id,
                scheduled_arrival_time_s=request.scheduled_arrival_time_s,
                backend_start_time_ns=start_ns,
                completion_time_ns=completion_ns,
                success=False,
                error_type="SimulatedBackendError",
                error_message="configured simulated failure",
                input_tokens=request.input_tokens,
                requested_output_tokens=request.requested_output_tokens,
                failure_classification="backend_error",
                backend_metadata=self.metadata(),
            )

        await asyncio.sleep(self.config.service_latency_ms / 1000.0)
        completion_ns = time.monotonic_ns()
        service_latency_s = (completion_ns - start_ns) / 1_000_000_000
        generated_tokens = self.config.generated_tokens
        if generated_tokens is None:
            generated_tokens = request.requested_output_tokens
        return RequestResult(
            request_id=request.request_id,
            workload_id=request.workload_id,
            scheduled_arrival_time_s=request.scheduled_arrival_time_s,
            backend_start_time_ns=start_ns,
            completion_time_ns=completion_ns,
            success=True,
            input_tokens=request.input_tokens,
            requested_output_tokens=request.requested_output_tokens,
            generated_tokens=generated_tokens,
            ttft_s=None if self.config.ttft_ms is None else self.config.ttft_ms / 1000.0,
            inter_token_latency_s=None
            if self.config.inter_token_latency_ms is None
            else self.config.inter_token_latency_ms / 1000.0,
            service_latency_s=service_latency_s,
            backend_metadata=self.metadata(),
        )

    def metadata(self) -> dict:
        data = super().metadata()
        data.update(
            {
                "service_latency_ms": self.config.service_latency_ms,
                "note": "Simulated CPU-safe backend; not a GPU or model performance measurement.",
            }
        )
        return data


class VllmBackend(Backend):
    """Placeholder adapter for future vLLM integration."""

    backend_type = "vllm"
    is_simulated = False

    def __init__(self, config: VllmBackendConfig) -> None:
        self.config = config

    async def infer(self, request: BenchmarkRequest) -> RequestResult:
        raise NotImplementedError("vLLM execution is intentionally deferred to a later phase")

    def metadata(self) -> dict:
        return {
            "type": self.backend_type,
            "is_simulated": self.is_simulated,
            "endpoint": self.config.endpoint,
            "model_id": self.config.model_id,
            "adapter_status": "stub",
        }


def create_backend(config: BackendConfig) -> Backend:
    """Construct a backend from validated configuration."""
    if config.type == "simulated":
        return SimulatedBackend(config)
    if config.type == "vllm":
        return VllmBackend(config)
    raise ValueError(f"unsupported backend type: {config.type}")
