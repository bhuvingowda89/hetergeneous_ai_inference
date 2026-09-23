"""Backend abstraction and CPU-only simulated implementation."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from heteroservebench.config import BackendConfig, SimulatedBackendConfig, VllmBackendConfig
from heteroservebench.results import RequestResult
from heteroservebench.tokenizer_prompt import (
    PromptConstructionError,
    construct_exact_prompt,
    load_tokenizer,
    tokenizer_identifier,
)
from heteroservebench.vllm_http import VllmHttpClient, http_error_type
from heteroservebench.workload import BenchmarkRequest


class BackendError(RuntimeError):
    """Backend request failure."""


EXACT_OUTPUT_CONTROL_FIELDS = {"max_tokens", "min_tokens", "ignore_eos"}


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
                requested_input_tokens=request.input_tokens,
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
            requested_input_tokens=request.input_tokens,
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
    """vLLM OpenAI-compatible HTTP backend."""

    backend_type = "vllm"
    is_simulated = False

    def __init__(self, config: VllmBackendConfig) -> None:
        self.config = config
        self.client = VllmHttpClient(config.base_url, config.request_timeout_s)

    async def infer(self, request: BenchmarkRequest) -> RequestResult:
        start_ns = time.monotonic_ns()
        actual_prompt_tokens: int | None = None
        try:
            payload, actual_prompt_tokens = self._payload(request)
        except PromptConstructionError as exc:
            return RequestResult(
                request_id=request.request_id,
                workload_id=request.workload_id,
                scheduled_arrival_time_s=request.scheduled_arrival_time_s,
                backend_start_time_ns=start_ns,
                completion_time_ns=time.monotonic_ns(),
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                input_tokens=request.input_tokens,
                requested_input_tokens=request.input_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                requested_output_tokens=request.requested_output_tokens,
                failure_classification="backend_error",
                backend_metadata=self.metadata(),
            )
        try:
            parsed = await asyncio.to_thread(self.client.stream_completion, payload)
        except Exception as exc:
            return RequestResult(
                request_id=request.request_id,
                workload_id=request.workload_id,
                scheduled_arrival_time_s=request.scheduled_arrival_time_s,
                backend_start_time_ns=start_ns,
                completion_time_ns=time.monotonic_ns(),
                success=False,
                error_type=http_error_type(exc),
                error_message=str(exc),
                input_tokens=request.input_tokens,
                requested_input_tokens=request.input_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                requested_output_tokens=request.requested_output_tokens,
                failure_classification="backend_error",
                backend_metadata=self.metadata(),
            )

        completion_ns = parsed.completion_time_ns or time.monotonic_ns()
        return RequestResult(
            request_id=request.request_id,
            workload_id=request.workload_id,
            scheduled_arrival_time_s=request.scheduled_arrival_time_s,
            backend_start_time_ns=start_ns,
            first_token_time_ns=parsed.first_token_time_ns,
            completion_time_ns=completion_ns,
            success=True,
            input_tokens=request.input_tokens,
            requested_input_tokens=request.input_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
            provider_prompt_tokens=parsed.provider_prompt_tokens,
            requested_output_tokens=request.requested_output_tokens,
            generated_tokens=parsed.generated_tokens,
            token_event_time_ns=parsed.token_event_time_ns,
            ttft_s=parsed.ttft_s(start_ns),
            inter_token_latency_s=parsed.inter_token_latency_s(),
            service_latency_s=(completion_ns - start_ns) / 1_000_000_000,
            backend_metadata=self.metadata(),
        )

    def _payload(self, request: BenchmarkRequest) -> tuple[dict, int]:
        prompt = self._prompt(request)
        exact_output_overrides = EXACT_OUTPUT_CONTROL_FIELDS.intersection(self.config.extra_body)
        if self.config.exact_output_tokens and exact_output_overrides:
            fields = ", ".join(sorted(exact_output_overrides))
            raise BackendError(f"extra_body cannot override exact-output controls: {fields}")
        max_tokens = min(self.config.max_tokens, request.requested_output_tokens)
        if self.config.exact_output_tokens:
            max_tokens = request.requested_output_tokens
        payload = {
            "model": self.config.model,
            "prompt": prompt.text,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "stream": self.config.stream,
            "stream_options": {"include_usage": True},
        }
        if self.config.exact_output_tokens:
            payload["min_tokens"] = request.requested_output_tokens
            payload["ignore_eos"] = True
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        payload.update(self.config.extra_body)
        return payload, prompt.actual_tokens

    def _prompt(self, request: BenchmarkRequest):
        identifier = tokenizer_identifier(self.config.model, self.config.tokenizer)
        tokenizer = load_tokenizer(identifier, self.config.requested_model_revision)
        return construct_exact_prompt(tokenizer, request.input_tokens)

    def metadata(self) -> dict:
        return {
            "type": self.backend_type,
            "is_simulated": self.is_simulated,
            "base_url": self.config.base_url,
            "model_id": self.config.model,
            "requested_model_revision": self.config.requested_model_revision,
            "tokenizer": self.config.tokenizer,
            "dtype": self.config.dtype,
            "quantization": self.config.quantization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_model_len": self.config.max_model_len,
            "serving_engine": "vllm",
            "serving_mode": self.config.serving_mode,
            "serving_engine_command": self.config.serving_engine_command,
            "streaming": self.config.stream,
        }


def create_backend(config: BackendConfig) -> Backend:
    """Construct a backend from validated configuration."""
    if config.type == "simulated":
        return SimulatedBackend(config)
    if config.type == "vllm":
        return VllmBackend(config)
    raise ValueError(f"unsupported backend type: {config.type}")
