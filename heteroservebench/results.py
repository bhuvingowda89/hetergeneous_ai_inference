"""Raw request measurement schema."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RequestResult(BaseModel):
    """Raw request-level observation with explicit seconds/nanoseconds fields."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    workload_id: str
    scheduled_arrival_time_s: float
    actual_dispatch_time_ns: Optional[int] = None
    backend_start_time_ns: Optional[int] = None
    first_token_time_ns: Optional[int] = None
    completion_time_ns: Optional[int] = None
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    input_tokens: int = Field(ge=0)
    requested_output_tokens: int = Field(ge=0)
    generated_tokens: Optional[int] = Field(default=None, ge=0)
    token_event_time_ns: list[int] = Field(default_factory=list)
    ttft_s: Optional[float] = Field(default=None, ge=0)
    inter_token_latency_s: Optional[float] = Field(default=None, ge=0)
    queue_latency_s: Optional[float] = Field(default=None, ge=0)
    service_latency_s: Optional[float] = Field(default=None, ge=0)
    end_to_end_latency_s: Optional[float] = Field(default=None, ge=0)
    slo_latency_ms: Optional[float] = Field(default=None, gt=0)
    slo_met: Optional[bool] = None
    failure_classification: Optional[Literal["backend_error", "oom", "timeout", "unknown"]] = None
    backend_metadata: dict = Field(default_factory=dict)
