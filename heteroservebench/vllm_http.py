"""OpenAI-compatible vLLM HTTP client helpers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StreamingParseResult:
    """Parsed streaming completion timing and usage information."""

    first_token_time_ns: Optional[int] = None
    completion_time_ns: Optional[int] = None
    token_event_time_ns: list[int] = field(default_factory=list)
    text_chunks: list[str] = field(default_factory=list)
    generated_tokens: Optional[int] = None
    raw_usage: Optional[dict] = None

    @property
    def text(self) -> str:
        return "".join(self.text_chunks)

    def ttft_s(self, start_ns: int) -> Optional[float]:
        if self.first_token_time_ns is None:
            return None
        return (self.first_token_time_ns - start_ns) / 1_000_000_000

    def inter_token_latency_s(self) -> Optional[float]:
        if len(self.token_event_time_ns) < 2:
            return None
        intervals = [
            (right - left) / 1_000_000_000
            for left, right in zip(self.token_event_time_ns, self.token_event_time_ns[1:])
        ]
        return sum(intervals) / len(intervals)


def parse_sse_data_line(line: bytes) -> Optional[dict]:
    """Parse one OpenAI streaming `data:` line."""
    text = line.decode("utf-8").strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    if text == "[DONE]":
        return {"done": True}
    return json.loads(text)


def observe_stream_event(result: StreamingParseResult, event: dict, timestamp_ns: int) -> None:
    """Update streaming parse state from one OpenAI-compatible event."""
    if event.get("done"):
        result.completion_time_ns = timestamp_ns
        return
    usage = event.get("usage")
    if isinstance(usage, dict):
        result.raw_usage = usage
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            result.generated_tokens = completion_tokens

    choices = event.get("choices") or []
    for choice in choices:
        text = choice.get("text")
        if text is None:
            delta = choice.get("delta") or {}
            text = delta.get("content")
        if text:
            if result.first_token_time_ns is None:
                result.first_token_time_ns = timestamp_ns
            result.token_event_time_ns.append(timestamp_ns)
            result.text_chunks.append(text)


class VllmHttpClient:
    """Small stdlib HTTP client for vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def models(self) -> dict:
        """Fetch `/v1/models`."""
        request = urllib.request.Request(f"{self.base_url}/v1/models", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def stream_completion(self, payload: dict[str, Any]) -> StreamingParseResult:
        """Submit a streaming completion request and parse token timing."""
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        result = StreamingParseResult()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            for line in response:
                event = parse_sse_data_line(line)
                if event is None:
                    continue
                now_ns = time.monotonic_ns()
                observe_stream_event(result, event, now_ns)
                if event.get("done"):
                    break
        if result.completion_time_ns is None:
            result.completion_time_ns = time.monotonic_ns()
        return result


def http_error_type(exc: Exception) -> str:
    """Return a stable error type for HTTP failures."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return "URLError"
    return type(exc).__name__
