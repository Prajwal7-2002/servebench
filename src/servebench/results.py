"""Data models for configs and results. Everything in run.json goes through these."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TokenSource = Literal["usage", "tokenizer", "chunks"]


class RequestResult(BaseModel):
    """Timings for one request. All times in seconds."""

    index: int
    start: float = Field(description="Send time, relative to the start of the level")
    ok: bool = True
    error: str | None = None
    ttft: float | None = Field(None, description="Send -> first chunk with non-empty content")
    e2e: float | None = Field(None, description="Send -> end of stream")
    chunk_times: list[float] = Field(
        default_factory=list, description="Arrival time of each content chunk, relative to send"
    )
    prompt_tokens: int | None = None
    output_tokens: int = 0
    token_source: TokenSource = "chunks"

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_times)

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first: (e2e - ttft) / (output_tokens - 1)."""
        if not self.ok or self.ttft is None or self.e2e is None or self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)

    @property
    def itls(self) -> list[float]:
        """Gaps between consecutive content chunks (one chunk may hold several tokens)."""
        t = self.chunk_times
        return [b - a for a, b in zip(t, t[1:], strict=False)]


class Stats(BaseModel):
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    max: float | None = None


class Summary(BaseModel):
    completed: int
    failed: int
    errors: dict[str, int] = Field(default_factory=dict)
    duration: float
    request_throughput: float
    output_throughput: float
    input_throughput: float | None = None
    goodput: float
    slo_attainment: float | None = Field(
        None, description="Share of all requests (failed ones included) that met the SLOs")
    slo_ttft_ms: float | None
    slo_tpot_ms: float | None
    ttft: Stats
    tpot: Stats
    itl: Stats
    e2e: Stats
    output_tokens_mean: float | None = None
    tokens_per_chunk: float | None = None
    token_sources: dict[str, int] = Field(default_factory=dict)


class LevelResult(BaseModel):
    # JSON has no inf; write it as the string "Infinity" so run.json stays valid JSON
    model_config = ConfigDict(ser_json_inf_nan="strings")

    mode: Literal["concurrency", "rate"]
    value: float = Field(description="Concurrency N, or request rate in req/s (inf = all at once)")
    duration: float
    client_lag_p99_ms: float | None = Field(
        None, description="p99 event-loop scheduling delay in the client; high means the "
        "client, not the server, may be the bottleneck"
    )
    requests: list[RequestResult]
    summary: Summary

    @property
    def label(self) -> str:
        if self.mode == "concurrency":
            return f"c={int(self.value)}"
        return "rate=inf" if self.value == float("inf") else f"rate={self.value:g}"


class RunConfig(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="strings")

    url: str
    model: str
    endpoint: Literal["chat", "completions"] = "chat"
    prompts: str = "synthetic"
    input_len: int = 256
    output_len: int = 128
    mode: Literal["concurrency", "rate"] = "concurrency"
    levels: list[float] = Field(default_factory=lambda: [1.0])
    max_concurrency: int | None = None
    num_requests: int | None = None
    warmup: int = 2
    seed: int = 0
    ignore_eos: bool = True
    temperature: float = 0.0
    timeout: float = 300.0
    slo_ttft_ms: float | None = 500.0
    slo_tpot_ms: float | None = 50.0
    tokenizer: str | None = None


class Environment(BaseModel):
    servebench_version: str
    python: str
    platform: str
    client_cpu: str | None = None
    timestamp: str
    server_models: list[str] | None = None
    notes: str | None = None


class RunResult(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="strings")

    label: str
    config: RunConfig
    environment: Environment
    levels: list[LevelResult]
    capacity: CapacityResult | None = Field(None, description="Set by `find-capacity`")


class CapacityResult(BaseModel):
    """Highest load that met the SLOs, found by `servebench find-capacity`."""

    model_config = ConfigDict(ser_json_inf_nan="strings")

    mode: Literal["concurrency", "rate"]
    target: float = Field(description="Share of requests that must meet the SLOs, e.g. 0.9")
    slo_ttft_ms: float | None
    slo_tpot_ms: float | None
    capacity: float | None = Field(None, description="Highest passing load; None if none passed")
    first_fail: float | None = Field(None, description="Lowest failing load; None if none failed")
    reason: Literal["converged", "max_reached", "start_failed"]


RunResult.model_rebuild()  # resolve the forward reference to CapacityResult
