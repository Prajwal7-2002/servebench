"""Data models for configs and results. Everything in run.json goes through these."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .telemetry import GpuTelemetrySummary

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
    reasoning_chunks: int = Field(
        0, description="Content chunks that were reasoning ('thinking') tokens, not answer"
    )
    answer_start: float | None = Field(
        None,
        description="Send -> first answer (non-reasoning) chunk; equals TTFT for "
        "models that don't stream reasoning",
    )
    server_prefill_ms: float | None = Field(
        None, description="Time server spent on prompt prefill evaluation in ms"
    )
    server_decode_ms: float | None = Field(
        None, description="Time server spent on token generation/decode in ms"
    )
    queue_network_overhead_ms: float | None = Field(
        None, description="Client TTFT - Server Prefill: request queueing delay + network RTT in ms"
    )

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
    count: int = 0

    @property
    def p99_is_approx(self) -> bool:
        """True if sample count is under 100, meaning p99 approximates the sample maximum."""
        return 0 < self.count < 100


class RepeatMetric(BaseModel):
    """Statistical summary across multiple repeated trials of a level."""

    mean: float
    std: float
    min: float
    max: float
    ci95: tuple[float, float] | None = Field(
        None, description="95% two-tailed confidence interval [lower, upper]"
    )


class Summary(BaseModel):
    completed: int
    failed: int
    errors: dict[str, int] = Field(default_factory=dict)
    duration: float
    request_throughput: float
    output_throughput: float
    input_throughput: float | None = None
    steady_state_request_throughput: float | None = Field(
        None, description="Request throughput excluding ramp-up"
    )
    steady_state_output_throughput: float | None = Field(
        None, description="Output token throughput excluding ramp-up"
    )
    goodput: float
    slo_attainment: float | None = Field(
        None, description="Share of all requests (failed ones included) that met the SLOs"
    )
    slo_ttft_ms: float | None
    slo_tpot_ms: float | None
    ttft: Stats
    tpot: Stats
    itl: Stats
    e2e: Stats
    output_tokens_mean: float | None = None
    tokens_per_chunk: float | None = None
    token_sources: dict[str, int] = Field(default_factory=dict)
    reasoning_share: float | None = Field(
        None, description="Share of content chunks that were reasoning tokens"
    )
    answer_start: Stats = Field(
        default_factory=Stats, description="Send -> first answer token (after reasoning)"
    )
    server_prefill_ms: Stats = Field(
        default_factory=Stats, description="Server prompt evaluation/prefill time in ms"
    )
    server_decode_ms: Stats = Field(
        default_factory=Stats, description="Server token generation/decode time in ms"
    )
    queue_network_overhead_ms: Stats = Field(
        default_factory=Stats,
        description="Client TTFT minus server prefill in ms (queue delay + network RTT)",
    )
    gpu_telemetry: GpuTelemetrySummary | None = Field(
        None, description="GPU metrics collected during this run/level"
    )


class LevelResult(BaseModel):
    # JSON has no inf; write it as the string "Infinity" so run.json stays valid JSON
    model_config = ConfigDict(ser_json_inf_nan="strings")

    mode: Literal["concurrency", "rate"]
    value: float = Field(description="Concurrency N, or request rate in req/s (inf = all at once)")
    duration: float
    client_lag_p99_ms: float | None = Field(
        None,
        description="p99 event-loop scheduling delay in the client; high means the "
        "client, not the server, may be the bottleneck",
    )
    requests: list[RequestResult]
    summary: Summary
    repeat_count: int = Field(1, description="Number of repeat trials conducted for this level")
    repeats: list[Summary] | None = Field(
        None, description="Summary from each repeat trial when repeat > 1"
    )
    repeat_metrics: dict[str, RepeatMetric] | None = Field(
        None, description="Mean, std, min, max, ci95 across repeats"
    )
    gpu_telemetry: GpuTelemetrySummary | None = Field(
        None, description="GPU metrics collected during this level"
    )
    prometheus_metrics: dict[str, float] | None = Field(
        None, description="Server Prometheus /metrics scraped before/after level"
    )
    bottlenecks: list[str] | None = Field(
        None, description="Bottleneck diagnosis findings for this level"
    )

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
    duration: float | None = Field(None, description="Seconds per level, for timed runs")
    min_duration: float | None = Field(
        None, description="Minimum duration in seconds per level for steady state"
    )
    repeat: int = Field(1, description="Number of times each level is repeated")
    warmup: int = 2
    seed: int = 0
    ignore_eos: bool = True
    temperature: float = 0.0
    timeout: float = 300.0
    slo_ttft_ms: float | None = 500.0
    slo_tpot_ms: float | None = 50.0
    tokenizer: str | None = None
    preset: str | None = Field(None, description="Workload preset (chat, rag, code, agent)")
    input_dist: str | None = Field(
        None, description="Input length distribution (fixed, normal, lognormal, uniform)"
    )
    output_dist: str | None = Field(
        None, description="Output length distribution (fixed, normal, lognormal, uniform)"
    )
    arrival_dist: str = Field("poisson", description="Arrival distribution (poisson, gamma)")
    burstiness: float = Field(1.0, description="Gamma shape parameter for arrival bursts")
    shared_prefix_tokens: int | None = Field(
        None, description="Shared prompt prefix length for prefix cache testing"
    )
    trace: str | None = Field(None, description="Path to JSONL requests trace to replay")
    gpu_telemetry: bool = Field(True, description="Capture GPU telemetry via NVML")
    scrape_metrics: str | None = Field(
        None, description="Prometheus /metrics endpoint URL or 'auto'"
    )


class Environment(BaseModel):
    servebench_version: str
    python: str
    platform: str
    client_cpu: str | None = None
    timestamp: str
    server_models: list[str] | None = None
    notes: str | None = None
    # machine running servebench (for a local server, the server's machine too)
    cpu: str | None = None
    ram_gb: float | None = None
    gpus: list[str] | None = None
    local_server: bool | None = Field(None, description="Server URL is this machine")
    # what answered at the URL, and what it said about the model after the run
    engine: str | None = None
    engine_version: str | None = None
    model_info: dict | None = None


SCHEMA_VERSION = 1
"""Bump when run.json changes in a way older servebench versions cannot read. Adding an
optional field with a default does not need a bump; renaming or changing meaning does."""


class RunResult(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="strings")

    # files written before this field existed have no version: they are version 1
    schema_version: int = SCHEMA_VERSION
    label: str
    config: RunConfig
    environment: Environment
    levels: list[LevelResult]
    capacity: CapacityResult | None = Field(None, description="Set by `find-capacity`")
    gpu_telemetry: GpuTelemetrySummary | None = Field(
        None, description="Overall GPU telemetry summary for the sweep"
    )
    bottleneck_diagnosis: dict[str, Any] | None = Field(
        None, description="Comprehensive bottleneck diagnosis and recommendations"
    )


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
