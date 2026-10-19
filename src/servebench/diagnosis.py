"""Automated Bottleneck Diagnosis Engine.

Analyzes benchmark metrics across load levels, GPU telemetry, and server timings
to identify whether the serving system is:
1. Slot/Queue-Bound (KV cache exhaustion or concurrent slot limits)
2. Memory-Bandwidth Bound (Decode phase memory bus saturation)
3. Compute-Bound (Prefill phase tensor/CUDA core saturation)
4. Client/Event-Loop Bound (Benchmarking client CPU/event-loop lag)
5. Balanced / Healthy (Linear scaling within configured SLOs)

Provides human-readable diagnostic summaries and actionable tuning recommendations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .results import RunResult


class DiagnosisFinding(BaseModel):
    category: str = Field(
        description="Bottleneck category (Queue, Memory, Compute, Client, Healthy)"
    )
    severity: str = Field(description="Severity: info, low, medium, high")
    headline: str = Field(description="Crisp one-sentence summary of the bottleneck")
    evidence: list[str] = Field(
        default_factory=list, description="Data points supporting the finding"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations to resolve or tune"
    )


class BottleneckReport(BaseModel):
    primary_bottleneck: str
    overall_health: str  # excellent, good, degraded, bottlenecked
    findings: list[DiagnosisFinding] = Field(default_factory=list)


def diagnose_run(run: RunResult) -> BottleneckReport:
    """Analyze a completed RunResult and generate a diagnostic report."""
    levels = run.levels
    if not levels:
        return BottleneckReport(
            primary_bottleneck="Insufficient Data",
            overall_health="unknown",
            findings=[
                DiagnosisFinding(
                    category="Info",
                    severity="info",
                    headline="No benchmark levels to diagnose.",
                    evidence=[],
                    recommendations=[],
                )
            ],
        )

    findings: list[DiagnosisFinding] = []

    # 1. Check Client-side Event Loop Lag
    max_lag = max((lv.client_lag_p99_ms or 0.0) for lv in levels)
    if max_lag > 50.0:
        findings.append(
            DiagnosisFinding(
                category="Client",
                severity="high" if max_lag > 100.0 else "medium",
                headline="Benchmark client event loop is saturated.",
                evidence=[
                    f"Client lag p99 reached {max_lag:.1f} ms (>50 ms threshold).",
                    "Asyncio loop delayed scheduling callbacks, adding artificial latency.",
                ],
                recommendations=[
                    "Run servebench on a machine with more idle CPU cores.",
                    "Reduce concurrency or rate to keep client CPU utilization below 85%.",
                ],
            )
        )

    # 2. Check Concurrency Scaling & Queue / Slot Bottlenecks
    c_levels = [lv for lv in levels if lv.mode == "concurrency"]
    if len(c_levels) >= 2:
        c_levels_sorted = sorted(c_levels, key=lambda x: x.value)
        base = c_levels_sorted[0]
        peak = c_levels_sorted[-1]

        base_ttft = base.summary.ttft.p50 or 0.0
        peak_ttft = peak.summary.ttft.p50 or 0.0
        base_itl = base.summary.itl.p50 or 0.0
        peak_itl = peak.summary.itl.p50 or 0.0

        ttft_ratio = (peak_ttft / base_ttft) if base_ttft > 0 else 1.0
        itl_ratio = (peak_itl / base_itl) if base_itl > 0 else 1.0

        # Queue-bound detection: TTFT spikes while ITL remains relatively flat
        if ttft_ratio >= 2.2 and itl_ratio < 1.4:
            c_val = int(peak.value)
            b_val = int(base.value)
            findings.append(
                DiagnosisFinding(
                    category="Queue/Slot",
                    severity="high" if ttft_ratio > 3.5 else "medium",
                    headline="Server is Slot/Queue-Bound: requests are waiting for KV cache slots.",
                    evidence=[
                        f"TTFT p50 grew {ttft_ratio:.1f}x ({base_ttft*1000:.0f} ms at c={b_val} "
                        f"to {peak_ttft*1000:.0f} ms at c={c_val}).",
                        f"Token decode latency (ITL p50) remained steady "
                        f"({itl_ratio:.1f}x: {base_itl*1000:.1f} ms -> {peak_itl*1000:.1f} ms).",
                        "Signature indicates requests queue waiting for active slots to finish.",
                    ],
                    recommendations=[
                        "Increase server max concurrent sequences / slots (e.g. `--max-num-seqs`).",
                        "Increase KV-cache memory allocation (e.g. `--gpu-memory-utilization`).",
                        "Enable chunked prefill to interleave prefill with decode iterations.",
                    ],
                )
            )

        # Memory-Bandwidth Bound detection: ITL degrades notably with batch size
        gpu_telemetry = peak.gpu_telemetry or run.gpu_telemetry
        avg_gpu_util = gpu_telemetry.avg_gpu_util if gpu_telemetry else None

        if itl_ratio >= 1.6 and (avg_gpu_util is None or avg_gpu_util < 75.0):
            findings.append(
                DiagnosisFinding(
                    category="Memory-Bandwidth",
                    severity="medium",
                    headline="Server is Memory-Bandwidth Bound during token generation.",
                    evidence=[
                        f"ITL p50 degraded by {itl_ratio:.1f}x under load "
                        f"({base_itl*1000:.1f} ms -> {peak_itl*1000:.1f} ms).",
                        f"GPU compute utilization was moderate ({avg_gpu_util:.0f}% avg)"
                        if avg_gpu_util is not None
                        else "Decode latency scaled with concurrent streams.",
                        "Token generation is reading weights and KV cache from VRAM on every step.",
                    ],
                    recommendations=[
                        "Quantize model weights (AWQ, GPTQ, INT4, FP8) to reduce bytes loaded.",
                        "Enable FP8 KV-cache quantization to halve memory traffic.",
                        "Use speculative decoding to generate multiple tokens per memory pass.",
                    ],
                )
            )

    # 3. Check Server Prefill vs Queue Overhead (P11 Server Timings)
    all_reqs = [r for lv in levels for r in lv.requests if r.ok]
    reqs_with_overheads = [
        r for r in all_reqs if r.queue_network_overhead_ms is not None and r.ttft
    ]
    if reqs_with_overheads:
        avg_overhead = sum(
            r.queue_network_overhead_ms for r in reqs_with_overheads
        ) / len(reqs_with_overheads)
        avg_ttft_ms = sum(
            r.ttft * 1000.0 for r in reqs_with_overheads
        ) / len(reqs_with_overheads)
        overhead_pct = (avg_overhead / avg_ttft_ms * 100.0) if avg_ttft_ms > 0 else 0.0

        if overhead_pct > 40.0 and avg_overhead > 50.0:
            findings.append(
                DiagnosisFinding(
                    category="Queue/Network",
                    severity="medium",
                    headline="Server queueing delay accounts for a major portion of TTFT.",
                    evidence=[
                        f"Average queue/net overhead is {avg_overhead:.1f} ms "
                        f"({overhead_pct:.0f}% of total TTFT).",
                        f"Server prefill compute only took {avg_ttft_ms - avg_overhead:.1f} ms.",
                    ],
                    recommendations=[
                        "Server is queuing requests prior to initiating prompt prefill.",
                        "Enable continuous batching or reduce concurrency to minimize wait queues.",
                    ],
                )
            )

    # 4. Check Prometheus Server Metrics if present
    for lv in levels:
        if lv.prometheus_metrics:
            waiting = lv.prometheus_metrics.get("num_requests_waiting", 0.0)
            cache_usage = lv.prometheus_metrics.get("gpu_cache_usage", 0.0)
            if waiting > 0:
                findings.append(
                    DiagnosisFinding(
                        category="Queue/Prometheus",
                        severity="high",
                        headline=f"Server Prometheus reports request backlog at {lv.label}.",
                        evidence=[
                            f"Queue depth: {waiting:.0f} requests waiting for GPU allocation.",
                            f"KV cache utilization: {cache_usage * 100:.1f}%.",
                        ],
                        recommendations=[
                            "Increase available KV cache or deploy additional serving replicas.",
                        ],
                    )
                )
                break

    # Determine overall primary bottleneck and health
    if not findings:
        primary = "Balanced"
        health = "excellent"
        findings.append(
            DiagnosisFinding(
                category="Balanced",
                severity="info",
                headline="System is performing nominally within configured load parameters.",
                evidence=[
                    "No excessive queueing delays, event-loop lag, or memory collapses detected.",
                    "Latency and throughput scale predictably across tested levels.",
                ],
                recommendations=[
                    "System capacity is healthy for current traffic levels.",
                ],
            )
        )
    else:
        severity_order = {"high": 3, "medium": 2, "low": 1, "info": 0}
        findings.sort(key=lambda f: severity_order.get(f.severity, 0), reverse=True)
        primary = findings[0].category
        health = "bottlenecked" if findings[0].severity == "high" else "degraded"

    return BottleneckReport(
        primary_bottleneck=primary,
        overall_health=health,
        findings=findings,
    )
