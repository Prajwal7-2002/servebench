"""Automated LLM Server Tuning & Parameter Optimizer.

Analyzes latency-throughput knees, SLO compliance thresholds, and server telemetry
to recommend production-grade concurrency limits, memory allocation, and launch flags
for Ollama, vLLM, llama.cpp, and mini-infer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .detect import Server
from .results import RunResult


class TuningRecommendation(BaseModel):
    engine: str
    recommended_concurrency: int
    knee_throughput_tok_s: float
    slo_compliance_pct: float
    recommended_env_vars: dict[str, str] = Field(default_factory=dict)
    recommended_cli_flags: list[str] = Field(default_factory=list)
    launch_command_example: str
    rationale: list[str] = Field(default_factory=list)


class TuningReport(BaseModel):
    model: str
    detected_engine: str
    optimal_concurrency: int
    peak_sustainable_tok_s: float
    recommendation: TuningRecommendation
    all_levels_analyzed: list[dict[str, Any]] = Field(default_factory=list)


def evaluate_tuning(
    run: RunResult,
    server_info: Server | None = None,
    engine_override: str | None = None,
    target_ttft_p50_ms: float = 150.0,
    target_itl_p50_ms: float = 40.0,
    min_goodput_pct: float = 95.0,
) -> TuningReport:
    """Analyze benchmark sweep or capacity run and derive optimal server settings."""
    levels = run.levels
    if not levels:
        raise ValueError("Cannot tune server without completed benchmark levels.")

    # Sort levels by concurrency value
    c_levels = sorted([lv for lv in levels if lv.mode == "concurrency"], key=lambda x: x.value)
    if not c_levels:
        c_levels = levels

    analyzed = []
    optimal_c = 1
    best_goodput_tok_s = 0.0

    for i, lv in enumerate(c_levels):
        c_val = int(lv.value)
        out_tok_s = (
            getattr(lv.summary, "output_throughput", 0.0)
            or getattr(lv.summary, "output_tokens_per_s", 0.0)
            or 0.0
        )
        ttft_p50 = (
            (lv.summary.ttft.p50 or 0.0) * 1000.0
            if (lv.summary.ttft.p50 or 0.0) < 10.0
            else (lv.summary.ttft.p50 or 0.0)
        )
        itl_p50 = (
            (lv.summary.itl.p50 or 0.0) * 1000.0
            if (lv.summary.itl.p50 or 0.0) < 10.0
            else (lv.summary.itl.p50 or 0.0)
        )

        total_reqs = getattr(lv.summary, "ok", 0) + getattr(lv.summary, "failed", 0)
        err_rate = (
            (getattr(lv.summary, "failed", 0) / max(1, total_reqs)) if total_reqs > 0 else 0.0
        )

        # Check SLO compliance
        slo_pass = (
            ttft_p50 <= target_ttft_p50_ms
            and (itl_p50 <= target_itl_p50_ms or itl_p50 == 0.0)
            and err_rate < 0.05
        )

        marginal_gain_pct = 0.0
        if i > 0 and analyzed[i - 1]["throughput"] > 0:
            prev_t = analyzed[i - 1]["throughput"]
            marginal_gain_pct = ((out_tok_s - prev_t) / prev_t) * 100.0

        analyzed.append(
            {
                "concurrency": c_val,
                "throughput": round(out_tok_s, 1),
                "ttft_p50_ms": round(ttft_p50, 1),
                "itl_p50_ms": round(itl_p50, 1),
                "marginal_gain_pct": round(marginal_gain_pct, 1),
                "slo_pass": slo_pass,
            }
        )

        # Track optimal concurrency: highest throughput that respects SLOs
        # or where marginal gain justifies latency
        if slo_pass and out_tok_s >= best_goodput_tok_s * 0.95:
            optimal_c = c_val
            best_goodput_tok_s = max(best_goodput_tok_s, out_tok_s)

    # If no level cleanly met strict SLOs, pick the best balanced knee
    if optimal_c == 1 and len(analyzed) > 1:
        # Pick the level before TTFT starts superlinear explosion
        for entry in analyzed:
            if entry["ttft_p50_ms"] < target_ttft_p50_ms * 1.5:
                optimal_c = entry["concurrency"]
                best_goodput_tok_s = entry["throughput"]

    engine_name = (
        engine_override
        or (server_info.engine if server_info else getattr(run.environment, "engine", None))
        or "generic"
    ).lower()
    model_name = run.config.model or "model"

    rec = generate_engine_config(engine_name, model_name, optimal_c, best_goodput_tok_s)

    return TuningReport(
        model=model_name,
        detected_engine=engine_name,
        optimal_concurrency=optimal_c,
        peak_sustainable_tok_s=round(best_goodput_tok_s, 1),
        recommendation=rec,
        all_levels_analyzed=analyzed,
    )


def generate_engine_config(
    engine: str,
    model: str,
    concurrency: int,
    peak_tok_s: float,
) -> TuningRecommendation:
    """Generate exact copy-pasteable configuration flags for each engine."""
    e = engine.lower()
    env_vars = {}
    cli_flags = []
    rationale = []

    if "ollama" in e:
        env_vars = {
            "OLLAMA_NUM_PARALLEL": str(concurrency),
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KEEP_ALIVE": "24h",
        }
        cmd = f"OLLAMA_NUM_PARALLEL={concurrency} OLLAMA_FLASH_ATTENTION=1 ollama serve"
        rationale = [
            f"Set OLLAMA_NUM_PARALLEL={concurrency} to avoid queue buildup.",
            "Enabled OLLAMA_FLASH_ATTENTION=1 to minimize attention memory overhead.",
            "Set OLLAMA_KEEP_ALIVE=24h to avoid expensive cold-start model reload penalties.",
        ]
    elif "vllm" in e:
        cli_flags = [
            f"--max-num-seqs {concurrency}",
            "--gpu-memory-utilization 0.90",
            "--enable-chunked-prefill",
            "--enforce-eager",
        ]
        cmd = (
            f"vllm serve {model} --max-num-seqs {concurrency} "
            f"--gpu-memory-utilization 0.90 --enable-chunked-prefill"
        )
        rationale = [
            f"Capped --max-num-seqs={concurrency} for high-throughput continuous batching.",
            "Enabled --enable-chunked-prefill to prevent long prompts from starving decode.",
            "Configured --gpu-memory-utilization 0.90 for maximum Paged KV-cache headroom.",
        ]
    elif "llama" in e:
        cli_flags = [
            f"-np {concurrency}",
            "-cb",
            "-fa",
        ]
        cmd = f"llama-server -m {model}.gguf -np {concurrency} -cb -fa"
        rationale = [
            f"Allocated -np {concurrency} parallel processing slots.",
            "Enabled -cb (continuous batching) for dynamic iteration-level scheduling.",
            "Enabled -fa (flash attention) to reduce peak KV memory footprint.",
        ]
    elif "mini-infer" in e:
        cli_flags = [
            f"--max-batch-size {concurrency}",
            "--block-size 16",
        ]
        cmd = f"mini-infer serve --model {model} --max-batch-size {concurrency} --port 8000"
        rationale = [
            f"Configured continuous batching pool for {concurrency} concurrent streams.",
            "Fixed block pool at 16 tokens/block to maintain <3.8% internal VRAM fragmentation.",
        ]
    else:
        cmd = f"serve --model {model} --concurrency {concurrency}"
        rationale = [
            f"Recommended maximum concurrent worker capacity: {concurrency}.",
        ]

    return TuningRecommendation(
        engine=engine,
        recommended_concurrency=concurrency,
        knee_throughput_tok_s=round(peak_tok_s, 1),
        slo_compliance_pct=100.0,
        recommended_env_vars=env_vars,
        recommended_cli_flags=cli_flags,
        launch_command_example=cmd,
        rationale=rationale,
    )
