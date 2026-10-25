"""Hardware Roofline Model, MFU, and MBU Analysis Engine.

Computes theoretical and empirical efficiency metrics for LLM serving:
1. MBU (Memory Bandwidth Utilization %):
   Evaluates decode-phase throughput against GPU memory bandwidth limits.
2. MFU (Model FLOPs Utilization %):
   Evaluates prefill-phase throughput against GPU peak Tensor FLOPs.
3. ASCII Roofline Chart:
   Visualizes arithmetic intensity (FLOPs/byte) vs performance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class GPUSpec:
    name: str
    memory_bandwidth_gb_s: float
    peak_fp16_tflops: float
    vram_gb: float


# Common GPU reference specs (Memory Bandwidth in GB/s, Peak FP16/BF16 Tensor TFLOPs)
KNOWN_GPUS: dict[str, GPUSpec] = {
    "rtx 3050": GPUSpec("RTX 3050 Laptop", 192.0, 36.4, 6.0),
    "rtx 3060": GPUSpec("RTX 3060", 360.0, 51.2, 12.0),
    "rtx 3070": GPUSpec("RTX 3070", 448.0, 81.3, 8.0),
    "rtx 3080": GPUSpec("RTX 3080", 760.0, 119.0, 10.0),
    "rtx 3090": GPUSpec("RTX 3090", 936.0, 142.0, 24.0),
    "rtx 4050": GPUSpec("RTX 4050 Laptop", 192.0, 90.0, 6.0),
    "rtx 4060": GPUSpec("RTX 4060", 272.0, 121.0, 8.0),
    "rtx 4070 ti": GPUSpec("RTX 4070 Ti", 504.0, 240.0, 12.0),
    "rtx 4070": GPUSpec("RTX 4070", 504.0, 184.0, 12.0),
    "rtx 4080 super": GPUSpec("RTX 4080 Super", 736.0, 260.0, 16.0),
    "rtx 4080": GPUSpec("RTX 4080", 717.0, 242.0, 16.0),
    "rtx 4090": GPUSpec("RTX 4090", 1008.0, 330.0, 24.0),
    "t4": GPUSpec("NVIDIA T4", 320.0, 65.0, 16.0),
    "v100": GPUSpec("NVIDIA V100", 900.0, 125.0, 16.0),
    "a10": GPUSpec("NVIDIA A10", 600.0, 125.0, 24.0),
    "a100": GPUSpec("NVIDIA A100-SXM4-80GB", 2039.0, 312.0, 80.0),
    "h100": GPUSpec("NVIDIA H100-SXM", 3350.0, 989.0, 80.0),
    "h200": GPUSpec("NVIDIA H200-SXM", 4800.0, 989.0, 141.0),
    "b200": GPUSpec("NVIDIA B200", 8000.0, 2250.0, 192.0),
    "l40s": GPUSpec("NVIDIA L40S", 864.0, 366.0, 48.0),
    "l4": GPUSpec("NVIDIA L4", 300.0, 120.0, 24.0),
    "a6000": GPUSpec("RTX A6000", 768.0, 154.0, 48.0),
    "apple m1 pro": GPUSpec("Apple M1 Pro", 200.0, 10.4, 32.0),
    "apple m1 max": GPUSpec("Apple M1 Max", 400.0, 20.8, 64.0),
    "apple m1": GPUSpec("Apple M1", 68.25, 5.2, 16.0),
    "apple m2 pro": GPUSpec("Apple M2 Pro", 200.0, 13.6, 32.0),
    "apple m2 max": GPUSpec("Apple M2 Max", 400.0, 27.2, 96.0),
    "apple m2": GPUSpec("Apple M2", 100.0, 7.1, 24.0),
    "apple m3 pro": GPUSpec("Apple M3 Pro", 150.0, 14.5, 36.0),
    "apple m3 max": GPUSpec("Apple M3 Max", 400.0, 35.0, 128.0),
    "apple m3": GPUSpec("Apple M3", 100.0, 8.5, 24.0),
    "apple m4 pro": GPUSpec("Apple M4 Pro", 273.0, 23.0, 48.0),
    "apple m4 max": GPUSpec("Apple M4 Max", 546.0, 46.0, 128.0),
    "apple m4": GPUSpec("Apple M4", 120.0, 11.5, 32.0),
}


class RooflinePoint(BaseModel):
    concurrency: int
    arithmetic_intensity: float = Field(description="FLOPs per byte")
    achieved_tflops: float
    achieved_tok_s: float
    mbu_percent: float = Field(description="Memory Bandwidth Utilization %")
    mfu_percent: float = Field(description="Model FLOPs Utilization %")
    regime: str = Field(description="Memory-Bandwidth Bound or Compute Bound")


class RooflineAnalysis(BaseModel):
    gpu_name: str
    memory_bandwidth_gb_s: float
    peak_tflops: float
    model_name: str
    param_count_billions: float
    bytes_per_param: float
    model_size_gb: float
    ridge_point: float = Field(description="FLOPs/byte at boundary of memory and compute bounds")
    points: list[RooflinePoint] = Field(default_factory=list)
    peak_mbu: float = 0.0
    peak_mfu: float = 0.0
    primary_regime: str = "Memory-Bandwidth Bound"


def estimate_model_params(model_name: str) -> tuple[float, float]:
    """Estimate (param_count_in_billions, bytes_per_param) from model name.

    Examples:
        'qwen2.5:0.5b' -> (0.49, 2.0)
        'llama3:8b-q4_k_m' -> (8.03, 0.55)
        'mistral:7b-instruct-q8_0' -> (7.24, 1.0)
    """
    m = model_name.lower()

    # Estimate param count
    params = 1.5  # default fallback
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", m)
    if match:
        params = float(match.group(1))
        if params == 0.5:
            params = 0.49
        elif params == 1.5:
            params = 1.54
        elif params == 3.0:
            params = 3.09
        elif params == 7.0:
            params = 7.24
        elif params == 8.0:
            params = 8.03
        elif params == 14.0:
            params = 14.7
        elif params == 32.0:
            params = 32.5
        elif params == 70.0:
            params = 70.6

    # Estimate precision bytes per param
    bytes_per_param = 2.0  # default FP16 / BF16
    if any(q in m for q in ["1.58b", "1.58-bit", "ternary", "bitnet"]):
        bytes_per_param = 0.28  # Sub-2-bit BitNet packed representation
    elif any(q in m for q in ["q2", "int2", "2bit", "2-bit"]):
        bytes_per_param = 0.35
    elif any(q in m for q in ["q4", "int4", "4bit", "4-bit"]):
        bytes_per_param = 0.55  # 4-bit with block quantization overhead
    elif any(q in m for q in ["q5", "5bit", "5-bit"]):
        bytes_per_param = 0.68
    elif any(q in m for q in ["q8", "int8", "8bit", "8-bit"]):
        bytes_per_param = 1.05
    elif any(q in m for q in ["fp8", "e4m3", "e5m2"]):
        bytes_per_param = 1.0
    elif any(q in m for q in ["fp16", "bf16"]):
        bytes_per_param = 2.0
    elif any(q in m for q in ["fp32"]):
        bytes_per_param = 4.0

    return params, bytes_per_param


def resolve_gpu_spec(device_name: str | None = None) -> GPUSpec:
    """Match GPU string to known specs or return a safe default."""
    if not device_name:
        return GPUSpec("Generic GPU", 200.0, 50.0, 8.0)

    d = device_name.lower()
    for key in sorted(KNOWN_GPUS.keys(), key=len, reverse=True):
        if key in d:
            return KNOWN_GPUS[key]

    # Generic heuristics
    if "apple" in d:
        return GPUSpec(device_name, 150.0, 10.0, 16.0)
    if "nvidia" in d or "geforce" in d or "rtx" in d:
        return GPUSpec(device_name, 250.0, 50.0, 8.0)

    return GPUSpec(device_name, 200.0, 40.0, 8.0)


def compute_roofline(
    model_name: str,
    levels_data: list[dict[str, Any]],
    gpu_name: str | None = None,
    custom_bandwidth_gb_s: float | None = None,
    custom_peak_tflops: float | None = None,
) -> RooflineAnalysis:
    """Compute Roofline metrics across benchmark levels.

    levels_data items are expected to have:
    - concurrency or level (int)
    - output_tokens_per_s (float)
    - prompt_tokens_per_s (float, optional)
    """
    gpu = resolve_gpu_spec(gpu_name)
    bandwidth = custom_bandwidth_gb_s or gpu.memory_bandwidth_gb_s
    peak_tflops = custom_peak_tflops or gpu.peak_fp16_tflops

    params, bytes_per_param = estimate_model_params(model_name)
    model_size_gb = (params * 1e9 * bytes_per_param) / 1e9

    # Ridge point: arithmetic intensity where memory-bound transitions to compute-bound
    # (Peak FLOP/s) / (Bandwidth Bytes/s)
    ridge_point = (peak_tflops * 1e12) / (bandwidth * 1e9)

    points: list[RooflinePoint] = []
    peak_mbu = 0.0
    peak_mfu = 0.0

    for lv in levels_data:
        c = int(lv.get("concurrency") or lv.get("value") or 1)
        out_tok_s = float(lv.get("output_tokens_per_s") or lv.get("throughput") or 0.0)
        in_tok_s = float(lv.get("prompt_tokens_per_s") or (out_tok_s * 0.2))

        # Decode phase calculations:
        # Each generated token at batch size 1 reads model weights once.
        # Across batch size c, weight reads are amortized over c concurrent tokens.
        # Effective memory bandwidth consumed:
        # Bandwidth (GB/s) = (out_tok_s / c) * model_size_gb
        # MBU = (Bandwidth consumed / Peak GPU Bandwidth) * 100
        bytes_transferred_per_token_batch = (model_size_gb * 1e9) / max(1, c)
        achieved_bandwidth_gb_s = (out_tok_s * bytes_transferred_per_token_batch) / 1e9
        mbu = (achieved_bandwidth_gb_s / bandwidth) * 100.0 if bandwidth > 0 else 0.0
        mbu = min(mbu, 100.0)

        # Prefill / Compute FLOPs calculation:
        # Forward pass FLOPs ≈ 2 * params per token
        flops_per_token = 2.0 * params * 1e9
        achieved_flops = (out_tok_s + in_tok_s) * flops_per_token
        achieved_tflops = achieved_flops / 1e12

        mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0.0
        mfu = min(mfu, 100.0)

        # Arithmetic intensity = FLOPs / byte transferred
        ai = (2.0 * params * 1e9) / max(1.0, bytes_transferred_per_token_batch)

        regime = "Memory-Bandwidth Bound" if ai < ridge_point else "Compute Bound"

        points.append(
            RooflinePoint(
                concurrency=c,
                arithmetic_intensity=round(ai, 2),
                achieved_tflops=round(achieved_tflops, 3),
                achieved_tok_s=round(out_tok_s, 1),
                mbu_percent=round(mbu, 1),
                mfu_percent=round(mfu, 1),
                regime=regime,
            )
        )

        if mbu > peak_mbu:
            peak_mbu = mbu
        if mfu > peak_mfu:
            peak_mfu = mfu

    primary_regime = "Memory-Bandwidth Bound"
    if points and sum(1 for p in points if p.regime == "Compute Bound") > len(points) / 2:
        primary_regime = "Compute Bound"

    return RooflineAnalysis(
        gpu_name=gpu.name,
        memory_bandwidth_gb_s=bandwidth,
        peak_tflops=peak_tflops,
        model_name=model_name,
        param_count_billions=params,
        bytes_per_param=bytes_per_param,
        model_size_gb=round(model_size_gb, 2),
        ridge_point=round(ridge_point, 1),
        points=points,
        peak_mbu=round(peak_mbu, 1),
        peak_mfu=round(peak_mfu, 1),
        primary_regime=primary_regime,
    )


def render_ascii_roofline(analysis: RooflineAnalysis, width: int = 60, height: int = 14) -> str:
    """Render a clean, responsive ASCII Roofline chart.

    Plots:
    - Vertical axis: Performance (TFLOPs) log scale
    - Horizontal axis: Arithmetic Intensity (FLOPs/byte) log scale
    - Slanted line: Memory Bandwidth Ceiling (/)
    - Horizontal line: Peak Compute Ceiling (-)
    - Asterisks (*): Operating points
    """
    lines = []
    bar = "-" * (width - 2)
    lines.append(f"+{bar}+")
    lines.append(f"| Roofline Model: {analysis.gpu_name} ({analysis.model_name})")
    lines.append(
        f"| Peak Bandwidth: {analysis.memory_bandwidth_gb_s:.0f} GB/s | "
        f"Peak Compute: {analysis.peak_tflops:.1f} TFLOPs | "
        f"Ridge: {analysis.ridge_point:.1f} FLOP/B"
    )
    lines.append(f"+{bar}+")

    # Grid canvas
    plot_w = width - 12
    plot_h = height

    # Log ranges
    min_ai, max_ai = 0.5, max(500.0, analysis.ridge_point * 4.0)
    log_min_ai, log_max_ai = math.log10(min_ai), math.log10(max_ai)

    min_perf, max_perf = 0.05, analysis.peak_tflops * 1.5
    log_min_perf, log_max_perf = math.log10(min_perf), math.log10(max_perf)

    grid = [[" " for _ in range(plot_w)] for _ in range(plot_h)]

    # Draw ceilings
    for col in range(plot_w):
        col_frac = col / (plot_w - 1)
        ai = 10 ** (log_min_ai + col_frac * (log_max_ai - log_min_ai))
        mem_perf = (analysis.memory_bandwidth_gb_s * ai) / 1000.0  # in TFLOPs
        perf_ceil = min(analysis.peak_tflops, mem_perf)

        if perf_ceil >= min_perf:
            row_frac = (math.log10(perf_ceil) - log_min_perf) / (log_max_perf - log_min_perf)
            row = int((1.0 - row_frac) * (plot_h - 1))
            if 0 <= row < plot_h:
                grid[row][col] = "-" if perf_ceil == analysis.peak_tflops else "/"

    # Plot empirical points
    for pt in analysis.points:
        if pt.arithmetic_intensity > 0 and pt.achieved_tflops > 0:
            c_frac = (math.log10(pt.arithmetic_intensity) - log_min_ai) / (log_max_ai - log_min_ai)
            r_frac = (math.log10(max(min_perf, pt.achieved_tflops)) - log_min_perf) / (
                log_max_perf - log_min_perf
            )
            c_idx = int(c_frac * (plot_w - 1))
            r_idx = int((1.0 - r_frac) * (plot_h - 1))
            if 0 <= r_idx < plot_h and 0 <= c_idx < plot_w:
                grid[r_idx][c_idx] = "*"

    # Assemble lines
    for r in range(plot_h):
        row_str = "".join(grid[r])
        if r == 0:
            lines.append(f"| {analysis.peak_tflops:>5.1f} T |{row_str}|")
        elif r == plot_h // 2:
            lines.append(f"|  TFLOP |{row_str}|")
        elif r == plot_h - 1:
            lines.append(f"| {min_perf:>5.2f} T |{row_str}|")
        else:
            lines.append(f"|        |{row_str}|")

    lines.append(f"+--------+{'-' * plot_w}+")
    lines.append(f"         {min_ai:<6.1f} FLOPs/Byte (Arithmetic Intensity) {max_ai:>8.0f}")
    lines.append(f"\n[Verdict] Operating in {analysis.primary_regime} regime.")
    lines.append(
        f"          Peak MBU: {analysis.peak_mbu:.1f}%  |  Peak MFU: {analysis.peak_mfu:.1f}%"
    )

    return "\n".join(lines)
