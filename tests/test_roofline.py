"""Tests for hardware roofline model, MBU, MFU, and ASCII chart generation."""

from servebench.roofline import (
    compute_roofline,
    estimate_model_params,
    render_ascii_roofline,
    resolve_gpu_spec,
)


def test_estimate_model_params():
    params, bpp = estimate_model_params("qwen2.5:0.5b")
    assert params == 0.49
    assert bpp == 2.0

    params, bpp = estimate_model_params("qwen2.5:3b")
    assert params == 3.09

    params, bpp = estimate_model_params("qwen2.5:14b")
    assert params == 14.7

    params, bpp = estimate_model_params("llama3:8b-q4_k_m")
    assert params == 8.03
    assert bpp == 0.55

    params, bpp = estimate_model_params("mistral:7b-q8_0")
    assert params == 7.24
    assert bpp == 1.05

    # Sub-2-bit ternary / BitNet
    params, bpp = estimate_model_params("bitnet:1.58b")
    assert params == 1.58
    assert bpp == 0.28


def test_resolve_gpu_spec():
    spec_3050 = resolve_gpu_spec("NVIDIA GeForce RTX 3050 Laptop GPU")
    assert spec_3050.memory_bandwidth_gb_s == 192.0
    assert spec_3050.peak_fp16_tflops == 36.4

    spec_a100 = resolve_gpu_spec("A100-SXM4-80GB")
    assert spec_a100.memory_bandwidth_gb_s == 2039.0

    spec_h200 = resolve_gpu_spec("NVIDIA H200")
    assert spec_h200.memory_bandwidth_gb_s == 4800.0

    spec_b200 = resolve_gpu_spec("B200")
    assert spec_b200.memory_bandwidth_gb_s == 8000.0
    assert spec_b200.peak_fp16_tflops == 2250.0

    spec_l40s = resolve_gpu_spec("NVIDIA L40S")
    assert spec_l40s.memory_bandwidth_gb_s == 864.0

    spec_m4_max = resolve_gpu_spec("Apple M4 Max")
    assert spec_m4_max.memory_bandwidth_gb_s == 546.0

    spec_generic = resolve_gpu_spec("Unknown Accelerator")
    assert spec_generic.memory_bandwidth_gb_s > 0
    assert spec_generic.peak_fp16_tflops > 0


def test_compute_roofline():
    levels = [
        {"concurrency": 1, "output_tokens_per_s": 85.0},
        {"concurrency": 2, "output_tokens_per_s": 145.0},
        {"concurrency": 4, "output_tokens_per_s": 220.0},
    ]

    analysis = compute_roofline(
        model_name="qwen2.5:0.5b",
        levels_data=levels,
        gpu_name="rtx 3050",
    )

    assert analysis.gpu_name == "RTX 3050 Laptop"
    assert len(analysis.points) == 3
    assert analysis.peak_mbu > 0.0
    assert analysis.peak_mfu > 0.0
    assert analysis.ridge_point > 0.0

    # At low concurrency, LLMs are memory bandwidth bound
    assert analysis.primary_regime == "Memory-Bandwidth Bound"

    p1 = analysis.points[0]
    assert p1.concurrency == 1
    assert p1.mbu_percent > 0.0
    assert p1.regime == "Memory-Bandwidth Bound"


def test_render_ascii_roofline():
    levels = [{"concurrency": 1, "output_tokens_per_s": 100.0}]
    analysis = compute_roofline(
        model_name="qwen2.5:0.5b",
        levels_data=levels,
        gpu_name="rtx 3050",
    )
    chart = render_ascii_roofline(analysis, width=50, height=10)
    assert "Roofline Model:" in chart
    assert "Peak Bandwidth:" in chart
    assert "TFLOP" in chart
    assert "Peak MBU:" in chart
