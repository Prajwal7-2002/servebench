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
    spec_3050 = resolve_gpu_spec("NVIDIA GeForce RTX 3050 Laptop GPU")  # 4 GB laptop
    assert spec_3050.memory_bandwidth_gb_s == 192.0
    assert spec_3050.peak_fp16_tflops == 24.0
    spec_3050_6gb = resolve_gpu_spec("NVIDIA GeForce RTX 3050 6GB Laptop GPU")
    assert spec_3050_6gb.memory_bandwidth_gb_s == 168.0  # 96-bit GDDR6
    assert resolve_gpu_spec("NVIDIA GeForce RTX 3050").peak_fp16_tflops == 36.4  # desktop
    assert resolve_gpu_spec("NVIDIA GeForce RTX 4060").peak_fp16_tflops == 60.0  # dense

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

    assert analysis.gpu_name == "RTX 3050 (desktop)"
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


def test_engine_reported_size_beats_name_guess():
    # Ollama's qwen2.5:0.5b is 4-bit (398 MB of weights); the name suggests 16-bit.
    levels = [{"concurrency": 1, "output_tokens_per_s": 200.0}]
    guessed = compute_roofline("qwen2.5:0.5b", levels, gpu_name="RTX 3050 6GB Laptop GPU")
    real = compute_roofline(
        "qwen2.5:0.5b",
        levels,
        gpu_name="RTX 3050 6GB Laptop GPU",
        param_count_billions=0.494,
        weights_gb=0.398,
    )
    assert guessed.model_source == "name guess"
    assert any("guessed from the name" in n for n in guessed.notes)
    assert real.model_source == "engine" and abs(real.model_size_gb - 0.398) < 0.01
    assert abs(real.points[0].mbu_percent - 200 * 0.398 / 168 * 100) < 0.1
    assert guessed.points[0].mbu_percent > 2 * real.points[0].mbu_percent


def test_impossible_utilisation_is_reported_not_clamped():
    levels = [{"concurrency": 1, "output_tokens_per_s": 1000.0}]
    a = compute_roofline("big:70b", levels, gpu_name="rtx 4090")
    assert a.peak_mbu > 100
    assert any("physically impossible" in n for n in a.notes)


def test_missing_prompt_throughput_is_not_invented():
    levels = [{"concurrency": 1, "output_tokens_per_s": 100.0}]
    a = compute_roofline(
        "m:1b", levels, gpu_name="rtx 4090", param_count_billions=1.0, weights_gb=2.0
    )
    expected_tflops = 100.0 * 2 * 1e9 / 1e12  # generated tokens only
    assert abs(a.points[0].achieved_tflops - expected_tflops) < 1e-3
    assert any("Prompt throughput was not available" in n for n in a.notes)


def test_unknown_gpu_is_flagged():
    a = compute_roofline(
        "m:1b", [{"concurrency": 1, "output_tokens_per_s": 10.0}], gpu_name="Mystery Accelerator"
    )
    assert a.gpu_source == "generic guess"
    assert any("not in the table" in n for n in a.notes)


def test_parse_param_size():
    from servebench.cli import _parse_param_size

    assert abs(_parse_param_size("494.03M") - 0.49403) < 1e-9
    assert _parse_param_size("7.6B") == 7.6
    assert _parse_param_size("unknown") is None


def test_roofline_cli_requires_a_run_file():
    from typer.testing import CliRunner

    from servebench.cli import app

    res = CliRunner().invoke(app, ["roofline", "--model", "qwen2.5:0.5b"])
    assert res.exit_code != 0
    assert "run file" in res.output
