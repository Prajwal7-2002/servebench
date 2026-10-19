"""Tests for v0.2 CLI commands: compare-models, find-context-limit, and workload flags."""

from pathlib import Path

from typer.testing import CliRunner

from servebench.cli import app
from servebench.metrics import summarize
from servebench.report import save_run
from servebench.results import (
    Environment,
    GpuTelemetrySummary,
    LevelResult,
    RequestResult,
    RunConfig,
    RunResult,
)

runner = CliRunner()


def _make_dummy_run(label: str, model: str, tok_s: float, ttft_s: float) -> RunResult:
    reqs = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=ttft_s,
            e2e=ttft_s + 0.1,
            output_tokens=10,
            chunk_times=[ttft_s, ttft_s + 0.1],
        )
        for i in range(10)
    ]
    summary = summarize(reqs, duration=1.0)
    # override output_throughput for explicit comparison testing
    summary.output_throughput = tok_s

    gpu = GpuTelemetrySummary(
        gpu_name="NVIDIA RTX 3050",
        gpu_index=0,
        avg_gpu_util=60.0,
        peak_gpu_util=70.0,
        avg_memory_used_mb=2500.0,
        peak_memory_used_mb=2700.0,
        avg_power_w=30.0,
        peak_power_w=40.0,
        total_energy_joules=100.0,
        tokens_per_joule=tok_s / 30.0,
    )

    lv = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs,
        summary=summary,
        gpu_telemetry=gpu,
    )

    return RunResult(
        label=label,
        config=RunConfig(
            url="http://127.0.0.1:8000/v1", model=model, mode="concurrency", levels=[1.0]
        ),
        environment=Environment(
            servebench_version="0.2.0",
            python="3.12.0",
            platform="win32",
            timestamp="2026-10-19T00:00:00Z",
            engine="llama.cpp",
        ),
        levels=[lv],
        bottleneck_diagnosis={"primary_bottleneck": "Balanced", "overall_health": "excellent"},
    )


def test_compare_models_cli(tmp_path: Path):
    run1 = _make_dummy_run("Llama-3-8B-Q4", "llama-3-8b-q4", tok_s=50.0, ttft_s=0.08)
    run2 = _make_dummy_run("Llama-3-8B-FP16", "llama-3-8b-fp16", tok_s=30.0, ttft_s=0.12)

    p1 = tmp_path / "run1.json"
    p2 = tmp_path / "run2.json"
    save_run(run1, p1)
    save_run(run2, p2)

    res = runner.invoke(app, ["compare-models", str(p1), str(p2)])
    assert res.exit_code == 0
    assert "Model Serving Comparison" in res.stdout
    assert "Llama-3-8B-Q4" in res.stdout
    assert "Llama-3-8B-FP16" in res.stdout
    assert "Tok/Joule" in res.stdout


def test_compare_models_requires_multiple_files(tmp_path: Path):
    p1 = tmp_path / "run1.json"
    p1.touch()
    res = runner.invoke(app, ["compare-models", str(p1)])
    assert res.exit_code != 0


def test_find_context_limit_cli(server):
    res = runner.invoke(
        app,
        [
            "find-context-limit",
            "--url",
            server.url,
            "--model",
            "mock-model",
            "--start",
            "16",
            "--max",
            "64",
            "--output-len",
            "4",
        ],
    )
    assert res.exit_code == 0
    assert "Finding context limit" in res.stdout
    assert "Maximum context supported" in res.stdout
