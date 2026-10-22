"""Tests for automated server tuning optimizer and config recommender."""

import pytest

from servebench.results import (
    Environment,
    LevelResult,
    RunConfig,
    RunResult,
    Stats,
    Summary,
)
from servebench.tune import evaluate_tuning, generate_engine_config


@pytest.fixture
def sample_run_result():
    summary_c1 = Summary(
        total_requests=10,
        completed=10,
        ok=10,
        failed=0,
        duration=5.0,
        request_throughput=2.0,
        output_throughput=100.0,
        goodput=100.0,
        slo_ttft_ms=500.0,
        slo_tpot_ms=50.0,
        ttft=Stats(count=10, mean=0.05, p50=0.045, p90=0.06, p99=0.07),
        tpot=Stats(count=10, mean=0.01, p50=0.01),
        itl=Stats(count=10, mean=0.01, p50=0.01),
        e2e=Stats(count=10, mean=0.5, p50=0.5),
    )

    summary_c4 = Summary(
        total_requests=20,
        completed=20,
        ok=20,
        failed=0,
        duration=5.0,
        request_throughput=4.0,
        output_throughput=300.0,
        goodput=300.0,
        slo_ttft_ms=500.0,
        slo_tpot_ms=50.0,
        ttft=Stats(count=20, mean=0.09, p50=0.085, p90=0.11, p99=0.13),
        tpot=Stats(count=20, mean=0.015, p50=0.014),
        itl=Stats(count=20, mean=0.015, p50=0.014),
        e2e=Stats(count=20, mean=0.8, p50=0.8),
    )

    summary_c8 = Summary(
        total_requests=20,
        completed=20,
        ok=20,
        failed=0,
        duration=5.0,
        request_throughput=4.2,
        output_throughput=310.0,
        goodput=310.0,
        slo_ttft_ms=500.0,
        slo_tpot_ms=50.0,
        ttft=Stats(count=20, mean=0.45, p50=0.42, p90=0.55, p99=0.65),
        tpot=Stats(count=20, mean=0.02, p50=0.02),
        itl=Stats(count=20, mean=0.02, p50=0.02),
        e2e=Stats(count=20, mean=1.8, p50=1.8),
    )

    levels = [
        LevelResult(mode="concurrency", value=1.0, duration=5.0, requests=[], summary=summary_c1),
        LevelResult(mode="concurrency", value=4.0, duration=5.0, requests=[], summary=summary_c4),
        LevelResult(mode="concurrency", value=8.0, duration=5.0, requests=[], summary=summary_c8),
    ]

    return RunResult(
        label="test",
        config=RunConfig(
            url="http://127.0.0.1:11434/v1", model="qwen2.5:0.5b", levels=[1.0, 4.0, 8.0]
        ),
        environment=Environment(
            servebench_version="1.1.0",
            python="3.12",
            platform="Windows",
            timestamp="2026-10-22T00:00:00Z",
            engine="Ollama",
        ),
        levels=levels,
    )


def test_evaluate_tuning(sample_run_result):
    report = evaluate_tuning(
        sample_run_result,
        target_ttft_p50_ms=150.0,
        target_itl_p50_ms=30.0,
    )

    # c=4 had 300 tok/s within 85ms TTFT (well within 150ms SLO)
    # c=8 had 310 tok/s but TTFT jumped to 420ms (>150ms SLO)
    # The optimal knee is c=4
    assert report.optimal_concurrency == 4
    assert report.peak_sustainable_tok_s == 300.0
    assert report.recommendation.engine == "ollama"
    assert "OLLAMA_NUM_PARALLEL" in report.recommendation.recommended_env_vars
    assert report.recommendation.recommended_env_vars["OLLAMA_NUM_PARALLEL"] == "4"


def test_generate_engine_config_engines():
    rec_vllm = generate_engine_config("vllm", "meta-llama/Llama-3-8B", 8, 450.0)
    assert any("--max-num-seqs 8" in flag for flag in rec_vllm.recommended_cli_flags)
    assert any("--enable-chunked-prefill" in flag for flag in rec_vllm.recommended_cli_flags)

    rec_llama = generate_engine_config("llama.cpp", "qwen2.5-0.5b", 4, 250.0)
    assert any("-np 4" in flag for flag in rec_llama.recommended_cli_flags)
    assert any("-cb" in flag for flag in rec_llama.recommended_cli_flags)

    rec_mini = generate_engine_config("mini-infer", "qwen2.5-0.5b", 6, 320.0)
    assert any("--max-batch-size 6" in flag for flag in rec_mini.recommended_cli_flags)
