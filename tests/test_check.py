"""Tests for CI Performance Regression Gate (`servebench check`)."""

from pathlib import Path

from typer.testing import CliRunner

from servebench.check import evaluate_regression
from servebench.cli import app
from servebench.metrics import summarize
from servebench.report import save_run
from servebench.results import (
    Environment,
    LevelResult,
    RequestResult,
    RunConfig,
    RunResult,
)

runner = CliRunner()


def _make_run(
    label: str,
    ttft_s: float,
    tpot_s: float,
    tok_s: float,
    error_count: int = 0,
) -> RunResult:
    reqs = []
    for i in range(20):
        is_ok = i >= error_count
        ttft = ttft_s if is_ok else None
        e2e = (ttft_s + tpot_s * 10) if is_ok else None
        c_times = [ttft_s + tpot_s * j for j in range(11)] if is_ok else []
        reqs.append(
            RequestResult(
                index=i,
                start=0.0,
                ok=is_ok,
                error=None if is_ok else "HTTP 500",
                ttft=ttft,
                e2e=e2e,
                output_tokens=10 if is_ok else 0,
                chunk_times=c_times,
            )
        )

    summary = summarize(reqs, duration=1.0)
    summary.output_throughput = tok_s

    lv = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs,
        summary=summary,
    )

    return RunResult(
        label=label,
        config=RunConfig(
            url="http://127.0.0.1:8000/v1", model="test-model", mode="concurrency", levels=[1.0]
        ),
        environment=Environment(
            servebench_version="1.0.1",
            python="3.12.0",
            platform="win32",
            timestamp="2026-10-19T00:00:00Z",
        ),
        levels=[lv],
    )


def test_evaluate_regression_pass():
    base = _make_run("baseline", ttft_s=0.100, tpot_s=0.020, tok_s=100.0)
    # Candidate within 5% variation -> should pass
    cand = _make_run("candidate", ttft_s=0.103, tpot_s=0.0205, tok_s=98.0)

    report = evaluate_regression(cand, base)
    assert report.passed is True
    assert len(report.failures) == 0


def test_evaluate_regression_ttft_failure():
    base = _make_run("baseline", ttft_s=0.100, tpot_s=0.020, tok_s=100.0)
    # Candidate TTFT +30% (>15% default limit)
    cand = _make_run("candidate", ttft_s=0.130, tpot_s=0.020, tok_s=100.0)

    report = evaluate_regression(cand, base)
    assert report.passed is False
    assert any("TTFT p50 degraded by +30.0%" in f for f in report.failures)


def test_evaluate_regression_throughput_failure():
    base = _make_run("baseline", ttft_s=0.100, tpot_s=0.020, tok_s=100.0)
    # Candidate throughput dropped by 25% (>10% default limit)
    cand = _make_run("candidate", ttft_s=0.100, tpot_s=0.020, tok_s=75.0)

    report = evaluate_regression(cand, base)
    assert report.passed is False
    assert any("throughput dropped by 25.0%" in f for f in report.failures)


def test_evaluate_regression_error_rate_failure():
    base = _make_run("baseline", ttft_s=0.100, tpot_s=0.020, tok_s=100.0)
    # Candidate has 2 errors out of 20 = 10% error rate (>1% limit)
    cand = _make_run("candidate", ttft_s=0.100, tpot_s=0.020, tok_s=95.0, error_count=2)

    report = evaluate_regression(cand, base)
    assert report.passed is False
    assert any("error rate" in f for f in report.failures)


def test_check_cli_pass_and_fail(tmp_path: Path):
    base_file = tmp_path / "base.json"
    cand_pass_file = tmp_path / "cand_pass.json"
    cand_fail_file = tmp_path / "cand_fail.json"

    base = _make_run("baseline", ttft_s=0.100, tpot_s=0.020, tok_s=100.0)
    cand_pass = _make_run("cand_pass", ttft_s=0.102, tpot_s=0.020, tok_s=99.0)
    cand_fail = _make_run("cand_fail", ttft_s=0.150, tpot_s=0.020, tok_s=60.0)

    save_run(base, base_file)
    save_run(cand_pass, cand_pass_file)
    save_run(cand_fail, cand_fail_file)

    # 1. Passing run returns exit code 0
    res_pass = runner.invoke(app, ["check", str(cand_pass_file), "--baseline", str(base_file)])
    assert res_pass.exit_code == 0
    assert "PASSED" in res_pass.stdout
    assert "All performance metrics are within" in res_pass.stdout

    # 2. Failing run returns exit code 1
    res_fail = runner.invoke(app, ["check", str(cand_fail_file), "--baseline", str(base_file)])
    assert res_fail.exit_code == 1
    assert "FAILED" in res_fail.stdout
    assert "Performance Regressions Detected" in res_fail.stdout
