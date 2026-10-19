"""Tests for the stable public Python API:
`servebench.benchmark`, `check_regression`, `compare`.
"""

from pathlib import Path

import servebench
from servebench.metrics import summarize
from servebench.report import save_run
from servebench.results import Environment, LevelResult, RequestResult, RunConfig, RunResult


def _dummy_run(label: str, tok_s: float) -> RunResult:
    reqs = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.05,
            e2e=0.15,
            output_tokens=10,
            chunk_times=[0.05, 0.15],
        )
        for i in range(10)
    ]
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
            url="http://127.0.0.1:8000/v1", model="m", mode="concurrency", levels=[1.0]
        ),
        environment=Environment(
            servebench_version=servebench.__version__,
            python="3.12.0",
            platform="win32",
            timestamp="2026-10-19T00:00:00Z",
        ),
        levels=[lv],
    )


def test_top_level_exports():
    assert hasattr(servebench, "benchmark")
    assert hasattr(servebench, "check_regression")
    assert hasattr(servebench, "compare")
    assert hasattr(servebench, "CheckReport")
    assert hasattr(servebench, "CheckThresholds")
    assert hasattr(servebench, "RunResult")
    assert hasattr(servebench, "LevelResult")
    assert hasattr(servebench, "RequestResult")
    assert hasattr(servebench, "Summary")
    assert hasattr(servebench, "Stats")
    assert hasattr(servebench, "__version__")


def test_compare_api(tmp_path: Path):
    r1 = _dummy_run("run-1", 50.0)
    r2 = _dummy_run("run-2", 80.0)
    p2 = tmp_path / "run2.json"
    save_run(r2, p2)

    # compare handles both RunResult instances and Path/str
    runs = servebench.compare([r1, p2, str(p2)])
    assert len(runs) == 3
    assert runs[0].label == "run-1"
    assert runs[1].label == "run-2"
    assert runs[2].label == "run-2"


def test_check_regression_api():
    r1 = _dummy_run("base", 100.0)
    r2 = _dummy_run("cand", 98.0)
    report = servebench.check_regression(r2, r1)
    assert report.passed is True

    # With strict 1% max throughput drop threshold
    strict = servebench.CheckThresholds(max_throughput_pct=1.0)
    report_strict = servebench.check_regression(r2, r1, thresholds=strict)
    assert report_strict.passed is False


def test_benchmark_api(server):
    res = servebench.benchmark(
        url=server.url,
        model="mock-model",
        mode="concurrency",
        levels=[1.0],
        num_requests=3,
        warmup=1,
        input_len=16,
        output_len=4,
        gpu_telemetry=False,
    )
    assert isinstance(res, servebench.RunResult)
    assert len(res.levels) == 1
    assert res.levels[0].summary.completed >= 3
    assert res.levels[0].summary.output_throughput > 0
