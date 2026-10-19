"""Tests for the Automated Bottleneck Diagnosis Engine."""

from servebench.diagnosis import diagnose_run
from servebench.metrics import summarize
from servebench.results import (
    Environment,
    GpuTelemetrySummary,
    LevelResult,
    RequestResult,
    RunConfig,
    RunResult,
)


def _make_dummy_env() -> Environment:
    return Environment(
        servebench_version="0.2.0",
        python="3.12.0",
        platform="win32",
        timestamp="2026-10-19T00:00:00Z",
        cpu="Mock CPU",
        ram_gb=32.0,
        gpus=["NVIDIA GeForce RTX 3050"],
    )


def test_diagnose_empty_run():
    run = RunResult(
        label="empty",
        config=RunConfig(url="http://x/v1", model="m", mode="concurrency", levels=[]),
        environment=_make_dummy_env(),
        levels=[],
    )
    report = diagnose_run(run)
    assert report.primary_bottleneck == "Insufficient Data"
    assert report.overall_health == "unknown"


def test_diagnose_client_lag_bottleneck():
    reqs = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.05,
            e2e=0.2,
            output_tokens=10,
            chunk_times=[0.05, 0.2],
        )
        for i in range(10)
    ]
    lv = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs,
        summary=summarize(reqs, 1.0, 50, 50),
        client_lag_p99_ms=120.0,  # >50 ms threshold
    )
    run = RunResult(
        label="client-lag-test",
        config=RunConfig(url="http://x/v1", model="m", mode="concurrency", levels=[1.0]),
        environment=_make_dummy_env(),
        levels=[lv],
    )
    report = diagnose_run(run)
    assert report.primary_bottleneck == "Client"
    assert report.overall_health == "bottlenecked"
    assert any("event loop is saturated" in f.headline for f in report.findings)


def test_diagnose_queue_slot_bottleneck():
    # Base level (c=1): TTFT = 100 ms, ITL = 20 ms
    c_times_base = [0.100 + 0.020 * j for j in range(11)]
    reqs_base = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.100,
            e2e=0.300,
            output_tokens=10,
            chunk_times=c_times_base,
        )
        for i in range(20)
    ]
    lv_base = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs_base,
        summary=summarize(reqs_base, 1.0, 50, 50),
    )

    # Peak level (c=8): TTFT = 350 ms (3.5x ballooning), ITL = 21 ms (flat 1.05x)
    c_times_peak = [0.350 + 0.021 * j for j in range(11)]
    reqs_peak = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.350,
            e2e=0.560,
            output_tokens=10,
            chunk_times=c_times_peak,
        )
        for i in range(20)
    ]
    lv_peak = LevelResult(
        mode="concurrency",
        value=8.0,
        duration=1.0,
        requests=reqs_peak,
        summary=summarize(reqs_peak, 1.0, 50, 50),
    )

    run = RunResult(
        label="queue-test",
        config=RunConfig(url="http://x/v1", model="m", mode="concurrency", levels=[1.0, 8.0]),
        environment=_make_dummy_env(),
        levels=[lv_base, lv_peak],
    )
    report = diagnose_run(run)
    assert report.primary_bottleneck == "Queue/Slot"
    assert any("Slot/Queue-Bound" in f.headline for f in report.findings)


def test_diagnose_memory_bandwidth_bottleneck():
    # Base level (c=1): TTFT = 100 ms, ITL = 15 ms
    c_times_base = [0.100 + 0.015 * j for j in range(11)]
    reqs_base = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.100,
            e2e=0.250,
            output_tokens=10,
            chunk_times=c_times_base,
        )
        for i in range(20)
    ]
    lv_base = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs_base,
        summary=summarize(reqs_base, 1.0, 50, 50),
    )

    # Peak level (c=8): TTFT = 120 ms, ITL = 30 ms (2.0x degradation), moderate GPU util
    c_times_peak = [0.120 + 0.030 * j for j in range(11)]
    reqs_peak = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.120,
            e2e=0.420,
            output_tokens=10,
            chunk_times=c_times_peak,
        )
        for i in range(20)
    ]
    gpu_telemetry = GpuTelemetrySummary(
        gpu_name="NVIDIA RTX 3050",
        gpu_index=0,
        avg_gpu_util=55.0,  # Moderate utilization < 75%
        peak_gpu_util=65.0,
        avg_memory_used_mb=2800.0,
        peak_memory_used_mb=3100.0,
        avg_power_w=35.0,
        peak_power_w=45.0,
        total_energy_joules=150.0,
        tokens_per_joule=4.2,
    )
    lv_peak = LevelResult(
        mode="concurrency",
        value=8.0,
        duration=1.0,
        requests=reqs_peak,
        summary=summarize(reqs_peak, 1.0, 50, 50),
        gpu_telemetry=gpu_telemetry,
    )

    run = RunResult(
        label="mem-bw-test",
        config=RunConfig(url="http://x/v1", model="m", mode="concurrency", levels=[1.0, 8.0]),
        environment=_make_dummy_env(),
        levels=[lv_base, lv_peak],
    )
    report = diagnose_run(run)
    assert report.primary_bottleneck == "Memory-Bandwidth"
    assert any("Memory-Bandwidth Bound" in f.headline for f in report.findings)


def test_diagnose_balanced_system():
    # Linear and smooth scaling
    c_times_base = [0.080 + 0.012 * j for j in range(11)]
    reqs_base = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.080,
            e2e=0.200,
            output_tokens=10,
            chunk_times=c_times_base,
        )
        for i in range(20)
    ]
    lv_base = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs_base,
        summary=summarize(reqs_base, 1.0, 50, 50),
    )

    c_times_peak = [0.095 + 0.0125 * j for j in range(11)]
    reqs_peak = [
        RequestResult(
            index=i,
            start=0.0,
            ok=True,
            ttft=0.095,
            e2e=0.220,
            output_tokens=10,
            chunk_times=c_times_peak,
        )
        for i in range(20)
    ]
    lv_peak = LevelResult(
        mode="concurrency",
        value=2.0,
        duration=1.0,
        requests=reqs_peak,
        summary=summarize(reqs_peak, 1.0, 50, 50),
    )

    run = RunResult(
        label="balanced-test",
        config=RunConfig(url="http://x/v1", model="m", mode="concurrency", levels=[1.0, 2.0]),
        environment=_make_dummy_env(),
        levels=[lv_base, lv_peak],
    )
    report = diagnose_run(run)
    assert report.primary_bottleneck == "Balanced"
    assert report.overall_health == "excellent"
