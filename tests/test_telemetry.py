"""Unit tests for GPU telemetry via NVML and energy efficiency."""

import asyncio
from unittest.mock import patch

from servebench.telemetry import (
    GpuSample,
    GpuSampler,
    NvmlHandle,
    summarize_samples,
)


def test_summarize_samples():
    s1 = GpuSample(
        timestamp=100.0,
        gpu_util=20.0,
        mem_util=10.0,
        vram_used_mb=2000.0,
        vram_total_mb=6000.0,
        power_w=50.0,
        temp_c=55.0,
        sm_clock_mhz=1200.0,
    )
    s2 = GpuSample(
        timestamp=102.0,
        gpu_util=60.0,
        mem_util=30.0,
        vram_used_mb=2500.0,
        vram_total_mb=6000.0,
        power_w=100.0,
        temp_c=65.0,
        sm_clock_mhz=1600.0,
    )

    summary = summarize_samples([s1, s2], device_name="Test GPU", output_throughput=150.0)
    assert summary is not None
    assert summary.device_name == "Test GPU"
    assert summary.samples_count == 2
    assert summary.duration_s == 2.0
    assert summary.avg_gpu_util == 40.0
    assert summary.peak_gpu_util == 60.0
    assert summary.avg_vram_used_mb == 2250.0
    assert summary.peak_vram_used_mb == 2500.0
    assert summary.avg_power_w == 75.0
    assert summary.peak_power_w == 100.0
    # Energy = avg_power (75 W) * duration (2.0 s) = 150 Joules
    assert summary.energy_consumed_joules == 150.0
    # Tokens per Joule = output_throughput (150 tok/s) / avg_power (75 W) = 2.0
    assert summary.tokens_per_joule == 2.0
    assert summary.avg_temp_c == 60.0
    assert summary.peak_temp_c == 65.0
    assert summary.avg_sm_clock_mhz == 1400.0


def test_summarize_samples_empty():
    assert summarize_samples([]) is None


@patch.object(NvmlHandle, "is_available", return_value=True)
@patch.object(NvmlHandle, "device_name", return_value="Mock RTX 3050")
@patch.object(
    NvmlHandle,
    "sample",
    return_value=GpuSample(
        timestamp=100.0,
        gpu_util=45.0,
        mem_util=15.0,
        vram_used_mb=1500.0,
        vram_total_mb=6144.0,
        power_w=25.0,
        temp_c=58.0,
        sm_clock_mhz=1000.0,
    ),
)
async def test_gpu_sampler_background(mock_sample, mock_name, mock_avail):
    sampler = GpuSampler(interval_s=0.01)
    sampler.start()
    await asyncio.sleep(0.04)
    summary = await sampler.stop(output_throughput=50.0)

    assert summary is not None
    assert summary.device_name == "Mock RTX 3050"
    assert summary.samples_count >= 1
    assert summary.avg_gpu_util == 45.0
    assert summary.avg_power_w == 25.0
    # 50 tok/s / 25 W = 2.0 tok/Joule
    assert summary.tokens_per_joule == 2.0


def test_nvml_handle_graceful_fallback():
    with patch("pynvml.nvmlInit", side_effect=Exception("No GPU")):
        NvmlHandle._available = None
        assert NvmlHandle.is_available() is False
        assert NvmlHandle.get_handle(0) is None
        assert NvmlHandle.device_name(0) is None
        assert NvmlHandle.sample(0) is None
