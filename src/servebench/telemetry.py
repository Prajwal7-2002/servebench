"""GPU Telemetry via NVML (nvidia-ml-py / pynvml).

Captures GPU compute utilization, memory utilization, VRAM usage, power draw, temperature,
and clock speeds during benchmarks. Computes energy efficiency (Tokens per Joule).

Designed with zero-fail tolerance: if NVML is not available, or running on non-NVIDIA
hardware (Apple Silicon, CPU, etc.), all telemetry gracefully degrades to None without
interrupting or slowing down the benchmark.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class GpuSample(BaseModel):
    """A point-in-time snapshot of GPU telemetry."""

    timestamp: float
    gpu_util: float = Field(description="GPU compute core utilization percentage (0-100)")
    mem_util: float = Field(description="Memory bus/controller utilization percentage (0-100)")
    vram_used_mb: float = Field(description="Used VRAM in MiB/MB")
    vram_total_mb: float = Field(description="Total VRAM in MiB/MB")
    power_w: float | None = Field(None, description="Power usage in Watts")
    temp_c: float | None = Field(None, description="GPU temperature in Celsius")
    sm_clock_mhz: float | None = Field(None, description="Streaming Multiprocessor clock in MHz")


class GpuTelemetrySummary(BaseModel):
    """Aggregated GPU metrics over a benchmark level or run."""

    device_name: str | None = None
    samples_count: int = 0
    duration_s: float = 0.0

    avg_gpu_util: float | None = Field(None, description="Average GPU compute utilization (%)")
    peak_gpu_util: float | None = Field(None, description="Peak GPU compute utilization (%)")

    avg_mem_util: float | None = Field(None, description="Average memory bus utilization (%)")
    peak_mem_util: float | None = Field(None, description="Peak memory bus utilization (%)")

    avg_vram_used_mb: float | None = Field(None, description="Average VRAM used (MB)")
    peak_vram_used_mb: float | None = Field(None, description="Peak VRAM used (MB)")
    vram_total_mb: float | None = Field(None, description="Total VRAM (MB)")

    avg_power_w: float | None = Field(None, description="Average power consumption in Watts")
    peak_power_w: float | None = Field(None, description="Peak power consumption in Watts")
    energy_consumed_joules: float | None = Field(
        None, description="Total energy consumed (avg_power * duration) in Joules"
    )

    avg_temp_c: float | None = Field(None, description="Average temperature in Celsius")
    peak_temp_c: float | None = Field(None, description="Peak temperature in Celsius")

    avg_sm_clock_mhz: float | None = Field(None, description="Average SM clock in MHz")

    tokens_per_joule: float | None = Field(
        None, description="Efficiency metric: Output Throughput (tok/s) / Average Power (W)"
    )


class NvmlHandle:
    """Lazy NVML wrapper that manages library initialization and queries cleanly."""

    _initialized: bool = False
    _available: bool | None = None
    _pynvml: Any = None

    @classmethod
    def is_available(cls) -> bool:
        if cls._available is not None:
            return cls._available
        try:
            import pynvml  # type: ignore

            cls._pynvml = pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            cls._initialized = True
            cls._available = count > 0
        except Exception:
            cls._available = False
        return cls._available

    @classmethod
    def get_handle(cls, index: int = 0) -> Any:
        if not cls.is_available() or cls._pynvml is None:
            return None
        try:
            return cls._pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            return None

    @classmethod
    def device_name(cls, index: int = 0) -> str | None:
        handle = cls.get_handle(index)
        if handle is None or cls._pynvml is None:
            return None
        try:
            name = cls._pynvml.nvmlDeviceGetName(handle)
            return name.decode("utf-8") if isinstance(name, bytes) else str(name)
        except Exception:
            return None

    @classmethod
    def sample(cls, index: int = 0) -> GpuSample | None:
        handle = cls.get_handle(index)
        if handle is None or cls._pynvml is None:
            return None
        p = cls._pynvml
        now = time.perf_counter()

        try:
            util = p.nvmlDeviceGetUtilizationRates(handle)
            gpu_u = float(util.gpu)
            mem_u = float(util.memory)
        except Exception:
            gpu_u, mem_u = 0.0, 0.0

        try:
            mem = p.nvmlDeviceGetMemoryInfo(handle)
            vram_used = float(mem.used) / (1024.0 * 1024.0)
            vram_total = float(mem.total) / (1024.0 * 1024.0)
        except Exception:
            vram_used, vram_total = 0.0, 0.0

        power_w: float | None = None
        try:
            # nvmlDeviceGetPowerUsage returns milliwatts
            mw = p.nvmlDeviceGetPowerUsage(handle)
            power_w = float(mw) / 1000.0
        except Exception:
            pass

        temp_c: float | None = None
        try:
            temp_c = float(p.nvmlDeviceGetTemperature(handle, p.NVML_TEMPERATURE_GPU))
        except Exception:
            pass

        sm_clock: float | None = None
        try:
            sm_clock = float(p.nvmlDeviceGetClockInfo(handle, p.NVML_CLOCK_SM))
        except Exception:
            pass

        return GpuSample(
            timestamp=now,
            gpu_util=gpu_u,
            mem_util=mem_u,
            vram_used_mb=round(vram_used, 1),
            vram_total_mb=round(vram_total, 1),
            power_w=round(power_w, 2) if power_w is not None else None,
            temp_c=round(temp_c, 1) if temp_c is not None else None,
            sm_clock_mhz=round(sm_clock, 1) if sm_clock is not None else None,
        )

    @classmethod
    def shutdown(cls) -> None:
        if cls._initialized and cls._pynvml is not None:
            try:
                cls._pynvml.nvmlShutdown()
            except Exception:
                pass
            cls._initialized = False
            cls._available = None


def summarize_samples(
    samples: list[GpuSample],
    device_name: str | None = None,
    output_throughput: float | None = None,
) -> GpuTelemetrySummary | None:
    """Compute aggregate statistics over collected GPU samples."""
    if not samples:
        return None

    n = len(samples)
    t0 = samples[0].timestamp
    t1 = samples[-1].timestamp
    duration = max(0.001, t1 - t0)

    gpu_utils = [s.gpu_util for s in samples]
    mem_utils = [s.mem_util for s in samples]
    vram_useds = [s.vram_used_mb for s in samples]
    powers = [s.power_w for s in samples if s.power_w is not None]
    temps = [s.temp_c for s in samples if s.temp_c is not None]
    clocks = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz is not None]

    vram_tot = samples[0].vram_total_mb if samples[0].vram_total_mb > 0 else None

    avg_pow = (sum(powers) / len(powers)) if powers else None
    peak_pow = max(powers) if powers else None
    joules = (avg_pow * duration) if avg_pow is not None else None

    # Tokens per Joule = Output Throughput (tokens/s) / Average Power (Joules/s)
    tokens_per_joule: float | None = None
    if output_throughput is not None and avg_pow and avg_pow > 0:
        tokens_per_joule = round(output_throughput / avg_pow, 2)

    return GpuTelemetrySummary(
        device_name=device_name,
        samples_count=n,
        duration_s=round(duration, 3),
        avg_gpu_util=round(sum(gpu_utils) / n, 1),
        peak_gpu_util=round(max(gpu_utils), 1),
        avg_mem_util=round(sum(mem_utils) / n, 1),
        peak_mem_util=round(max(mem_utils), 1),
        avg_vram_used_mb=round(sum(vram_useds) / n, 1),
        peak_vram_used_mb=round(max(vram_useds), 1),
        vram_total_mb=round(vram_tot, 1) if vram_tot is not None else None,
        avg_power_w=round(avg_pow, 2) if avg_pow is not None else None,
        peak_power_w=round(peak_pow, 2) if peak_pow is not None else None,
        energy_consumed_joules=round(joules, 2) if joules is not None else None,
        avg_temp_c=round(sum(temps) / len(temps), 1) if temps else None,
        peak_temp_c=round(max(temps), 1) if temps else None,
        avg_sm_clock_mhz=round(sum(clocks) / len(clocks), 1) if clocks else None,
        tokens_per_joule=tokens_per_joule,
    )


@dataclass
class GpuSampler:
    """Asynchronous background GPU sampler that runs concurrently with a load level."""

    gpu_index: int = 0
    interval_s: float = 0.2
    samples: list[GpuSample] = field(default_factory=list)
    device_name: str | None = None
    _task: asyncio.Task | None = None
    _running: bool = False

    def start(self) -> None:
        if not NvmlHandle.is_available():
            return
        self.device_name = NvmlHandle.device_name(self.gpu_index)
        self.samples.clear()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        while self._running:
            sample = NvmlHandle.sample(self.gpu_index)
            if sample:
                self.samples.append(sample)
            await asyncio.sleep(self.interval_s)

    async def stop(self, output_throughput: float | None = None) -> GpuTelemetrySummary | None:
        if not self._running:
            return None
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return summarize_samples(self.samples, self.device_name, output_throughput)
