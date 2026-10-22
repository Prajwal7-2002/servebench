"""Load generation: closed-loop concurrency sweeps and open-loop Poisson arrivals."""

from __future__ import annotations

import asyncio
import itertools
import math
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator

import aiohttp

from .client import RequestSpec, send_request
from .metrics import percentile, summarize, summarize_repeats
from .prometheus import extract_standard_metrics, scrape_prometheus
from .prompts import Prompt
from .results import LevelResult, RequestResult
from .telemetry import GpuSampler
from .tokens import Counter


class LagMonitor:
    """Measures how long a ready callback waits for the event loop.

    This is independent of timer resolution (Windows timers tick at ~15 ms), so a high value
    really means the client loop is busy and timings include client-side delay.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.lags: list[float] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self.interval)
            fut = loop.create_future()
            t = time.perf_counter()
            loop.call_soon(fut.set_result, None)
            await fut
            self.lags.append(time.perf_counter() - t)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> float | None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return percentile(sorted(self.lags), 99) * 1000 if self.lags else None


def poisson_delays(n: int, rate: float, seed: int = 0) -> list[float]:
    """Gaps between arrivals for a Poisson process of `rate` req/s (all zero for inf)."""
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    return [rng.expovariate(rate) for _ in range(n)]


def make_session(max_connections: int | None, timeout: float) -> aiohttp.ClientSession:
    """One session per level. `timeout` is per read (no gap between chunks may exceed it),
    like a stalled stream; there is no total limit, since long answers are legitimate."""
    connector = aiohttp.TCPConnector(limit=max_connections or 0, limit_per_host=0)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=None, sock_connect=min(timeout, 30.0), sock_read=timeout
        ),
    )


async def run_concurrency(
    client: aiohttp.ClientSession,
    spec: RequestSpec,
    prompts: Iterable[Prompt],
    concurrency: int,
    counter: Counter | None = None,
    duration: float | None = None,
) -> tuple[list[RequestResult], float]:
    """Closed loop: keep exactly `concurrency` requests in flight until the prompts run out
    or, with `duration`, until no new request should start (in-flight ones then finish)."""
    it = enumerate(prompts)
    results: list[RequestResult] = []
    t0 = time.perf_counter()
    deadline = t0 + duration if duration else None

    async def worker() -> None:
        while deadline is None or time.perf_counter() < deadline:
            try:
                i, p = next(it)  # one event-loop thread, so no lock needed
            except StopIteration:
                return
            results.append(await send_request(client, spec, p, index=i, t0=t0, counter=counter))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return sorted(results, key=lambda r: r.index), time.perf_counter() - t0


async def run_rate(
    client: aiohttp.ClientSession,
    spec: RequestSpec,
    prompts: Iterable[Prompt],
    rate: float,
    seed: int = 0,
    max_concurrency: int | None = None,
    counter: Counter | None = None,
    duration: float | None = None,
    burstiness: float = 1.0,
) -> tuple[list[RequestResult], float]:
    """Open loop: requests arrive as a Poisson (or Gamma if burstiness != 1.0) process.
    With `duration`, arrivals stop at the deadline and the in-flight requests finish."""
    if duration and math.isinf(rate):
        raise ValueError("a duration needs a finite rate (rate=inf sends everything at once)")
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    rng = random.Random(seed)
    t0 = time.perf_counter()
    deadline = t0 + duration if duration else None

    async def one(i: int, p: Prompt) -> RequestResult:
        if sem is None:
            return await send_request(client, spec, p, index=i, t0=t0, counter=counter)
        async with sem:
            return await send_request(client, spec, p, index=i, t0=t0, counter=counter)

    tasks = []
    next_at = t0
    for i, p in enumerate(prompts):
        if math.isinf(rate):
            delay = 0.0
        elif burstiness != 1.0 and burstiness > 0:
            delay = rng.gammavariate(burstiness, 1.0 / (rate * burstiness))
        else:
            delay = rng.expovariate(rate)
        next_at += delay
        if deadline is not None and next_at >= deadline:
            break
        wait = next_at - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(one(i, p)))
    results = await asyncio.gather(*tasks)
    return list(results), time.perf_counter() - t0


async def run_level(
    spec: RequestSpec,
    prompts: Iterable[Prompt],
    mode: str,
    value: float,
    *,
    duration: float | None = None,
    timeout: float = 300.0,
    seed: int = 0,
    max_concurrency: int | None = None,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    counter: Counter | None = None,
    burstiness: float = 1.0,
    gpu_telemetry: bool = False,
    gpu_index: int = 0,
    metrics_url: str | None = None,
) -> LevelResult:
    if mode == "concurrency":
        conns = int(value)
    else:
        conns = max_concurrency  # None = unlimited

    gpu_sampler = GpuSampler(gpu_index=gpu_index) if gpu_telemetry else None
    if gpu_sampler:
        gpu_sampler.start()

    start_metrics = await scrape_prometheus(metrics_url) if metrics_url else None

    async with make_session(conns, timeout) as client:
        lag = LagMonitor()
        lag.start()
        if mode == "concurrency":
            reqs, dur = await run_concurrency(client, spec, prompts, int(value), counter, duration)
        else:
            reqs, dur = await run_rate(
                client,
                spec,
                prompts,
                value,
                seed,
                max_concurrency,
                counter,
                duration,
                burstiness=burstiness,
            )
        lag_p99 = await lag.stop()

    sm = summarize(reqs, dur, slo_ttft_ms, slo_tpot_ms)
    gpu_summary = await gpu_sampler.stop(sm.output_throughput) if gpu_sampler else None
    end_metrics = await scrape_prometheus(metrics_url) if metrics_url else None
    prom_metrics = extract_standard_metrics(start_metrics, end_metrics) if end_metrics else None
    sm.gpu_telemetry = gpu_summary

    return LevelResult(
        mode=mode,
        value=value,
        duration=dur,
        client_lag_p99_ms=lag_p99,
        requests=reqs,
        summary=sm,
        gpu_telemetry=gpu_summary,
        prometheus_metrics=prom_metrics,
    )


async def warmup(spec: RequestSpec, prompts: list[Prompt], timeout: float) -> list[RequestResult]:
    """Send a few requests one by one before measuring (loads the model, fills caches)."""
    async with make_session(1, timeout) as client:
        return [await send_request(client, spec, p, index=i) for i, p in enumerate(prompts)]


async def run_sweep(
    spec: RequestSpec,
    prompts_for: Callable[[int, int], list[Prompt]],
    mode: str,
    levels: list[float],
    num_requests: int | None,
    *,
    duration: float | None = None,
    min_duration: float | None = None,
    repeat: int = 1,
    warmup_requests: int = 2,
    timeout: float = 300.0,
    seed: int = 0,
    max_concurrency: int | None = None,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    counter: Counter | None = None,
    on_level: Callable[[LevelResult], None] | None = None,
    burstiness: float = 1.0,
    gpu_telemetry: bool = False,
    gpu_index: int = 0,
    metrics_url: str | None = None,
) -> list[LevelResult]:
    """Run every level. `prompts_for(n, seed)` returns n prompts; each level gets fresh ones."""
    await check_warmup(spec, prompts_for, warmup_requests, timeout, seed)
    measure = level_runner(
        spec,
        prompts_for,
        mode,
        num_requests,
        duration=duration,
        min_duration=min_duration,
        repeat=repeat,
        timeout=timeout,
        seed=seed,
        max_concurrency=max_concurrency,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        counter=counter,
        on_level=on_level,
        burstiness=burstiness,
        gpu_telemetry=gpu_telemetry,
        gpu_index=gpu_index,
        metrics_url=metrics_url,
    )
    return [await measure(value) for value in levels]


async def check_warmup(
    spec: RequestSpec,
    prompts_for: Callable[[int, int], list[Prompt]],
    n: int,
    timeout: float,
    seed: int = 0,
) -> None:
    """Run the warm-up requests; raise if every one of them fails (wrong URL, model, ...)."""
    if not n:
        return
    wr = await warmup(spec, prompts_for(n, seed + 10_000), timeout)
    bad = [r for r in wr if not r.ok]
    if len(bad) == len(wr):
        err = bad[0].error or ""
        # the client reports refused and timed-out connects both as "connect error"
        if err.startswith("connect error"):
            raise RuntimeError(f"could not connect to {spec.url}. Is the server running? ({err})")
        raise RuntimeError(f"all warm-up requests failed, first error: {err}")


def level_runner(
    spec: RequestSpec,
    prompts_for: Callable[[int, int], list[Prompt]],
    mode: str,
    num_requests: int | None,
    *,
    duration: float | None = None,
    min_duration: float | None = None,
    repeat: int = 1,
    timeout: float = 300.0,
    seed: int = 0,
    max_concurrency: int | None = None,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    counter: Counter | None = None,
    on_level: Callable[[LevelResult], None] | None = None,
    burstiness: float = 1.0,
    gpu_telemetry: bool = False,
    gpu_index: int = 0,
    metrics_url: str | None = None,
) -> Callable[[float], Awaitable[LevelResult]]:
    """Return `measure(value)`, which runs one level; the k-th call uses seed + k * 1000.

    Without `duration` a level sends `num_requests` (or a default per load). With it, a level
    runs for `duration` seconds on an endless prompt stream, capped by `num_requests` if given.
    With `repeat > 1`, each level is executed N times and statistics (mean ± std) are recorded.
    """
    effective_duration = max(duration or 0.0, min_duration or 0.0) or None
    calls = itertools.count()

    async def measure(value: float) -> LevelResult:
        k = next(calls)
        trials: list[LevelResult] = []
        for r_idx in range(repeat):
            trial_seed = seed + k * 1000 + r_idx
            prompts: Iterable[Prompt]
            if effective_duration:
                prompts = prompt_stream(prompts_for, trial_seed)
                if num_requests:
                    prompts = itertools.islice(prompts, num_requests)
            else:
                prompts = prompts_for(num_requests or default_num_requests(mode, value), trial_seed)
            trial = await run_level(
                spec,
                prompts,
                mode,
                value,
                duration=effective_duration,
                timeout=timeout,
                seed=trial_seed,
                max_concurrency=max_concurrency,
                slo_ttft_ms=slo_ttft_ms,
                slo_tpot_ms=slo_tpot_ms,
                counter=counter,
                burstiness=burstiness,
                gpu_telemetry=gpu_telemetry,
                gpu_index=gpu_index,
                metrics_url=metrics_url,
            )
            trials.append(trial)

        if repeat <= 1:
            level = trials[0]
        else:
            all_requests: list[RequestResult] = []
            idx_offset = 0
            for tr in trials:
                for req in tr.requests:
                    req_copy = req.model_copy()
                    req_copy.index = idx_offset
                    idx_offset += 1
                    all_requests.append(req_copy)

            total_duration = sum(tr.duration for tr in trials)
            combined_summary = summarize(all_requests, total_duration, slo_ttft_ms, slo_tpot_ms)
            repeat_summaries = [tr.summary for tr in trials]
            repeat_metrics = summarize_repeats(repeat_summaries)
            lags = [tr.client_lag_p99_ms for tr in trials if tr.client_lag_p99_ms is not None]
            combined_lag = max(lags) if lags else None

            # Inherit GPU and Prometheus metrics from the trials
            last_gpu = next((tr.gpu_telemetry for tr in reversed(trials) if tr.gpu_telemetry), None)
            last_prom = next(
                (tr.prometheus_metrics for tr in reversed(trials) if tr.prometheus_metrics), None
            )
            combined_summary.gpu_telemetry = last_gpu

            level = LevelResult(
                mode=mode,
                value=value,
                duration=total_duration,
                client_lag_p99_ms=combined_lag,
                requests=all_requests,
                summary=combined_summary,
                repeat_count=repeat,
                repeats=repeat_summaries,
                repeat_metrics=repeat_metrics,
                gpu_telemetry=last_gpu,
                prometheus_metrics=last_prom,
            )

        if on_level:
            on_level(level)
        return level

    return measure


def prompt_stream(
    prompts_for: Callable[[int, int], list[Prompt]], seed: int, batch: int = 256
) -> Iterator[Prompt]:
    """Endless prompts for timed levels, made in batches (each batch its own seed, so
    synthetic prompts stay distinct and nothing is generated that isn't used)."""
    for b in itertools.count():
        yield from prompts_for(batch, seed * 100_003 + b)


def default_num_requests(mode: str, value: float) -> int:
    """Enough requests that each level reaches a steady state, without taking forever."""
    if mode == "concurrency":
        return max(16, 4 * int(value))
    return 64
