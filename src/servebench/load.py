"""Load generation: closed-loop concurrency sweeps and open-loop Poisson arrivals."""

from __future__ import annotations

import asyncio
import itertools
import math
import random
import time
from collections.abc import Awaitable, Callable

import httpx
import numpy as np

from .client import RequestSpec, send_request
from .metrics import summarize
from .prompts import Prompt
from .results import LevelResult, RequestResult
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
        return float(np.percentile(self.lags, 99) * 1000) if self.lags else None


def poisson_delays(n: int, rate: float, seed: int = 0) -> list[float]:
    """Gaps between arrivals for a Poisson process of `rate` req/s (all zero for inf)."""
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    return [rng.expovariate(rate) for _ in range(n)]


def make_client(max_connections: int | None, timeout: float) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=None)
    return httpx.AsyncClient(
        limits=limits, timeout=httpx.Timeout(timeout, connect=min(timeout, 30.0))
    )


async def run_concurrency(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    prompts: list[Prompt],
    concurrency: int,
    counter: Counter | None = None,
) -> tuple[list[RequestResult], float]:
    """Closed loop: keep exactly `concurrency` requests in flight until all prompts are sent."""
    queue: asyncio.Queue[tuple[int, Prompt]] = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))
    results: list[RequestResult] = []
    t0 = time.perf_counter()

    async def worker() -> None:
        while True:
            try:
                i, p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await send_request(client, spec, p, index=i, t0=t0, counter=counter))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return sorted(results, key=lambda r: r.index), time.perf_counter() - t0


async def run_rate(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    prompts: list[Prompt],
    rate: float,
    seed: int = 0,
    max_concurrency: int | None = None,
    counter: Counter | None = None,
) -> tuple[list[RequestResult], float]:
    """Open loop: requests arrive as a Poisson process, whether or not earlier ones finished."""
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    t0 = time.perf_counter()
    delays = poisson_delays(len(prompts), rate, seed)

    async def one(i: int, p: Prompt) -> RequestResult:
        if sem is None:
            return await send_request(client, spec, p, index=i, t0=t0, counter=counter)
        async with sem:
            return await send_request(client, spec, p, index=i, t0=t0, counter=counter)

    tasks = []
    next_at = t0
    for i, (p, d) in enumerate(zip(prompts, delays, strict=True)):
        # schedule against absolute times so sleep overshoot does not accumulate
        next_at += d
        wait = next_at - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(one(i, p)))
    results = await asyncio.gather(*tasks)
    return list(results), time.perf_counter() - t0


async def run_level(
    spec: RequestSpec,
    prompts: list[Prompt],
    mode: str,
    value: float,
    *,
    timeout: float = 300.0,
    seed: int = 0,
    max_concurrency: int | None = None,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    counter: Counter | None = None,
) -> LevelResult:
    if mode == "concurrency":
        conns = int(value)
    else:
        conns = max_concurrency  # None = unlimited
    async with make_client(conns, timeout) as client:
        lag = LagMonitor()
        lag.start()
        if mode == "concurrency":
            reqs, dur = await run_concurrency(client, spec, prompts, int(value), counter)
        else:
            reqs, dur = await run_rate(client, spec, prompts, value, seed, max_concurrency,
                                       counter)
        lag_p99 = await lag.stop()
    return LevelResult(
        mode=mode,
        value=value,
        duration=dur,
        client_lag_p99_ms=lag_p99,
        requests=reqs,
        summary=summarize(reqs, dur, slo_ttft_ms, slo_tpot_ms),
    )


async def warmup(spec: RequestSpec, prompts: list[Prompt], timeout: float) -> list[RequestResult]:
    """Send a few requests one by one before measuring (loads the model, fills caches)."""
    async with make_client(1, timeout) as client:
        return [await send_request(client, spec, p, index=i) for i, p in enumerate(prompts)]


async def run_sweep(
    spec: RequestSpec,
    prompts_for: Callable[[int, int], list[Prompt]],
    mode: str,
    levels: list[float],
    num_requests: int | None,
    *,
    warmup_requests: int = 2,
    timeout: float = 300.0,
    seed: int = 0,
    max_concurrency: int | None = None,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    counter: Counter | None = None,
    on_level: Callable[[LevelResult], None] | None = None,
) -> list[LevelResult]:
    """Run every level. `prompts_for(n, seed)` returns n prompts; each level gets fresh ones."""
    await check_warmup(spec, prompts_for, warmup_requests, timeout, seed)
    measure = level_runner(
        spec, prompts_for, mode, num_requests, timeout=timeout, seed=seed,
        max_concurrency=max_concurrency, slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms,
        counter=counter, on_level=on_level,
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
        # refused, or (often on Windows) no answer before the connect timeout
        if err.startswith("ConnectError") or "ConnectTimeout" in err:
            raise RuntimeError(f"could not connect to {spec.url}. Is the server running? "
                               f"({err})")
        raise RuntimeError(f"all warm-up requests failed, first error: {err}")


def level_runner(
    spec: RequestSpec,
    prompts_for: Callable[[int, int], list[Prompt]],
    mode: str,
    num_requests: int | None,
    *,
    timeout: float = 300.0,
    seed: int = 0,
    max_concurrency: int | None = None,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
    counter: Counter | None = None,
    on_level: Callable[[LevelResult], None] | None = None,
) -> Callable[[float], Awaitable[LevelResult]]:
    """Return `measure(value)`, which runs one level; the k-th call uses seed + k."""
    calls = itertools.count()

    async def measure(value: float) -> LevelResult:
        k = next(calls)
        n = num_requests or default_num_requests(mode, value)
        level = await run_level(
            spec, prompts_for(n, seed + k), mode, value, timeout=timeout, seed=seed + k,
            max_concurrency=max_concurrency, slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms,
            counter=counter,
        )
        if on_level:
            on_level(level)
        return level

    return measure


def default_num_requests(mode: str, value: float) -> int:
    """Enough requests that each level reaches a steady state, without taking forever."""
    if mode == "concurrency":
        return max(16, 4 * int(value))
    return 64
