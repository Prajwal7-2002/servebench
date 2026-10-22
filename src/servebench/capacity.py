"""Find the highest load that still meets the SLOs.

Search: double the load from `start` until a level fails (or `max_value` passes), then
binary-search between the last pass and the first fail until the gap is within `tolerance`
of the passing value (and at least 1 for whole-number concurrency).

A level passes when no request failed and at least `target` of the requests met the SLOs.
Each level is measured once, so a noisy server near its limit can pass a level above one
that failed; the search assumes latency grows with load.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .results import CapacityResult, LevelResult


@dataclass
class SearchResult:
    capacity: float | None
    first_fail: float | None
    reason: str  # "converged", "max_reached" or "start_failed"
    tried: list[tuple[float, bool]] = field(default_factory=list)


async def search_capacity(
    probe: Callable[[float], Awaitable[bool]],
    *,
    start: float,
    max_value: float,
    integer: bool,
    tolerance: float = 0.1,
) -> SearchResult:
    """Generic search over `probe(value) -> passed`; knows nothing about HTTP."""
    tried: list[tuple[float, bool]] = []

    async def check(v: float) -> bool:
        ok = await probe(v)
        tried.append((v, ok))
        return ok

    lo: float | None = None
    v = min(start, max_value)
    while True:
        if not await check(v):
            hi = v
            break
        lo = v
        if v >= max_value:
            return SearchResult(lo, None, "max_reached", tried)
        v = min(v * 2, max_value)
    if lo is None:
        return SearchResult(None, hi, "start_failed", tried)

    min_gap = 1 if integer else 0
    while hi - lo > max(min_gap, tolerance * lo):
        mid = (lo + hi) // 2 if integer else (lo + hi) / 2
        if await check(mid):
            lo = mid
        else:
            hi = mid
    return SearchResult(lo, hi, "converged", tried)


def passes(level: LevelResult, target: float) -> bool:
    s = level.summary
    return s.failed == 0 and s.slo_attainment is not None and s.slo_attainment >= target


async def find_capacity(
    measure: Callable[[float], Awaitable[LevelResult]],
    mode: str,
    *,
    start: float = 1,
    max_value: float = 256,
    target: float = 0.9,
    tolerance: float = 0.1,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
) -> tuple[list[LevelResult], CapacityResult]:
    """Run the search with real levels. Returns every measured level, sorted by load."""
    if slo_ttft_ms is None and slo_tpot_ms is None:
        raise ValueError("find-capacity needs at least one SLO (TTFT or TPOT)")
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    integer = mode == "concurrency"
    if integer:
        start, max_value = max(1, int(start)), max(1, int(max_value))
    levels: list[LevelResult] = []

    async def probe(v: float) -> bool:
        level = await measure(v)
        levels.append(level)
        return passes(level, target)

    res = await search_capacity(
        probe, start=start, max_value=max_value, integer=integer, tolerance=tolerance
    )
    result = CapacityResult(
        mode=mode,
        target=target,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        capacity=res.capacity,
        first_fail=res.first_fail,
        reason=res.reason,
    )
    return sorted(levels, key=lambda lv: lv.value), result
