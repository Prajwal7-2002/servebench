"""Turn per-request timings into summary metrics."""

from __future__ import annotations

from collections import Counter

from .results import RequestResult, Stats, Summary


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of already-sorted values (numpy's default method).

    Plain Python on purpose: numpy costs ~70 ms of import time on every command, and a
    level has at most a few thousand values.
    """
    if not sorted_values:
        raise ValueError("percentile of no values")
    pos = (len(sorted_values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def stats(values: list[float]) -> Stats:
    if not values:
        return Stats()
    a = sorted(values)
    return Stats(mean=sum(a) / len(a), p50=percentile(a, 50), p90=percentile(a, 90),
                 p99=percentile(a, 99), max=a[-1])


def meets_slo(r: RequestResult, slo_ttft_ms: float | None, slo_tpot_ms: float | None) -> bool:
    if not r.ok or r.ttft is None:
        return False
    if slo_ttft_ms is not None and r.ttft * 1000 > slo_ttft_ms:
        return False
    if slo_tpot_ms is not None:
        tpot = r.tpot
        # a one-token answer has no TPOT, so it can only fail on TTFT
        if tpot is not None and tpot * 1000 > slo_tpot_ms:
            return False
    return True


def summarize(
    requests: list[RequestResult],
    duration: float,
    slo_ttft_ms: float | None = None,
    slo_tpot_ms: float | None = None,
) -> Summary:
    ok = [r for r in requests if r.ok]
    failed = [r for r in requests if not r.ok]
    # group errors by kind, not by full message ("HTTP 500: ..." -> "HTTP 500")
    errors = Counter((r.error or "unknown").split(":")[0] for r in failed)
    dur = max(duration, 1e-9)
    out_tokens = sum(r.output_tokens for r in ok)
    prompt_tokens = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    chunks = sum(r.num_chunks for r in ok)
    good = sum(meets_slo(r, slo_ttft_ms, slo_tpot_ms) for r in requests)
    return Summary(
        completed=len(ok),
        failed=len(failed),
        errors=dict(errors),
        duration=duration,
        request_throughput=len(ok) / dur,
        output_throughput=out_tokens / dur,
        input_throughput=(sum(prompt_tokens) / dur) if len(prompt_tokens) == len(ok) and ok
        else None,
        goodput=good / dur,
        slo_attainment=(good / len(requests)) if requests else None,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        ttft=stats([r.ttft for r in ok if r.ttft is not None]),
        tpot=stats([t for r in ok if (t := r.tpot) is not None]),
        itl=stats([g for r in ok for g in r.itls]),
        e2e=stats([r.e2e for r in ok if r.e2e is not None]),
        output_tokens_mean=(out_tokens / len(ok)) if ok else None,
        tokens_per_chunk=(out_tokens / chunks) if chunks else None,
        token_sources=dict(Counter(r.token_source for r in ok)),
        reasoning_share=(sum(r.reasoning_chunks for r in ok) / chunks) if chunks else None,
        answer_start=stats([r.answer_start for r in ok if r.answer_start is not None]),
    )
