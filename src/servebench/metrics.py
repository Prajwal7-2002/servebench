"""Turn per-request timings into summary metrics."""

from __future__ import annotations

import math
from collections import Counter

from .results import RepeatMetric, RequestResult, Stats, Summary


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
    return Stats(
        mean=sum(a) / len(a),
        p50=percentile(a, 50),
        p90=percentile(a, 90),
        p99=percentile(a, 99),
        max=a[-1],
        count=len(a),
    )


# Student's t critical value for two-tailed 95% confidence interval
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def compute_repeat_metric(values: list[float]) -> RepeatMetric | None:
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return RepeatMetric(mean=mean, std=0.0, min=values[0], max=values[0], ci95=(mean, mean))
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    df = n - 1
    t_val = _T_CRIT_95.get(df, 1.96)
    ci = (mean - t_val * se, mean + t_val * se)
    return RepeatMetric(mean=mean, std=std, min=min(values), max=max(values), ci95=ci)


def summarize_repeats(summaries: list[Summary]) -> dict[str, RepeatMetric]:
    if not summaries:
        return {}
    metrics: dict[str, list[float]] = {
        "output_throughput": [s.output_throughput for s in summaries],
        "request_throughput": [s.request_throughput for s in summaries],
        "goodput": [s.goodput for s in summaries],
    }
    ttfts = [s.ttft.p50 for s in summaries if s.ttft.p50 is not None]
    if ttfts:
        metrics["ttft_p50"] = ttfts
    tpots = [s.tpot.p50 for s in summaries if s.tpot.p50 is not None]
    if tpots:
        metrics["tpot_p50"] = tpots
    e2es = [s.e2e.p50 for s in summaries if s.e2e.p50 is not None]
    if e2es:
        metrics["e2e_p50"] = e2es

    out = {}
    for k, vals in metrics.items():
        if m := compute_repeat_metric(vals):
            out[k] = m
    return out


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

    steady_req_tp = None
    steady_out_tp = None
    if len(ok) >= 8:
        # Exclude initial 20% of requests (at least 1, at most 16) as connection/model ramp-up
        ramp = min(max(1, len(ok) // 5), 16)
        steady_ok = ok[ramp:]
        t_start = steady_ok[0].start
        t_end = max((r.start + (r.e2e or 0.0)) for r in steady_ok)
        steady_dur = max(t_end - t_start, 1e-9)
        steady_req_tp = len(steady_ok) / steady_dur
        steady_out_tp = sum(r.output_tokens for r in steady_ok) / steady_dur

    return Summary(
        completed=len(ok),
        failed=len(failed),
        errors=dict(errors),
        duration=duration,
        request_throughput=len(ok) / dur,
        output_throughput=out_tokens / dur,
        input_throughput=(sum(prompt_tokens) / dur) if len(prompt_tokens) == len(ok) and ok
        else None,
        steady_state_request_throughput=steady_req_tp,
        steady_state_output_throughput=steady_out_tp,
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
