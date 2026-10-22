import math

import pytest

from servebench.metrics import meets_slo, stats, summarize
from servebench.results import RequestResult


def req(i, ttft, e2e, n, ok=True, error=None, chunks=None, source="usage"):
    return RequestResult(
        index=i,
        start=0,
        ok=ok,
        error=error,
        ttft=ttft if ok else None,
        e2e=e2e if ok else None,
        output_tokens=n if ok else 0,
        token_source=source,
        chunk_times=chunks
        if chunks is not None
        else ([ttft + k * (e2e - ttft) / max(n - 1, 1) for k in range(n)] if ok else []),
    )


def test_stats_percentiles():
    s = stats([float(x) for x in range(1, 101)])
    assert s.mean == pytest.approx(50.5)
    assert s.p50 == pytest.approx(50.5)
    assert s.p90 == pytest.approx(90.1)
    assert s.p99 == pytest.approx(99.01)
    assert s.max == 100
    assert stats([]).p50 is None


def test_tpot_and_itl():
    r = req(0, 0.1, 1.0, 10)
    assert r.tpot == pytest.approx(0.1)
    assert len(r.itls) == 9
    assert all(g == pytest.approx(0.1) for g in r.itls)
    assert req(0, 0.1, 0.1, 1).tpot is None  # one token has no TPOT


def test_summary_throughput_and_errors():
    reqs = [req(i, 0.1, 1.1, 11) for i in range(8)]
    reqs += [
        req(8, 0, 0, 0, ok=False, error="HTTP 500: boom"),
        req(9, 0, 0, 0, ok=False, error="timeout (ReadTimeout)"),
    ]
    s = summarize(reqs, duration=4.0)
    assert s.completed == 8 and s.failed == 2
    assert s.errors == {"HTTP 500": 1, "timeout (ReadTimeout)": 1}
    assert s.request_throughput == pytest.approx(2.0)
    assert s.output_throughput == pytest.approx(8 * 11 / 4)
    assert s.ttft.p50 == pytest.approx(0.1)
    assert s.tpot.p50 == pytest.approx(0.1)
    assert s.tokens_per_chunk == pytest.approx(1.0)
    assert s.token_sources == {"usage": 8}


def test_goodput_counts_only_requests_meeting_both_slos():
    reqs = [
        req(0, 0.2, 1.2, 11),  # ttft 200 ms, tpot 100 ms
        req(1, 0.6, 1.1, 11),  # ttft 600 ms -> fails TTFT
        req(2, 0.2, 0.7, 11),  # tpot 50 ms
        req(3, 0, 0, 0, ok=False, error="x"),
    ]
    assert summarize(reqs, 1.0, 500, None).goodput == pytest.approx(2)
    assert summarize(reqs, 1.0, 500, 60).goodput == pytest.approx(1)
    assert summarize(reqs, 1.0, None, None).goodput == pytest.approx(3)
    assert not meets_slo(reqs[3], None, None)


def test_input_throughput_only_when_every_request_has_prompt_tokens():
    a, b = req(0, 0.1, 1, 5), req(1, 0.1, 1, 5)
    a.prompt_tokens = 100
    assert summarize([a, b], 1.0).input_throughput is None
    b.prompt_tokens = 300
    assert summarize([a, b], 2.0).input_throughput == pytest.approx(200)


def test_empty_level():
    s = summarize([], 0.0)
    assert s.completed == 0 and s.output_throughput == 0
    assert not math.isnan(s.request_throughput)


def test_new_error_kinds_group_cleanly():
    errs = [
        "connection dropped mid-stream: RemoteProtocolError peer closed",
        "malformed chunk: '{bad'",
        "incomplete stream: ended without [DONE]",
        "connection dropped mid-stream: ReadError x",
    ]
    reqs = [req(i, 0, 0, 0, ok=False, error=e) for i, e in enumerate(errs)]
    assert summarize(reqs, 1.0).errors == {
        "connection dropped mid-stream": 2,
        "malformed chunk": 1,
        "incomplete stream": 1,
    }


def test_percentile_matches_numpy_linear_method():
    import random

    import numpy as np

    from servebench.metrics import percentile

    rng = random.Random(0)
    for n in (1, 2, 3, 10, 101, 1000):
        vals = sorted(rng.random() for _ in range(n))
        for q in (0, 1, 50, 90, 99, 100):
            assert percentile(vals, q) == pytest.approx(float(np.percentile(vals, q)))


def test_cli_startup_does_not_import_heavy_libraries():
    # `servebench --help` / `run` should start fast: charts load matplotlib only when drawn
    import subprocess
    import sys

    code = (
        "import sys, servebench.cli; "
        "print(','.join(m for m in ('matplotlib', 'numpy') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""
