"""M2: load modes keep the right number of requests in flight and arrive at the right rate."""

import math
import statistics

import pytest

from servebench.client import RequestSpec
from servebench.load import level_runner, poisson_delays, run_level, run_sweep
from servebench.prompts import synthetic


def spec(server, n=5):
    return RequestSpec(url=server.url, model="mock-model", output_len=n)


@pytest.mark.parametrize("c", [1, 4])
async def test_concurrency_level_keeps_n_in_flight(server, c):
    server.cfg.ttft, server.cfg.itl = 0.05, 0.01
    lv = await run_level(spec(server), synthetic(3 * c, 16), "concurrency", c)
    assert server.stats.peak == c
    assert lv.summary.completed == 3 * c and lv.summary.failed == 0
    # 3 rounds of (50 ms + 4 * 10 ms) each, plus opening c new connections in the first round
    assert 3 * 0.09 - 0.02 < lv.duration < 3 * 0.09 + 0.2
    assert [r.index for r in lv.requests] == list(range(3 * c))


def test_poisson_delays_have_the_right_mean_and_are_seeded():
    d = poisson_delays(4000, rate=8.0, seed=1)
    assert statistics.mean(d) == pytest.approx(1 / 8, rel=0.05)
    assert d == poisson_delays(4000, rate=8.0, seed=1)
    assert d != poisson_delays(4000, rate=8.0, seed=2)
    assert poisson_delays(3, math.inf) == [0.0, 0.0, 0.0]


async def test_rate_level_arrivals(server):
    server.cfg.ttft, server.cfg.itl = 0.02, 0.005
    n, rate = 40, 50.0
    lv = await run_level(spec(server, 2), synthetic(n, 8), "rate", rate, seed=3)
    starts = sorted(r.start for r in lv.requests)
    expected = sum(poisson_delays(n, rate, seed=3))
    assert starts[-1] == pytest.approx(expected, abs=0.05)
    assert lv.summary.completed == n


async def test_rate_inf_sends_everything_at_once(server):
    server.cfg.ttft, server.cfg.itl = 0.2, 0.01
    lv = await run_level(spec(server, 2), synthetic(10, 8), "rate", math.inf)
    assert server.stats.peak == 10
    assert max(r.start for r in lv.requests) < 0.1


async def test_rate_with_max_concurrency(server):
    server.cfg.ttft, server.cfg.itl = 0.1, 0.01
    await run_level(spec(server, 2), synthetic(10, 8), "rate", math.inf, max_concurrency=3)
    assert server.stats.peak == 3


async def test_slowdown_shows_in_sweep(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.slowdown = 0.05, 0.01, 0.5
    levels = await run_sweep(
        spec(server),
        lambda n, s: synthetic(n, 16, s),
        "concurrency",
        [1, 4],
        num_requests=8,
        warmup_requests=1,
    )
    assert [lv.value for lv in levels] == [1, 4]
    t1, t4 = levels[0].summary.ttft.p50, levels[1].summary.ttft.p50
    assert t1 == pytest.approx(0.05, abs=0.03)
    assert t4 > 1.8 * t1  # with 4 in flight the mock server slows to ~2.5x
    assert levels[1].client_lag_p99_ms is None or levels[1].client_lag_p99_ms < 50


async def test_errors_are_counted(server):
    from servebench.prompts import Prompt

    prompts = [Prompt("fine"), Prompt("FAIL"), Prompt("fine"), Prompt("FAIL")]
    lv = await run_level(spec(server), prompts, "concurrency", 2)
    assert lv.summary.completed == 2 and lv.summary.failed == 2
    assert lv.summary.errors == {"HTTP 500": 2}


async def test_sweep_aborts_if_warmup_all_fails():
    s = RequestSpec(url="http://127.0.0.1:9/v1", model="x")
    with pytest.raises(RuntimeError, match="Is the server running"):
        await run_sweep(s, lambda n, _: synthetic(n, 4), "concurrency", [1], 2, timeout=2)


# --duration: levels run for a time, not a request count


async def test_concurrency_level_with_duration(server):
    server.cfg.ttft, server.cfg.itl = 0.05, 0.005
    from servebench.load import prompt_stream

    lv = await run_level(
        spec(server, 2),
        prompt_stream(lambda n, s: synthetic(n, 8, s), 0),
        "concurrency",
        2,
        duration=0.5,
    )
    assert server.stats.peak == 2
    assert all(r.start < 0.5 for r in lv.requests)  # nothing new starts after the deadline
    assert 0.5 <= lv.duration < 0.7  # in-flight requests finish
    assert 10 <= lv.summary.completed <= 25  # ~2 * 0.5 s / 55 ms each


async def test_rate_level_with_duration_matches_poisson_arrivals(server):
    server.cfg.ttft, server.cfg.itl = 0.01, 0.002
    from itertools import accumulate

    from servebench.load import prompt_stream

    lv = await run_level(
        spec(server, 2),
        prompt_stream(lambda n, s: synthetic(n, 8, s), 0),
        "rate",
        40.0,
        seed=5,
        duration=0.5,
    )
    expected = sum(1 for t in accumulate(poisson_delays(200, 40.0, seed=5)) if t < 0.5)
    assert lv.summary.completed == expected


async def test_duration_with_request_cap(server):
    server.cfg.ttft, server.cfg.itl = 0.01, 0.002
    measure = level_runner(
        spec(server, 2), lambda n, s: synthetic(n, 8, s), "concurrency", 6, duration=30
    )
    lv = await measure(2)
    assert lv.summary.completed == 6 and lv.duration < 5  # the cap ended it, not the clock


def test_prompt_stream_batches_are_distinct():
    from itertools import islice

    from servebench.load import prompt_stream

    got = list(islice(prompt_stream(lambda n, s: synthetic(n, 8, s), 0, batch=4), 12))
    assert len({p.text for p in got}) == 12


async def test_duration_rejects_infinite_rate():
    with pytest.raises(ValueError, match="finite rate"):
        await run_level(RequestSpec(url="http://x/v1", model="m"), [], "rate", math.inf, duration=1)
