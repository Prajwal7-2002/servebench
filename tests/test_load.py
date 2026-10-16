"""M2: load modes keep the right number of requests in flight and arrive at the right rate."""

import math
import statistics

import pytest

from servebench.client import RequestSpec
from servebench.load import poisson_delays, run_level, run_sweep
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
    levels = await run_sweep(spec(server), lambda n, s: synthetic(n, 16, s), "concurrency",
                             [1, 4], num_requests=8, warmup_requests=1)
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
