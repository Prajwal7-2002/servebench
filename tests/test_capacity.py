"""find-capacity: the search logic on its own, then end to end against the mock server."""

import re

import pytest
from typer.testing import CliRunner

from servebench.capacity import find_capacity, search_capacity
from servebench.cli import app
from servebench.client import RequestSpec
from servebench.load import level_runner
from servebench.prompts import synthetic
from servebench.report import capacity_text, load_run


def threshold_probe(limit):
    """Passes up to `limit`, fails above it."""
    async def probe(v):
        return v <= limit
    return probe


@pytest.mark.parametrize("limit", [1, 4, 5, 37, 100, 255])
async def test_integer_search_finds_exact_limit(limit):
    res = await search_capacity(threshold_probe(limit), start=1, max_value=256, integer=True,
                                tolerance=0)
    assert res.reason == "converged"
    assert (res.capacity, res.first_fail) == (limit, limit + 1)


async def test_search_order_doubles_then_bisects():
    res = await search_capacity(threshold_probe(5), start=1, max_value=256, integer=True)
    assert [v for v, _ in res.tried] == [1, 2, 4, 8, 6, 5]


async def test_tolerance_stops_early_on_big_values():
    res = await search_capacity(threshold_probe(100), start=1, max_value=256, integer=True,
                                tolerance=0.1)
    assert res.capacity <= 100 < res.first_fail
    assert res.first_fail - res.capacity <= 0.1 * res.capacity
    assert len(res.tried) < 12


async def test_rate_search_is_continuous():
    res = await search_capacity(threshold_probe(3.3), start=0.5, max_value=64, integer=False,
                                tolerance=0.05)
    assert res.capacity <= 3.3 < res.first_fail
    assert res.first_fail - res.capacity <= 0.05 * res.capacity


async def test_start_fails():
    res = await search_capacity(threshold_probe(0), start=1, max_value=256, integer=True)
    assert (res.reason, res.capacity, res.first_fail) == ("start_failed", None, 1)


async def test_max_reached_never_goes_past_max():
    res = await search_capacity(threshold_probe(1000), start=1, max_value=48, integer=True)
    assert (res.reason, res.capacity, res.first_fail) == ("max_reached", 48, None)
    assert max(v for v, _ in res.tried) == 48


async def test_find_capacity_needs_an_slo():
    with pytest.raises(ValueError, match="SLO"):
        await find_capacity(None, "concurrency")


# Mock server: TTFT = 80 ms * in_flight (slowdown 1.0), so with a 360 ms TTFT SLO,
# 4 in flight gives 320 ms (pass) and 5 gives 400 ms (fail): 40 ms margin either way for
# connection setup and timer jitter (25 ms was not enough on Windows CI runners).
# Capacity should be exactly 4.
def _slow_server(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.slowdown = 0.08, 0.005, 1.0


async def test_find_capacity_on_mock_server(server):
    _slow_server(server)
    spec = RequestSpec(url=server.url, model="mock-model", output_len=2)
    measure = level_runner(spec, lambda n, s: synthetic(n, 8, s), "concurrency", None,
                           slo_ttft_ms=360, slo_tpot_ms=None)
    levels, cap = await find_capacity(measure, "concurrency", max_value=16, slo_ttft_ms=360)
    assert (cap.reason, cap.capacity, cap.first_fail) == ("converged", 4, 5)
    assert [lv.value for lv in levels] == [1, 2, 4, 5, 6, 8]  # sorted for charts
    # assert the pass rule (>= 90%), not 100%: one late request on a slow CI runner is fine
    assert levels[2].summary.slo_attainment >= 0.9
    assert levels[3].summary.slo_attainment < 0.9


def test_cli_find_capacity(server, tmp_path):
    _slow_server(server)
    out = tmp_path / "cap.json"
    res = CliRunner().invoke(app, [
        "find-capacity", "--url", server.url, "--model", "mock-model", "--max", "16",
        "--slo-ttft-ms", "360", "--slo-tpot-ms", "1000", "--output-len", "2",
        "--input-len", "8", "--warmup", "1", "--out", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert "-> pass" in res.output and "-> FAIL" in res.output
    run = load_run(out)
    assert run.capacity.capacity == 4
    text = capacity_text(run)
    assert re.match(r"Capacity: 4 concurrent requests\. At 4, \d+% met the SLO; at 5, only \d+%",
                    text), text
    assert text.endswith("SLO: TTFT < 360 ms and TPOT < 1000 ms for 90% of requests.")
    assert run.config.levels == [1, 2, 4, 5, 6, 8]


def test_cli_find_capacity_validates_args():
    res = CliRunner().invoke(app, ["find-capacity", "--url", "http://x/v1", "--model", "m",
                                   "--target", "1.5"])
    assert res.exit_code != 0
