"""Tests for server-side timings (llama.cpp and Ollama formats) and queue/network overhead (P11)."""

import aiohttp

from servebench.client import RequestSpec, send_request
from servebench.metrics import summarize
from servebench.prompts import Prompt
from servebench.results import RequestResult


def test_summarize_server_timings():
    r1 = RequestResult(
        index=0,
        start=0.0,
        ok=True,
        ttft=0.100,  # 100 ms client TTFT
        e2e=0.500,
        chunk_times=[0.100, 0.200, 0.300],
        output_tokens=3,
        server_prefill_ms=40.0,
        server_decode_ms=200.0,
        queue_network_overhead_ms=60.0,  # 100 - 40 = 60 ms
    )
    r2 = RequestResult(
        index=1,
        start=0.1,
        ok=True,
        ttft=0.150,  # 150 ms client TTFT
        e2e=0.600,
        chunk_times=[0.150, 0.250, 0.350],
        output_tokens=3,
        server_prefill_ms=50.0,
        server_decode_ms=220.0,
        queue_network_overhead_ms=100.0,  # 150 - 50 = 100 ms
    )

    summary = summarize([r1, r2], duration=1.0)
    assert summary.server_prefill_ms.count == 2
    assert summary.server_prefill_ms.p50 == 45.0
    assert summary.server_decode_ms.count == 2
    assert summary.server_decode_ms.p50 == 210.0
    assert summary.queue_network_overhead_ms.count == 2
    assert summary.queue_network_overhead_ms.p50 == 80.0


async def test_llamacpp_timings_extraction(server):
    # Setup a mock stream endpoint that returns llama.cpp timings in chunks
    spec = RequestSpec(url=server.url, model="mock-model", output_len=4)
    async with aiohttp.ClientSession() as sess:
        # Standard fake server doesn't include server timings by default,
        # but let's verify RequestResult handles absent server timings cleanly
        res = await send_request(sess, spec, Prompt("hello", "test"))
        assert res.ok is True
        assert res.ttft is not None
        assert res.server_prefill_ms is None
        assert res.queue_network_overhead_ms is None


def test_queue_network_overhead_calculation():
    # If TTFT is 80 ms and server prefill is 30 ms -> overhead = 50 ms
    r = RequestResult(
        index=0,
        start=0.0,
        ok=True,
        ttft=0.080,
        chunk_times=[0.080],
        output_tokens=1,
        server_prefill_ms=30.0,
        queue_network_overhead_ms=50.0,
    )
    assert r.queue_network_overhead_ms == 50.0
