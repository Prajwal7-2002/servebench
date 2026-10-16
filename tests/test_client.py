"""M1: per-request timings match the fake server's set delays."""

import httpx
import pytest

from servebench.client import RequestSpec, send_request
from servebench.prompts import Prompt

# Windows timers tick at ~15 ms, so single timings get that much slack; averages are tighter.
TOL = 0.03

PROMPT = Prompt("hello there general kenobi")


async def _one(fake, **kw) -> object:
    spec = RequestSpec(url=fake.url, model="fake-model", **kw)
    async with httpx.AsyncClient(timeout=30) as client:
        return await send_request(client, spec, PROMPT)


@pytest.mark.parametrize("endpoint", ["chat", "completions"])
async def test_ttft_and_tpot_match_server(fake, endpoint):
    fake.cfg.ttft, fake.cfg.itl = 0.2, 0.03
    r = await _one(fake, endpoint=endpoint, output_len=20)
    assert r.ok, r.error
    assert r.ttft == pytest.approx(0.2, abs=TOL)
    assert r.tpot == pytest.approx(0.03, abs=0.005)
    assert r.e2e == pytest.approx(0.2 + 19 * 0.03, abs=TOL)
    assert r.output_tokens == 20
    assert r.prompt_tokens == 4
    assert r.token_source == "usage"
    assert r.num_chunks == 20
    assert sum(r.itls) / len(r.itls) == pytest.approx(0.03, abs=0.005)


async def test_role_and_empty_chunks_do_not_count_for_ttft(fake):
    # the fake chat stream sends a role-only and an empty chunk right away; if they counted,
    # TTFT would be ~0 instead of the 0.3 s before the first real token
    fake.cfg.ttft = 0.3
    r = await _one(fake, output_len=3)
    assert r.ttft == pytest.approx(0.3, abs=TOL)


async def test_several_tokens_per_chunk(fake):
    fake.cfg.ttft, fake.cfg.itl, fake.cfg.tokens_per_chunk = 0.1, 0.02, 4
    r = await _one(fake, output_len=20)
    assert r.ok
    assert r.num_chunks == 5
    assert r.output_tokens == 20  # from usage, not the chunk count
    # 4 chunk gaps of 4 * 20 ms, spread over 19 tokens after the first
    assert r.tpot == pytest.approx(4 * 4 * 0.02 / 19, abs=0.004)
    assert sum(r.itls) / len(r.itls) == pytest.approx(0.08, abs=0.006)


async def test_falls_back_to_chunk_count_without_usage(fake):
    fake.cfg.send_usage = False
    r = await _one(fake, output_len=7)
    assert r.token_source == "chunks"
    assert r.output_tokens == 7


async def test_falls_back_to_counter_without_usage(fake):
    fake.cfg.send_usage, fake.cfg.tokens_per_chunk = False, 3
    spec = RequestSpec(url=fake.url, model="fake-model", output_len=9)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await send_request(client, spec, PROMPT, counter=lambda t: len(t.split()))
    assert r.token_source == "tokenizer"
    assert r.output_tokens == 9  # "tok " * 9 -> 9 words
    assert r.num_chunks == 3


async def test_http_error_is_recorded_not_raised(fake):
    spec = RequestSpec(url=fake.url, model="fake-model")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await send_request(client, spec, Prompt("please FAIL now"))
    assert not r.ok
    assert r.error.startswith("HTTP 500")
    assert r.ttft is None


async def test_connection_error_is_recorded():
    spec = RequestSpec(url="http://127.0.0.1:9/v1", model="x")
    async with httpx.AsyncClient(timeout=2) as client:
        r = await send_request(client, spec, PROMPT)
    assert not r.ok
    assert "Connect" in r.error


async def test_timeout_is_recorded(fake):
    fake.cfg.ttft = 2.0
    spec = RequestSpec(url=fake.url, model="fake-model", output_len=2)
    async with httpx.AsyncClient(timeout=0.3) as client:
        r = await send_request(client, spec, PROMPT)
    assert not r.ok
    assert r.error.startswith("timeout")


def test_payload_shapes():
    chat = RequestSpec(url="http://x/v1/", model="m", output_len=5)
    body = chat.payload(PROMPT)
    assert chat.full_url == "http://x/v1/chat/completions"
    assert body["messages"][0]["content"] == PROMPT.text
    assert body["stream_options"] == {"include_usage": True}
    assert body["ignore_eos"] is True and body["max_tokens"] == 5
    comp = RequestSpec(url="http://x/v1", model="m", endpoint="completions", ignore_eos=False)
    body = comp.payload(PROMPT)
    assert comp.full_url == "http://x/v1/completions"
    assert body["prompt"] == PROMPT.text and "ignore_eos" not in body


async def test_itl_median_matches_short_gaps(fake):
    # 15 ms is just under the ~15.6 ms Windows timer tick: a coarse sleep in the server
    # would send tokens in pairs (gaps of ~31 ms and ~0 ms) and the median would collapse
    fake.cfg.ttft, fake.cfg.itl = 0.05, 0.015
    r = await _one(fake, output_len=30)
    itls = sorted(r.itls)
    assert itls[len(itls) // 2] == pytest.approx(0.015, abs=0.003)


async def test_connections_are_reused(fake):
    # breaking out of the stream at [DONE] used to close the connection, so every request
    # paid a new TCP connect (about 0.3 s each for `localhost` on Windows)
    fake.cfg.ttft, fake.cfg.itl = 0.01, 0.002
    spec = RequestSpec(url=fake.url, model="fake-model", output_len=3)
    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(4):
            r = await send_request(client, spec, PROMPT, index=i)
            assert r.ok, r.error
    assert fake.stats.total == 4
    assert len(fake.stats.peers) == 1
