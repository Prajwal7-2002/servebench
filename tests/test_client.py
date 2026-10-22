"""M1: per-request timings match the mock server's set delays."""

import pytest

from servebench.client import RequestSpec, send_request
from servebench.load import make_session
from servebench.prompts import Prompt

# Windows timers tick at ~15 ms, so single timings get that much slack; averages are tighter.
TOL = 0.03

PROMPT = Prompt("hello there general kenobi")


async def _one(server, **kw) -> object:
    spec = RequestSpec(url=server.url, model="mock-model", **kw)
    async with make_session(None, 30) as client:
        return await send_request(client, spec, PROMPT)


@pytest.mark.parametrize("endpoint", ["chat", "completions"])
async def test_ttft_and_tpot_match_server(server, endpoint):
    server.cfg.ttft, server.cfg.itl = 0.2, 0.03
    r = await _one(server, endpoint=endpoint, output_len=20)
    assert r.ok, r.error
    assert r.ttft == pytest.approx(0.2, abs=TOL)
    assert r.tpot == pytest.approx(0.03, abs=0.005)
    assert r.e2e == pytest.approx(0.2 + 19 * 0.03, abs=TOL)
    assert r.output_tokens == 20
    assert r.prompt_tokens == 4
    assert r.token_source == "usage"
    assert r.num_chunks == 20
    assert sum(r.itls) / len(r.itls) == pytest.approx(0.03, abs=0.005)


async def test_role_and_empty_chunks_do_not_count_for_ttft(server):
    # the server chat stream sends a role-only and an empty chunk right away; if they counted,
    # TTFT would be ~0 instead of the 0.3 s before the first real token
    server.cfg.ttft = 0.3
    r = await _one(server, output_len=3)
    assert r.ttft == pytest.approx(0.3, abs=TOL)


async def test_several_tokens_per_chunk(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.tokens_per_chunk = 0.1, 0.02, 4
    r = await _one(server, output_len=20)
    assert r.ok
    assert r.num_chunks == 5
    assert r.output_tokens == 20  # from usage, not the chunk count
    # 4 chunk gaps of 4 * 20 ms, spread over 19 tokens after the first
    assert r.tpot == pytest.approx(4 * 4 * 0.02 / 19, abs=0.004)
    assert sum(r.itls) / len(r.itls) == pytest.approx(0.08, abs=0.006)


async def test_falls_back_to_chunk_count_without_usage(server):
    server.cfg.send_usage = False
    r = await _one(server, output_len=7)
    assert r.token_source == "chunks"
    assert r.output_tokens == 7


async def test_falls_back_to_counter_without_usage(server):
    server.cfg.send_usage, server.cfg.tokens_per_chunk = False, 3
    spec = RequestSpec(url=server.url, model="mock-model", output_len=9)
    async with make_session(None, 30) as client:
        r = await send_request(client, spec, PROMPT, counter=lambda t: len(t.split()))
    assert r.token_source == "tokenizer"
    assert r.output_tokens == 9  # "tok " * 9 -> 9 words
    assert r.num_chunks == 3


async def test_http_error_is_recorded_not_raised(server):
    spec = RequestSpec(url=server.url, model="mock-model")
    async with make_session(None, 30) as client:
        r = await send_request(client, spec, Prompt("please FAIL now"))
    assert not r.ok
    assert r.error.startswith("HTTP 500")
    assert r.ttft is None


async def test_connection_error_is_recorded():
    spec = RequestSpec(url="http://127.0.0.1:9/v1", model="x")
    async with make_session(None, 2) as client:
        r = await send_request(client, spec, PROMPT)
    assert not r.ok
    assert r.error.startswith("connect error")


async def test_timeout_is_recorded(server):
    server.cfg.ttft = 2.0
    spec = RequestSpec(url=server.url, model="mock-model", output_len=2)
    async with make_session(None, 0.3) as client:
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


async def test_itl_median_matches_short_gaps(server):
    # 15 ms is just under the ~15.6 ms Windows timer tick: a coarse sleep in the server
    # would send tokens in pairs (gaps of ~31 ms and ~0 ms) and the median would collapse
    server.cfg.ttft, server.cfg.itl = 0.05, 0.015
    r = await _one(server, output_len=30)
    itls = sorted(r.itls)
    assert itls[len(itls) // 2] == pytest.approx(0.015, abs=0.003)


async def test_connections_are_reused(server):
    # breaking out of the stream at [DONE] used to close the connection, so every request
    # paid a new TCP connect (about 0.3 s each for `localhost` on Windows)
    server.cfg.ttft, server.cfg.itl = 0.01, 0.002
    spec = RequestSpec(url=server.url, model="mock-model", output_len=3)
    async with make_session(None, 30) as client:
        for i in range(4):
            r = await send_request(client, spec, PROMPT, index=i)
            assert r.ok, r.error
    assert server.stats.total == 4
    assert server.stats.connections == 1


# Broken streams must count as failures: a request that lost half its tokens and still
# counted as "ok" would make TPOT and throughput look better than the server really is.


async def test_connection_dropped_mid_stream_is_a_failure(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.drop_after = 0.01, 0.002, 3
    r = await _one(server, output_len=10)
    assert not r.ok
    assert r.error.startswith("connection dropped mid-stream")
    assert r.num_chunks == 3  # what did arrive is kept for inspection


async def test_stalled_stream_times_out(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.stall_after = 0.01, 0.002, 2
    spec = RequestSpec(url=server.url, model="mock-model", output_len=10)
    async with make_session(None, 0.5) as client:
        r = await send_request(client, spec, PROMPT)
    assert not r.ok
    assert r.error.startswith("timeout")


async def test_malformed_chunk_is_a_failure(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.malformed_after = 0.01, 0.002, 2
    r = await _one(server, output_len=5)
    assert not r.ok
    assert r.error.startswith("malformed chunk")


async def test_stream_without_done_or_finish_reason_is_incomplete(server):
    server.cfg.ttft, server.cfg.itl, server.cfg.no_done = 0.01, 0.002, True
    r = await _one(server, output_len=5)
    assert not r.ok
    assert r.error.startswith("incomplete stream")


async def test_finish_reason_without_done_is_fine(server):
    # some servers end with finish_reason and close the body without the [DONE] line
    from servebench.client import _parse_stream_end

    assert _parse_stream_end(done=False, finish_reason="stop") is None
    assert _parse_stream_end(done=True, finish_reason=None) is None
    assert _parse_stream_end(done=False, finish_reason=None).startswith("incomplete stream")


async def test_reasoning_tokens_count_as_generated(server):
    # reasoning models stream "thinking" in delta.reasoning_content before the answer;
    # those are generated tokens, so TTFT is the first of them and they count for TPOT
    server.cfg.ttft, server.cfg.itl, server.cfg.reasoning_tokens = 0.1, 0.02, 5
    r = await _one(server, output_len=10)
    assert r.ok, r.error
    assert r.num_chunks == 10 and r.reasoning_chunks == 5
    assert r.ttft == pytest.approx(0.1, abs=TOL)
    assert r.answer_start == pytest.approx(0.1 + 5 * 0.02, abs=TOL)
    assert r.output_tokens == 10


async def test_answer_start_equals_ttft_without_reasoning(server):
    r = await _one(server, output_len=4)
    assert r.reasoning_chunks == 0 and r.answer_start == r.ttft


def test_chunk_text_reads_both_reasoning_field_names():
    from servebench.client import _chunk_text

    def mk(delta):
        return {"choices": [{"delta": delta}]}

    assert _chunk_text(mk({"reasoning_content": "hm"}), "chat") == ("hm", True)
    assert _chunk_text(mk({"reasoning": "hm", "content": None}), "chat") == ("hm", True)
    assert _chunk_text(mk({"content": "hi"}), "chat") == ("hi", False)
    assert _chunk_text(mk({"role": "assistant", "content": ""}), "chat") == ("", False)
    assert _chunk_text({"choices": [{"text": "x"}]}, "completions") == ("x", False)
