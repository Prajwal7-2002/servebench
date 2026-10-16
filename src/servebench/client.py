"""Async streaming client: sends one request and records when each content chunk arrives.

Uses aiohttp (as vLLM's and SGLang's serving benchmarks do). httpx was replaced after
profiling: its connection pool re-checks every connection, with a select() call each, on
every request, so client CPU grew with the square of concurrency and the client, not the
server, became the bottleneck at ~60 req/s. aiohttp has a C HTTP parser and an O(1) pool.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp

from .prompts import Prompt
from .results import RequestResult
from .tokens import Counter


@dataclass
class RequestSpec:
    url: str  # base URL, e.g. http://localhost:11434/v1
    model: str
    endpoint: str = "chat"  # "chat" or "completions"
    output_len: int = 128
    ignore_eos: bool = True
    temperature: float = 0.0
    api_key: str | None = None

    @property
    def full_url(self) -> str:
        path = "chat/completions" if self.endpoint == "chat" else "completions"
        return f"{self.url.rstrip('/')}/{path}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def payload(self, prompt: Prompt) -> dict:
        body: dict = {
            "model": self.model,
            "max_tokens": self.output_len,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.endpoint == "chat":
            body["messages"] = [{"role": "user", "content": prompt.text}]
        else:
            body["prompt"] = prompt.text
        if self.ignore_eos:
            # vLLM, SGLang and llama.cpp honour this; servers that don't know it ignore it
            body["ignore_eos"] = True
        return body


# Field names engines use for streamed reasoning ("thinking") tokens: vLLM/SGLang/DeepSeek
# use reasoning_content, some newer servers use reasoning.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _chunk_text(chunk: dict, endpoint: str) -> tuple[str, bool]:
    """(text, is_reasoning) for the first choice of a streamed chunk."""
    choices = chunk.get("choices") or []
    if not choices:
        return "", False
    choice = choices[0]
    if endpoint != "chat":
        return choice.get("text") or "", False
    delta = choice.get("delta") or {}
    if delta.get("content"):
        return delta["content"], False
    for field in _REASONING_FIELDS:
        if delta.get(field):
            return delta[field], True
    return "", False


def _finish_reason(chunk: dict) -> str | None:
    choices = chunk.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def _parse_stream_end(*, done: bool, finish_reason: str | None) -> str | None:
    """None if the stream ended properly, else the error to record.

    Proper = a `[DONE]` line, or a finish_reason (some servers close without `[DONE]`).
    A body that just stops has lost tokens we can't count, so it must not pass as a success.
    """
    if done or finish_reason:
        return None
    return "incomplete stream: ended without [DONE] or a finish_reason"


async def _lines(resp: aiohttp.ClientResponse) -> AsyncIterator[str]:
    """Lines of the body as they arrive. Own splitting instead of aiohttp's readline, which
    has a per-line size limit: one big SSE chunk must not fail a request."""
    buf = b""
    async for data in resp.content.iter_any():
        buf += data
        while (i := buf.find(b"\n")) >= 0:
            line, buf = buf[:i], buf[i + 1:]
            yield line.decode("utf-8", errors="replace").rstrip("\r")
    if buf:
        yield buf.decode("utf-8", errors="replace").rstrip("\r")


async def send_request(
    session: aiohttp.ClientSession,
    spec: RequestSpec,
    prompt: Prompt,
    *,
    index: int = 0,
    t0: float | None = None,
    counter: Counter | None = None,
) -> RequestResult:
    """Send one streaming request. Never raises for server/network errors; they go in `error`."""
    start = time.perf_counter()
    res = RequestResult(index=index, start=start - (t0 if t0 is not None else start))
    pieces: list[str] = []
    usage: dict | None = None
    finish_reason: str | None = None
    done_at: float | None = None
    streaming = False  # True once a 200 response started: errors after this are mid-stream
    try:
        async with session.post(spec.full_url, json=spec.payload(prompt),
                                headers=spec.headers()) as resp:
            if resp.status != 200:
                body = (await resp.read()).decode(errors="replace")[:200]
                res.ok, res.error = False, f"HTTP {resp.status}: {body}"
                return res
            streaming = True
            # Read to the very end of the body, even after [DONE]: leaving the stream early
            # closes the connection instead of returning it to the pool, so every request
            # would pay a new TCP connect.
            async for line in _lines(resp):
                if done_at is not None or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_at = time.perf_counter()
                    continue
                now = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # a lost chunk may be lost tokens: fail rather than undercount quietly
                    res.ok, res.error = False, f"malformed chunk: {data[:80]!r}"
                    return res
                if "error" in chunk:
                    res.ok, res.error = False, f"stream error: {str(chunk['error'])[:200]}"
                    return res
                if chunk.get("usage"):
                    usage = chunk["usage"]
                finish_reason = _finish_reason(chunk) or finish_reason
                text, reasoning = _chunk_text(chunk, spec.endpoint)
                # role-only and empty chunks do not count: TTFT is the first generated token,
                # reasoning included (the model is generating it); answer_start marks the
                # first token of the actual answer
                if text:
                    t = round(now - start, 6)  # us is plenty
                    res.chunk_times.append(t)
                    pieces.append(text)
                    if reasoning:
                        res.reasoning_chunks += 1
                    elif res.answer_start is None:
                        res.answer_start = t
        res.e2e = (done_at if done_at is not None else time.perf_counter()) - start
    except (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError) as e:
        # refused, or (often on Windows) no answer before the connect timeout: either way
        # the server isn't reachable, which is a different problem from a slow response
        res.ok, res.error = False, f"connect error: {type(e).__name__} {e}"[:200]
        return res
    except TimeoutError as e:  # includes aiohttp's read timeouts
        res.ok, res.error = False, f"timeout ({type(e).__name__})"
        return res
    except (aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError,
            aiohttp.ClientOSError) as e:
        where = "connection dropped mid-stream" if streaming else "connection error"
        res.ok, res.error = False, f"{where}: {type(e).__name__} {e}"[:200]
        return res
    except aiohttp.ClientError as e:
        res.ok, res.error = False, f"{type(e).__name__}: {e}"[:200]
        return res

    if err := _parse_stream_end(done=done_at is not None, finish_reason=finish_reason):
        res.ok, res.error = False, err
        return res

    if not res.chunk_times:
        res.ok, res.error = False, "empty response"
        return res
    res.ttft = res.chunk_times[0]
    if usage and usage.get("completion_tokens"):
        res.output_tokens = int(usage["completion_tokens"])
        res.prompt_tokens = usage.get("prompt_tokens")
        res.token_source = "usage"
    elif counter is not None:
        res.output_tokens = counter("".join(pieces))
        res.prompt_tokens = counter(prompt.text)
        res.token_source = "tokenizer"
    else:
        res.output_tokens = len(res.chunk_times)
        res.token_source = "chunks"
    return res
