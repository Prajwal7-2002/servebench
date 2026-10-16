"""Async streaming client: sends one request and records when each content chunk arrives."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

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


def _chunk_text(chunk: dict, endpoint: str) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if endpoint == "chat":
        return (choice.get("delta") or {}).get("content") or ""
    return choice.get("text") or ""


async def send_request(
    client: httpx.AsyncClient,
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
    try:
        async with client.stream(
            "POST", spec.full_url, json=spec.payload(prompt), headers=spec.headers()
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:200]
                res.ok, res.error = False, f"HTTP {resp.status_code}: {body}"
                return res
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    res.ok, res.error = False, f"stream error: {str(chunk['error'])[:200]}"
                    return res
                if chunk.get("usage"):
                    usage = chunk["usage"]
                text = _chunk_text(chunk, spec.endpoint)
                # role-only and empty chunks do not count: TTFT is the first real content
                if text:
                    res.chunk_times.append(round(now - start, 6))  # us is plenty
                    pieces.append(text)
        res.e2e = time.perf_counter() - start
    except httpx.TimeoutException as e:
        res.ok, res.error = False, f"timeout ({type(e).__name__})"
        return res
    except httpx.HTTPError as e:
        res.ok, res.error = False, f"{type(e).__name__}: {e}"[:200]
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
