"""A fake OpenAI-compatible streaming server with known timings.

Used by the tests to check every metric against delays we set, and handy for trying
servebench without a model: `servebench fake-server --ttft-ms 200 --itl-ms 30`.
Needs `install the extra: servebench[fake]`.

Timing model for each request, with `active` = requests in flight when it starts:
    TTFT = ttft * (1 + slowdown * (active - 1))
    gap between chunks = itl * tokens_per_chunk * (1 + slowdown * (active - 1))
`slowdown > 0` imitates a server that gets slower under load, so sweeps have a shape.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


async def sleep_until(due: float) -> None:
    """Sleep until a perf_counter deadline, accurate to ~0.1 ms.

    asyncio.sleep on Windows wakes on a ~15.6 ms timer tick, which would make a 15 ms ITL
    come out as alternating 31 ms and 0 ms gaps. Sleep coarsely, then yield-spin the rest.
    """
    while (left := due - time.perf_counter()) > 0.02:
        await asyncio.sleep(left - 0.017)
    while time.perf_counter() < due:
        await asyncio.sleep(0)


@dataclass
class FakeConfig:
    ttft: float = 0.1  # seconds
    itl: float = 0.02  # seconds per token
    tokens_per_chunk: int = 1
    slowdown: float = 0.0
    send_usage: bool = True
    model: str = "fake-model"


@dataclass
class FakeStats:
    active: int = 0
    peak: int = 0
    total: int = 0

    def reset(self) -> None:
        self.active = self.peak = self.total = 0


def create_app(cfg: FakeConfig | None = None) -> FastAPI:
    app = FastAPI(title="servebench fake server")
    app.state.cfg = cfg or FakeConfig()
    app.state.stats = FakeStats()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": app.state.cfg.model, "object": "model"}]}

    @app.get("/stats")
    async def get_stats():
        s = app.state.stats
        return {"active": s.active, "peak": s.peak, "total": s.total}

    async def handle(request: Request, chat: bool):
        cfg: FakeConfig = app.state.cfg
        body = await request.json()
        text = (body["messages"][-1]["content"] if chat else body.get("prompt", "")) or ""
        if "FAIL" in text:  # lets tests produce server errors on demand
            return JSONResponse({"error": {"message": "forced failure"}}, status_code=500)
        n_out = int(body.get("max_tokens") or 16)
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        stats: FakeStats = app.state.stats
        stats.active += 1
        stats.total += 1
        stats.peak = max(stats.peak, stats.active)
        factor = 1 + cfg.slowdown * (stats.active - 1)
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(content: str | None, *, role: bool = False, finish: str | None = None) -> str:
            if chat:
                delta = {"role": "assistant", "content": ""} if role else {"content": content}
                choice = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": content or "", "finish_reason": finish}
            data = {"id": "fake", "object": obj, "created": created, "model": cfg.model,
                    "choices": [choice]}
            return f"data: {json.dumps(data)}\n\n"

        async def stream():
            start = time.perf_counter()
            try:
                if chat:
                    yield chunk(None, role=True)  # real servers send a role-only chunk first
                    yield chunk("")  # and sometimes an empty one; neither counts for TTFT
                sent = 0
                k = 0
                while sent < n_out:
                    m = min(cfg.tokens_per_chunk, n_out - sent)
                    # sleep to absolute deadlines so timer jitter does not add up
                    due = start + (cfg.ttft + k * cfg.itl * cfg.tokens_per_chunk) * factor
                    await sleep_until(due)
                    sent += m
                    k += 1
                    yield chunk("tok " * m, finish="length" if sent >= n_out else None)
                if want_usage and cfg.send_usage:
                    usage = {"prompt_tokens": len(text.split()), "completion_tokens": n_out,
                             "total_tokens": len(text.split()) + n_out}
                    data = {"id": "fake", "object": obj, "created": created, "model": cfg.model,
                            "choices": [], "usage": usage}
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                stats.active -= 1

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle(request, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request, chat=False)

    return app


def serve(cfg: FakeConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
