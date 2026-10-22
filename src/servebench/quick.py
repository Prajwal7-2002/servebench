"""`servebench quick`: pick a server and model, then say what the numbers mean."""

from __future__ import annotations

import re

from . import detect
from .results import LevelResult, RunResult

# defaults for a ~1 minute measurement on a laptop
QUICK_INPUT_LEN = 256
QUICK_OUTPUT_LEN = 128
QUICK_REQUESTS_PER_LEVEL = 16
QUICK_MAX_USERS = 32


def pick_model(server: detect.Server) -> str | None:
    """The model to measure when the user didn't say: for Ollama the one already loaded
    (no load wait), else the first listed chat model (embedding models can't chat)."""
    if server.engine == "Ollama":
        ps = detect._get_json(f"{detect._root(server.url)}/api/ps", 2.0)
        if isinstance(ps, dict):
            loaded = [m.get("name") for m in ps.get("models", []) if m.get("name")]
            if loaded:
                return loaded[0]
    chat = [m for m in server.models if "embed" not in m.lower()]
    return chat[0] if chat else (server.models[0] if server.models else None)


def default_out(engine: str, model: str) -> str:
    def slug(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9.]+", "-", x).strip("-").lower()

    return f"servebench-quick-{slug(engine)}-{slug(model)}.json"


def _level(run: RunResult, value: float | None) -> LevelResult | None:
    return next((lv for lv in run.levels if lv.value == value), None)


def _per_user_speed(lv: LevelResult) -> str:
    tpot = lv.summary.tpot.p50
    return f"{1 / tpot:.0f} tokens/s" if tpot else "? tokens/s"


def _ms(x: float | None) -> str:
    return "?" if x is None else f"{x * 1000:.0f} ms"


def summary(run: RunResult) -> list[str]:
    """Plain-language result lines for people who don't speak TTFT/TPOT (yet)."""
    cap = run.capacity
    e = run.environment
    lines = []
    one = _level(run, 1)
    if one and one.summary.completed:
        lines.append(
            f"One user:    first token in {_ms(one.summary.ttft.p50)}, then {_per_user_speed(one)}"
        )
    if cap is None:
        return lines
    target = f"first token < {cap.slo_ttft_ms:g} ms" if cap.slo_ttft_ms else ""
    if cap.slo_tpot_ms:
        speed = f">= {1000 / cap.slo_tpot_ms:.0f} tokens/s each"
        target = f"{target} and {speed}" if target else speed
    share = f"{cap.target * 100:.0f}% of requests"

    if cap.reason == "start_failed":
        lines.append(f"Capacity:    even 1 user misses the target ({target} for {share}).")
        lines.append(
            "             Try a smaller model, or looser targets: "
            "--slo-ttft-ms 2000 --slo-tpot-ms 200"
        )
        return lines

    users = f"{cap.capacity:g} user" + ("" if cap.capacity == 1 else "s")
    prefix = "at least " if cap.reason == "max_reached" else ""
    lines.append(f"Capacity:    {prefix}{users} at once ({target}, for {share})")
    at = _level(run, cap.capacity)
    if at and at.summary.completed:
        s = at.summary
        lines.append(
            f"At {users}:".ljust(13) + f"{s.output_throughput:.0f} tokens/s in total, "
            f"{_per_user_speed(at)} each, first token p99 {_ms(s.ttft.p99)}"
        )

    # Why it stops there: if a request at the first failing load waits about as long as a
    # whole answer at capacity, it's queueing for a free slot, not getting slower to compute.
    fail = _level(run, cap.first_fail)
    if at and fail and fail.summary.ttft.p50 and at.summary.e2e.p50:
        if fail.summary.ttft.p50 > 0.5 * at.summary.e2e.p50:
            lines.append(
                f"Why:         above {users} requests queue: the server works on "
                f"about {cap.capacity:g} at once (first token p50 "
                f"{_ms(fail.summary.ttft.p50)} at {cap.first_fail:g})."
            )
            if e.engine == "Ollama" and cap.capacity <= 2:
                lines.append(
                    "Tip:         Ollama serves one request at a time unless told "
                    "otherwise. Set OLLAMA_NUM_PARALLEL=4, restart Ollama, run again."
                )
    return lines
