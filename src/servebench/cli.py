"""Command line: `servebench run`, `servebench report`, `servebench fake-server`."""

from __future__ import annotations

import asyncio
import math
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console

from . import __version__
from .client import RequestSpec
from .load import run_sweep
from .prompts import load_prompts
from .report import load_run, print_run, resummarize, save_run, write_report
from .results import Environment, RunConfig, RunResult
from .tokens import make_counter

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Load-test any OpenAI-compatible LLM endpoint.")
# keep the wide results table readable when output is piped or captured
console = Console(width=None if sys.stdout.isatty() else 160)


def parse_levels(text: str) -> list[float]:
    """'1,2,4' -> [1, 2, 4]; 'inf' allowed for rates."""
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        v = math.inf if part in ("inf", "infinity") else float(part)
        if v <= 0:
            raise typer.BadParameter(f"levels must be positive, got {part}")
        out.append(v)
    if not out:
        raise typer.BadParameter("give at least one level")
    return out


def _cpu() -> str | None:
    name = platform.processor() or None
    return f"{name}, {os.cpu_count()} logical CPUs" if name else f"{os.cpu_count()} logical CPUs"


def _server_models(url: str, api_key: str | None) -> list[str] | None:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.get(f"{url.rstrip('/')}/models", headers=headers, timeout=5)
        r.raise_for_status()
        return [m.get("id", "?") for m in r.json().get("data", [])]
    except Exception:
        return None


@app.command()
def run(
    url: Annotated[str, typer.Option(help="Base URL of the API, e.g. http://localhost:11434/v1")],
    model: Annotated[str, typer.Option(help="Model name as the server knows it")],
    concurrency: Annotated[str | None, typer.Option(
        help="Closed-loop sweep: comma-separated in-flight request counts, e.g. 1,2,4,8")] = None,
    rate: Annotated[str | None, typer.Option(
        help="Open-loop sweep: comma-separated Poisson arrival rates in req/s; 'inf' sends "
             "all at once")] = None,
    max_concurrency: Annotated[int | None, typer.Option(
        help="Cap on in-flight requests in rate mode")] = None,
    num_requests: Annotated[int | None, typer.Option(
        help="Requests per level (default: max(16, 4*N) for concurrency, 64 for rate)")] = None,
    input_len: Annotated[int, typer.Option(
        help="Prompt length in tokens (synthetic) or max words (dolly)")] = 256,
    output_len: Annotated[int, typer.Option(help="max_tokens per request")] = 128,
    prompts: Annotated[str, typer.Option(
        help="'synthetic', 'dolly' (downloads dolly-15k once), or a .jsonl/.txt file"),
    ] = "synthetic",
    endpoint: Annotated[str, typer.Option(help="'chat' or 'completions'")] = "chat",
    ignore_eos: Annotated[bool, typer.Option(
        help="Ask the server to always generate output-len tokens")] = True,
    temperature: Annotated[float, typer.Option()] = 0.0,
    warmup: Annotated[int, typer.Option(help="Warm-up requests, not measured")] = 2,
    seed: Annotated[int, typer.Option()] = 0,
    timeout: Annotated[float, typer.Option(help="Per-request timeout in seconds")] = 300.0,
    slo_ttft_ms: Annotated[float | None, typer.Option(help="Goodput SLO on TTFT")] = 500.0,
    slo_tpot_ms: Annotated[float | None, typer.Option(help="Goodput SLO on TPOT")] = 50.0,
    tokenizer: Annotated[str | None, typer.Option(
        help="Hugging Face tokenizer for exact prompt lengths and for counting tokens when "
             "the server sends no usage (needs servebench[hf])")] = None,
    api_key: Annotated[str | None, typer.Option(
        envvar="OPENAI_API_KEY", help="Bearer token, if the server needs one")] = None,
    label: Annotated[str | None, typer.Option(help="Name for this run in reports")] = None,
    notes: Annotated[str | None, typer.Option(
        help="Free text saved with the run, e.g. hardware or server flags")] = None,
    out: Annotated[Path | None, typer.Option(help="Write results to this run.json")] = None,
) -> None:
    """Benchmark an endpoint over a sweep of load levels."""
    if (concurrency is None) == (rate is None):
        raise typer.BadParameter("give exactly one of --concurrency or --rate")
    if endpoint not in ("chat", "completions"):
        raise typer.BadParameter("--endpoint must be 'chat' or 'completions'")
    mode = "concurrency" if concurrency is not None else "rate"
    levels = parse_levels(concurrency if concurrency is not None else rate)
    if mode == "concurrency" and any(math.isinf(v) or v != int(v) for v in levels):
        raise typer.BadParameter("--concurrency levels must be whole numbers")

    cfg = RunConfig(
        url=url, model=model, endpoint=endpoint, prompts=prompts, input_len=input_len,
        output_len=output_len, mode=mode, levels=levels, max_concurrency=max_concurrency,
        num_requests=num_requests, warmup=warmup, seed=seed, ignore_eos=ignore_eos,
        temperature=temperature, timeout=timeout, slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms, tokenizer=tokenizer,
    )
    env = Environment(
        servebench_version=__version__, python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}", client_cpu=_cpu(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        server_models=_server_models(url, api_key), notes=notes,
    )
    spec = RequestSpec(url=url, model=model, endpoint=endpoint, output_len=output_len,
                       ignore_eos=ignore_eos, temperature=temperature, api_key=api_key)
    counter = make_counter(tokenizer)

    def prompts_for(n: int, s: int):
        return load_prompts(prompts, n, input_len, s, tokenizer)

    def on_level(lv) -> None:
        sm = lv.summary
        ttft = f"{sm.ttft.p50 * 1000:.0f} ms" if sm.ttft.p50 is not None else "-"
        console.print(f"  {lv.label:>10}  {sm.completed} ok, {sm.failed} failed, "
                      f"{sm.output_throughput:.1f} tok/s, TTFT p50 {ttft}")

    console.print(f"servebench {__version__}: {mode} sweep {levels} against {url} ({model})")
    try:
        levels_out = asyncio.run(run_sweep(
            spec, prompts_for, mode, levels, num_requests, warmup_requests=warmup,
            timeout=timeout, seed=seed, max_concurrency=max_concurrency,
            slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms, counter=counter,
            on_level=on_level,
        ))
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    result = RunResult(label=label or (out.stem if out else model), config=cfg,
                       environment=env, levels=levels_out)
    print_run(result, console)
    if out:
        save_run(result, out)
        console.print(f"Saved {out}")


@app.command()
def report(
    runs: Annotated[list[Path], typer.Argument(help="One or more run.json files")],
    out: Annotated[Path | None, typer.Option(help="Write an .html or .md report")] = None,
    slo_ttft_ms: Annotated[float | None, typer.Option(
        help="Recompute goodput with this TTFT SLO")] = None,
    slo_tpot_ms: Annotated[float | None, typer.Option(
        help="Recompute goodput with this TPOT SLO")] = None,
) -> None:
    """Print and compare saved runs; optionally write an HTML/Markdown report with charts."""
    loaded = [load_run(p) for p in runs]
    if slo_ttft_ms is not None or slo_tpot_ms is not None:
        for r in loaded:
            resummarize(r, slo_ttft_ms if slo_ttft_ms is not None else r.config.slo_ttft_ms,
                        slo_tpot_ms if slo_tpot_ms is not None else r.config.slo_tpot_ms)
    for r in loaded:
        print_run(r, console)
    if out:
        try:
            write_report(loaded, out)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        console.print(f"Wrote {out}")


@app.command("fake-server")
def fake_server(
    port: int = 8000,
    host: str = "127.0.0.1",
    ttft_ms: float = 100.0,
    itl_ms: float = 20.0,
    tokens_per_chunk: int = 1,
    slowdown: Annotated[float, typer.Option(
        help="Extra latency per extra in-flight request, as a fraction (0.1 = +10%)")] = 0.0,
) -> None:
    """Run a fake OpenAI-compatible server with known timings (needs servebench[fake])."""
    try:
        from .fake_server import FakeConfig, serve
    except ImportError as e:
        console.print("[red]fake-server needs: pip install 'servebench[fake]'[/red]")
        raise typer.Exit(1) from e
    console.print(f"Fake server on http://{host}:{port}/v1 (model 'fake-model'), "
                  f"TTFT {ttft_ms} ms, ITL {itl_ms} ms. Ctrl+C to stop.")
    serve(FakeConfig(ttft=ttft_ms / 1000, itl=itl_ms / 1000, tokens_per_chunk=tokens_per_chunk,
                     slowdown=slowdown), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
