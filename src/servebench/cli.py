"""Command line: `servebench run`, `find-capacity`, `report`, `fake-server`."""

from __future__ import annotations

import asyncio
import math
import os
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console

from . import __version__
from .capacity import find_capacity, passes
from .client import RequestSpec
from .load import check_warmup, level_runner, run_sweep
from .prompts import Prompt, load_prompts
from .report import load_run, print_run, resummarize, save_run, write_report
from .results import Environment, LevelResult, RunConfig, RunResult
from .tokens import Counter, make_counter

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


# Options shared by `run` and `find-capacity`, defined once.
Url = Annotated[str, typer.Option(help="Base URL of the API, e.g. http://localhost:11434/v1")]
Model = Annotated[str, typer.Option(help="Model name as the server knows it")]
MaxConcurrency = Annotated[int | None, typer.Option(
    help="Cap on in-flight requests in rate mode")]
NumRequests = Annotated[int | None, typer.Option(
    help="Requests per level (default: max(16, 4*N) for concurrency, 64 for rate)")]
InputLen = Annotated[int, typer.Option(
    help="Prompt length in tokens (synthetic) or max words (dolly)")]
OutputLen = Annotated[int, typer.Option(help="max_tokens per request")]
Prompts = Annotated[str, typer.Option(
    help="'synthetic', 'dolly' (downloads dolly-15k once), or a .jsonl/.txt file")]
Endpoint = Annotated[str, typer.Option(help="'chat' or 'completions'")]
IgnoreEos = Annotated[bool, typer.Option(
    help="Ask the server to always generate output-len tokens")]
Temperature = Annotated[float, typer.Option()]
Warmup = Annotated[int, typer.Option(help="Warm-up requests, not measured")]
Seed = Annotated[int, typer.Option()]
Timeout = Annotated[float, typer.Option(help="Per-request timeout in seconds")]
SloTtft = Annotated[float | None, typer.Option(help="SLO on TTFT (goodput, capacity)")]
SloTpot = Annotated[float | None, typer.Option(help="SLO on TPOT (goodput, capacity)")]
Tokenizer = Annotated[str | None, typer.Option(
    help="Hugging Face tokenizer for exact prompt lengths and for counting tokens when "
         r"the server sends no usage (needs the servebench\[hf] extra)")]
ApiKey = Annotated[str | None, typer.Option(
    envvar="OPENAI_API_KEY", help="Bearer token, if the server needs one")]
Label = Annotated[str | None, typer.Option(help="Name for this run in reports")]
Notes = Annotated[str | None, typer.Option(
    help="Free text saved with the run, e.g. hardware or server flags")]
Out = Annotated[Path | None, typer.Option(help="Write results to this run.json")]


@dataclass
class Setup:
    """Everything a benchmark command needs, built from the shared options."""

    cfg: RunConfig
    env: Environment
    spec: RequestSpec
    counter: Counter | None
    prompts_for: Callable[[int, int], list[Prompt]]


def _setup(*, url, model, mode, levels, max_concurrency, num_requests, input_len, output_len,
           prompts, endpoint, ignore_eos, temperature, warmup, seed, timeout, slo_ttft_ms,
           slo_tpot_ms, tokenizer, api_key, notes) -> Setup:
    if endpoint not in ("chat", "completions"):
        raise typer.BadParameter("--endpoint must be 'chat' or 'completions'")
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

    def prompts_for(n: int, s: int) -> list[Prompt]:
        return load_prompts(prompts, n, input_len, s, tokenizer)

    return Setup(cfg, env, spec, make_counter(tokenizer), prompts_for)


def _print_level(lv: LevelResult, verdict: str = "") -> None:
    sm = lv.summary
    ttft = f"{sm.ttft.p50 * 1000:.0f} ms" if sm.ttft.p50 is not None else "-"
    in_slo = f", {sm.slo_attainment * 100:.0f}% in SLO" if sm.slo_attainment is not None else ""
    console.print(f"  {lv.label:>10}  {sm.completed} ok, {sm.failed} failed, "
                  f"{sm.output_throughput:.1f} tok/s, TTFT p50 {ttft}{in_slo}{verdict}")


def _finish(result: RunResult, out: Path | None) -> None:
    print_run(result, console)
    if out:
        save_run(result, out)
        console.print(f"Saved {out}")


@app.command()
def run(
    url: Url,
    model: Model,
    concurrency: Annotated[str | None, typer.Option(
        help="Closed-loop sweep: comma-separated in-flight request counts, e.g. 1,2,4,8")] = None,
    rate: Annotated[str | None, typer.Option(
        help="Open-loop sweep: comma-separated Poisson arrival rates in req/s; 'inf' sends "
             "all at once")] = None,
    max_concurrency: MaxConcurrency = None,
    num_requests: NumRequests = None,
    input_len: InputLen = 256,
    output_len: OutputLen = 128,
    prompts: Prompts = "synthetic",
    endpoint: Endpoint = "chat",
    ignore_eos: IgnoreEos = True,
    temperature: Temperature = 0.0,
    warmup: Warmup = 2,
    seed: Seed = 0,
    timeout: Timeout = 300.0,
    slo_ttft_ms: SloTtft = 500.0,
    slo_tpot_ms: SloTpot = 50.0,
    tokenizer: Tokenizer = None,
    api_key: ApiKey = None,
    label: Label = None,
    notes: Notes = None,
    out: Out = None,
) -> None:
    """Benchmark an endpoint over a sweep of load levels."""
    if (concurrency is None) == (rate is None):
        raise typer.BadParameter("give exactly one of --concurrency or --rate")
    mode = "concurrency" if concurrency is not None else "rate"
    levels = parse_levels(concurrency if concurrency is not None else rate)
    if mode == "concurrency" and any(math.isinf(v) or v != int(v) for v in levels):
        raise typer.BadParameter("--concurrency levels must be whole numbers")
    st = _setup(url=url, model=model, mode=mode, levels=levels, max_concurrency=max_concurrency,
                num_requests=num_requests, input_len=input_len, output_len=output_len,
                prompts=prompts, endpoint=endpoint, ignore_eos=ignore_eos,
                temperature=temperature, warmup=warmup, seed=seed, timeout=timeout,
                slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms, tokenizer=tokenizer,
                api_key=api_key, notes=notes)

    console.print(f"servebench {__version__}: {mode} sweep {levels} against {url} ({model})")
    try:
        levels_out = asyncio.run(run_sweep(
            st.spec, st.prompts_for, mode, levels, num_requests, warmup_requests=warmup,
            timeout=timeout, seed=seed, max_concurrency=max_concurrency,
            slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms, counter=st.counter,
            on_level=_print_level,
        ))
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    _finish(RunResult(label=label or (out.stem if out else model), config=st.cfg,
                      environment=st.env, levels=levels_out), out)


@app.command("find-capacity")
def find_capacity_cmd(
    url: Url,
    model: Model,
    mode: Annotated[str, typer.Option(
        help="'concurrency' (in-flight requests) or 'rate' (Poisson req/s)")] = "concurrency",
    start: Annotated[float, typer.Option(help="First load level to try")] = 1,
    max_load: Annotated[float, typer.Option("--max", help="Highest load level to try")] = 256,
    target: Annotated[float, typer.Option(
        help="Share of requests that must meet the SLOs for a level to pass")] = 0.9,
    tolerance: Annotated[float, typer.Option(
        help="Stop when the pass/fail gap is within this fraction of the passing load")] = 0.1,
    max_concurrency: MaxConcurrency = None,
    num_requests: NumRequests = None,
    input_len: InputLen = 256,
    output_len: OutputLen = 128,
    prompts: Prompts = "synthetic",
    endpoint: Endpoint = "chat",
    ignore_eos: IgnoreEos = True,
    temperature: Temperature = 0.0,
    warmup: Warmup = 2,
    seed: Seed = 0,
    timeout: Timeout = 300.0,
    slo_ttft_ms: SloTtft = 500.0,
    slo_tpot_ms: SloTpot = 50.0,
    tokenizer: Tokenizer = None,
    api_key: ApiKey = None,
    label: Label = None,
    notes: Notes = None,
    out: Out = None,
) -> None:
    """Find the highest load that still meets the SLOs (doubling, then binary search).

    A level passes when no request failed and at least --target of the requests met
    --slo-ttft-ms and --slo-tpot-ms. Every level tried is saved, so `report` can chart it.
    """
    if mode not in ("concurrency", "rate"):
        raise typer.BadParameter("--mode must be 'concurrency' or 'rate'")
    if not 0 < target <= 1:
        raise typer.BadParameter("--target must be between 0 and 1, e.g. 0.9")
    if slo_ttft_ms is None and slo_tpot_ms is None:
        raise typer.BadParameter("find-capacity needs --slo-ttft-ms and/or --slo-tpot-ms")
    if start <= 0 or max_load < start or math.isinf(max_load):
        raise typer.BadParameter("need 0 < --start <= --max, and a finite --max")
    st = _setup(url=url, model=model, mode=mode, levels=[], max_concurrency=max_concurrency,
                num_requests=num_requests, input_len=input_len, output_len=output_len,
                prompts=prompts, endpoint=endpoint, ignore_eos=ignore_eos,
                temperature=temperature, warmup=warmup, seed=seed, timeout=timeout,
                slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms, tokenizer=tokenizer,
                api_key=api_key, notes=notes)

    def on_level(lv: LevelResult) -> None:
        _print_level(lv, "  -> pass" if passes(lv, target) else "  -> FAIL")

    async def go():
        await check_warmup(st.spec, st.prompts_for, warmup, timeout, seed)
        measure = level_runner(
            st.spec, st.prompts_for, mode, num_requests, timeout=timeout, seed=seed,
            max_concurrency=max_concurrency, slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms,
            counter=st.counter, on_level=on_level,
        )
        return await find_capacity(measure, mode, start=start, max_value=max_load,
                                   target=target, tolerance=tolerance,
                                   slo_ttft_ms=slo_ttft_ms, slo_tpot_ms=slo_tpot_ms)

    console.print(f"servebench {__version__}: finding {mode} capacity against {url} ({model}); "
                  f"a level passes with no errors and {target * 100:.0f}% of requests in SLO")
    try:
        levels_out, cap = asyncio.run(go())
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    st.cfg.levels = [lv.value for lv in levels_out]
    _finish(RunResult(label=label or (out.stem if out else model), config=st.cfg,
                      environment=st.env, levels=levels_out, capacity=cap), out)


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
    """Run a fake OpenAI-compatible server with known timings (needs the fake extra)."""
    try:
        from .fake_server import FakeConfig, serve
    except ImportError as e:
        console.print(r"[red]fake-server needs the servebench\[fake] extra[/red]")
        raise typer.Exit(1) from e
    console.print(f"Fake server on http://{host}:{port}/v1 (model 'fake-model'), "
                  f"TTFT {ttft_ms} ms, ITL {itl_ms} ms. Ctrl+C to stop.")
    serve(FakeConfig(ttft=ttft_ms / 1000, itl=itl_ms / 1000, tokens_per_chunk=tokens_per_chunk,
                     slowdown=slowdown), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
