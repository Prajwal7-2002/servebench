"""Command line: `servebench run`, `find-capacity`, `report`, `mock-server`."""

from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import random
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, detect, system
from . import quick as quick_mod
from .capacity import find_capacity, passes
from .check import CheckThresholds, evaluate_regression, print_check_report
from .client import RequestSpec, send_request
from .diagnosis import diagnose_run
from .load import check_warmup, level_runner, make_session, run_sweep
from .prometheus import resolve_metrics_url
from .prompts import Prompt, load_prompts
from .report import (
    RunFileError,
    load_run,
    print_run,
    resummarize,
    save_run,
    write_csv,
    write_report,
    write_requests_csv,
)
from .results import Environment, LevelResult, RunConfig, RunResult
from .roofline import compute_roofline, render_ascii_roofline
from .tokens import Counter, make_counter
from .top import run_top_monitor
from .tune import evaluate_tuning
from .workloads import PRESETS, generate_workload_requests, load_trace_file

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Load-test any OpenAI-compatible LLM endpoint."
)
# keep the wide results table readable when output is piped or captured
console = Console(width=None if sys.stdout.isatty() else 160)


def parse_levels(text: str) -> list[float]:
    """'1,2,4' -> [1, 2, 4]; 'inf' allowed for rates."""
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("inf", "infinity"):
            v = math.inf
        else:
            try:
                v = float(part)
            except ValueError:
                raise typer.BadParameter(f"expected numbers or 'inf', got '{part}'") from None
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
    """Model ids from GET /models, or None if the server doesn't answer (it's optional)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(f"{url.rstrip('/')}/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return [m.get("id", "?") for m in json.load(r).get("data", [])]
    except Exception:
        return None


# Options shared by `run` and `find-capacity`, defined once.
Url = Annotated[str, typer.Option(help="Base URL of the API, e.g. http://localhost:11434/v1")]
Model = Annotated[str, typer.Option(help="Model name as the server knows it")]
MaxConcurrency = Annotated[int | None, typer.Option(help="Cap on in-flight requests in rate mode")]
NumRequests = Annotated[
    int | None,
    typer.Option(
        help="Requests per level (default: max(16, 4*N) for concurrency, 64 for rate); with "
        "--duration, an optional cap"
    ),
]
Duration = Annotated[
    float | None,
    typer.Option(
        help="Run each level for this many seconds instead of a fixed number of requests "
        "(in-flight requests then finish)"
    ),
]
InputLen = Annotated[
    int, typer.Option(help="Prompt length in tokens (synthetic) or max words (dolly)")
]
OutputLen = Annotated[int, typer.Option(help="max_tokens per request")]
Prompts = Annotated[
    str, typer.Option(help="'synthetic', 'dolly' (downloads dolly-15k once), or a .jsonl/.txt file")
]
Endpoint = Annotated[str, typer.Option(help="'chat' or 'completions'")]
IgnoreEos = Annotated[
    bool, typer.Option(help="Ask the server to always generate output-len tokens")
]
Temperature = Annotated[float, typer.Option()]
Warmup = Annotated[int, typer.Option(help="Warm-up requests, not measured")]
Seed = Annotated[
    int | None,
    typer.Option(
        help="Prompt seed; default: a new random one per run (printed and saved), so a "
        "rerun doesn't hit the server's prompt cache. Pass it to repeat a run exactly"
    ),
]
Timeout = Annotated[float, typer.Option(help="Per-request timeout in seconds")]
SloTtft = Annotated[float | None, typer.Option(help="SLO on TTFT (goodput, capacity)")]
SloTpot = Annotated[float | None, typer.Option(help="SLO on TPOT (goodput, capacity)")]
Tokenizer = Annotated[
    str | None,
    typer.Option(
        help="Hugging Face tokenizer for exact prompt lengths and for counting tokens when "
        r"the server sends no usage (needs the servebench\[hf] extra)"
    ),
]
ApiKey = Annotated[
    str | None, typer.Option(envvar="OPENAI_API_KEY", help="Bearer token, if the server needs one")
]
Label = Annotated[str | None, typer.Option(help="Name for this run in reports")]
Notes = Annotated[
    str | None, typer.Option(help="Free text saved with the run, e.g. hardware or server flags")
]
Out = Annotated[Path | None, typer.Option(help="Write results to this run.json")]


@dataclass
class Setup:
    """Everything a benchmark command needs, built from the shared options."""

    cfg: RunConfig
    env: Environment
    spec: RequestSpec
    counter: Counter | None
    prompts_for: Callable[[int, int], list[Prompt]]
    server: detect.Server | None = None
    metrics_url: str | None = None

    def after_run(self) -> None:
        """Ask the server about the model now that it's loaded (e.g. Ollama: on GPU?)."""
        if self.server:
            self.env.model_info = detect.model_info(self.server, self.cfg.model)


def load_config_file(path: Path) -> dict:
    if not path.is_file():
        raise typer.BadParameter(f"config file '{path}' does not exist")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(text)
        except Exception as e:
            raise typer.BadParameter(f"error parsing YAML config '{path}': {e}") from e
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except Exception as e:
            raise typer.BadParameter(f"error parsing JSON config '{path}': {e}") from e
    elif suffix == ".toml":
        import tomllib

        try:
            data = tomllib.loads(text)
        except Exception as e:
            raise typer.BadParameter(f"error parsing TOML config '{path}': {e}") from e
    else:
        try:
            import yaml

            data = yaml.safe_load(text)
        except Exception:
            try:
                data = json.loads(text)
            except Exception as e:
                raise typer.BadParameter(
                    f"could not parse config '{path}' as YAML or JSON: {e}"
                ) from e
    if not isinstance(data, dict):
        raise typer.BadParameter(f"config file '{path}' must define a dictionary")
    return data


def _setup(
    *,
    url,
    model,
    mode,
    levels,
    max_concurrency,
    num_requests,
    duration,
    min_duration=None,
    repeat=1,
    input_len,
    output_len,
    prompts,
    endpoint,
    ignore_eos,
    temperature,
    warmup,
    seed,
    timeout,
    slo_ttft_ms,
    slo_tpot_ms,
    tokenizer,
    api_key,
    notes,
    preset=None,
    input_dist=None,
    output_dist=None,
    arrival_dist="poisson",
    burstiness=1.0,
    shared_prefix_tokens=None,
    trace=None,
    gpu_telemetry=True,
    scrape_metrics=None,
) -> Setup:
    if endpoint not in ("chat", "completions"):
        raise typer.BadParameter("--endpoint must be 'chat' or 'completions'")
    if duration is not None and duration <= 0:
        raise typer.BadParameter("--duration must be a positive number of seconds")
    if min_duration is not None and min_duration <= 0:
        raise typer.BadParameter("--min-duration must be a positive number of seconds")
    if repeat < 1:
        raise typer.BadParameter("--repeat must be at least 1")
    if seed is None:
        # Fresh prompts each run: with a fixed default seed, a second run sends the same
        # prompts and servers with prompt/prefix caching (Ollama, llama.cpp, vLLM) skip
        # prefill. Measured: TTFT 43 ms fresh vs 20 ms repeated on Ollama.
        seed = random.SystemRandom().randrange(1, 2**31)
        console.print(f"[dim]seed {seed} (--seed {seed} repeats these exact prompts)[/dim]")

    if preset:
        preset_key = preset.lower()
        if preset_key not in PRESETS:
            raise typer.BadParameter(
                f"--preset must be one of {list(PRESETS.keys())}, got {preset!r}"
            )
        p_info = PRESETS[preset_key]
        if input_len == 256:  # default was not explicitly changed
            input_len = p_info["input_len"]
        if output_len == 128:  # default was not explicitly changed
            output_len = p_info["output_len"]

    cfg = RunConfig(
        url=url,
        model=model,
        endpoint=endpoint,
        prompts=prompts,
        input_len=input_len,
        output_len=output_len,
        mode=mode,
        levels=levels,
        max_concurrency=max_concurrency,
        num_requests=num_requests,
        duration=duration,
        min_duration=min_duration,
        repeat=repeat,
        warmup=warmup,
        seed=seed,
        ignore_eos=ignore_eos,
        temperature=temperature,
        timeout=timeout,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        tokenizer=tokenizer,
        preset=preset,
        input_dist=input_dist,
        output_dist=output_dist,
        arrival_dist=arrival_dist,
        burstiness=burstiness,
        shared_prefix_tokens=shared_prefix_tokens,
        trace=str(trace) if trace else None,
        gpu_telemetry=gpu_telemetry,
        scrape_metrics=scrape_metrics,
    )
    server = detect.identify(url)
    host = urlparse(url).hostname or ""
    env = Environment(
        servebench_version=__version__,
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        client_cpu=_cpu(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        server_models=_server_models(url, api_key),
        notes=notes,
        local_server=host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"),
        engine=server.engine if server else None,
        engine_version=server.version if server else None,
        **system.describe_for_env(),
    )
    spec = RequestSpec(
        url=url,
        model=model,
        endpoint=endpoint,
        output_len=output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
        api_key=api_key,
    )

    if urlparse(url).hostname == "localhost":
        console.print(
            "[yellow]! 'localhost' may resolve to IPv6 ::1 first; if the server only "
            "listens on IPv4 (Ollama does), every new connection waits ~0.3 s on "
            "Windows and that lands in TTFT. Use 127.0.0.1 instead.[/yellow]"
        )
    if env.server_models and model not in env.server_models:
        # a warning, not an error: some servers accept aliases they do not list
        console.print(
            f"[yellow]! The server does not list model '{model}'. It has: "
            f"{', '.join(env.server_models)}. For Ollama: ollama pull {model}[/yellow]"
        )

    if trace:

        def prompts_for(n: int, s: int) -> list[Prompt]:
            return load_trace_file(trace, n)
    elif input_dist or output_dist or shared_prefix_tokens:

        def prompts_for(n: int, s: int) -> list[Prompt]:
            return generate_workload_requests(
                n,
                input_len,
                output_len,
                input_dist=input_dist,
                output_dist=output_dist,
                shared_prefix_tokens=shared_prefix_tokens,
                seed=s,
                tokenizer=tokenizer,
            )
    else:

        def prompts_for(n: int, s: int) -> list[Prompt]:
            return load_prompts(prompts, n, input_len, s, tokenizer)

    metrics_url = resolve_metrics_url(url, scrape_metrics)
    return Setup(
        cfg, env, spec, make_counter(tokenizer), prompts_for, server, metrics_url=metrics_url
    )


def _print_level(lv: LevelResult, verdict: str = "") -> None:
    sm = lv.summary
    ttft = f"{sm.ttft.p50 * 1000:.0f} ms" if sm.ttft.p50 is not None else "-"
    in_slo = f", {sm.slo_attainment * 100:.0f}% in SLO" if sm.slo_attainment is not None else ""
    console.print(
        f"  {lv.label:>10}  {sm.completed} ok, {sm.failed} failed, "
        f"{sm.output_throughput:.1f} tok/s, TTFT p50 {ttft}{in_slo}{verdict}"
    )


def _finish(
    result: RunResult,
    out: Path | None,
    csv_path: Path | None = None,
    csv_requests_path: Path | None = None,
) -> None:
    print_run(result, console)
    if out:
        save_run(result, out)
        console.print(f"Saved {out}")
    if csv_path:
        write_csv([result], csv_path)
        console.print(f"Saved CSV {csv_path}")
    if csv_requests_path:
        write_requests_csv(result, csv_requests_path)
        console.print(f"Saved requests CSV {csv_requests_path}")


@app.command()
def run(
    ctx: typer.Context,
    url: Annotated[
        str | None, typer.Option(help="Base URL of the API, e.g. http://localhost:11434/v1")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Model name as the server knows it")] = None,
    concurrency: Annotated[
        str | None,
        typer.Option(
            help="Closed-loop sweep: comma-separated in-flight request counts, e.g. 1,2,4,8"
        ),
    ] = None,
    rate: Annotated[
        str | None,
        typer.Option(
            help="Open-loop sweep: comma-separated Poisson arrival rates in req/s; 'inf' sends "
            "all at once"
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("-c", "--config", help="YAML, TOML, or JSON benchmark config file"),
    ] = None,
    repeat: Annotated[
        int,
        typer.Option(
            "-r", "--repeat", help="Run each level N times to measure variance (mean ± spread)"
        ),
    ] = 1,
    min_duration: Annotated[
        float | None,
        typer.Option(
            help="Minimum duration in seconds per level to ensure steady-state measurement"
        ),
    ] = None,
    csv: Annotated[Path | None, typer.Option(help="Write level summaries to this CSV file")] = None,
    csv_requests: Annotated[
        Path | None, typer.Option(help="Write per-request details to this CSV file")
    ] = None,
    max_concurrency: MaxConcurrency = None,
    num_requests: NumRequests = None,
    duration: Duration = None,
    input_len: InputLen = 256,
    output_len: OutputLen = 128,
    prompts: Prompts = "synthetic",
    endpoint: Endpoint = "chat",
    ignore_eos: IgnoreEos = True,
    temperature: Temperature = 0.0,
    warmup: Warmup = 2,
    seed: Seed = None,
    timeout: Timeout = 300.0,
    slo_ttft_ms: SloTtft = 500.0,
    slo_tpot_ms: SloTpot = 50.0,
    tokenizer: Tokenizer = None,
    api_key: ApiKey = None,
    label: Label = None,
    notes: Notes = None,
    out: Out = None,
    preset: Annotated[
        str | None, typer.Option(help="Workload preset: 'chat', 'rag', 'code', 'agent'")
    ] = None,
    input_dist: Annotated[
        str | None,
        typer.Option(help="Input length distribution: 'fixed', 'normal', 'lognormal', 'uniform'"),
    ] = None,
    output_dist: Annotated[
        str | None,
        typer.Option(help="Output length distribution: 'fixed', 'normal', 'lognormal', 'uniform'"),
    ] = None,
    arrival_dist: Annotated[
        str, typer.Option(help="Arrival distribution: 'poisson' or 'gamma'")
    ] = "poisson",
    burstiness: Annotated[
        float, typer.Option(help="Gamma shape parameter for arrival bursts (<1.0 is bursty)")
    ] = 1.0,
    shared_prefix_tokens: Annotated[
        int | None, typer.Option(help="Tokens in shared prefix to test prefix caching")
    ] = None,
    trace: Annotated[
        Path | None, typer.Option(help="Replay requests from a JSONL trace file")
    ] = None,
    gpu: Annotated[
        bool, typer.Option("--gpu/--no-gpu", help="Capture GPU telemetry via NVML")
    ] = True,
    gpu_index: Annotated[int, typer.Option(help="NVIDIA GPU device index to monitor")] = 0,
    scrape_metrics: Annotated[
        str | None, typer.Option(help="Prometheus /metrics endpoint URL or 'auto'")
    ] = None,
) -> None:
    """Benchmark an endpoint over a sweep of load levels."""
    if config:
        cfg_data = load_config_file(config)

        def _get(name, current):
            src = ctx.get_parameter_source(name)
            if src is None or str(src).endswith("DEFAULT") or current is None:
                return cfg_data.get(name, current)
            return current

        url = _get("url", url)
        model = _get("model", model)
        concurrency = _get("concurrency", concurrency)
        rate = _get("rate", rate)
        max_concurrency = _get("max_concurrency", max_concurrency)
        num_requests = _get("num_requests", num_requests)
        duration = _get("duration", duration)
        min_duration = _get("min_duration", min_duration)
        repeat = _get("repeat", repeat)
        input_len = _get("input_len", input_len)
        output_len = _get("output_len", output_len)
        prompts = _get("prompts", prompts)
        endpoint = _get("endpoint", endpoint)
        ignore_eos = _get("ignore_eos", ignore_eos)
        temperature = _get("temperature", temperature)
        warmup = _get("warmup", warmup)
        seed = _get("seed", seed)
        timeout = _get("timeout", timeout)
        slo_ttft_ms = _get("slo_ttft_ms", slo_ttft_ms)
        slo_tpot_ms = _get("slo_tpot_ms", slo_tpot_ms)
        tokenizer = _get("tokenizer", tokenizer)
        api_key = _get("api_key", api_key)
        label = _get("label", label)
        notes = _get("notes", notes)
        preset = _get("preset", preset)
        input_dist = _get("input_dist", input_dist)
        output_dist = _get("output_dist", output_dist)
        arrival_dist = _get("arrival_dist", arrival_dist)
        burstiness = _get("burstiness", burstiness)
        shared_prefix_tokens = _get("shared_prefix_tokens", shared_prefix_tokens)
        if trace is None and "trace" in cfg_data:
            trace = Path(cfg_data["trace"])
        gpu = _get("gpu", gpu)
        gpu_index = _get("gpu_index", gpu_index)
        scrape_metrics = _get("scrape_metrics", scrape_metrics)
        if out is None and "out" in cfg_data:
            out = Path(cfg_data["out"])
        if csv is None and "csv" in cfg_data:
            csv = Path(cfg_data["csv"])
        if csv_requests is None and "csv_requests" in cfg_data:
            csv_requests = Path(cfg_data["csv_requests"])

    if not url:
        raise typer.BadParameter("give --url or specify it in --config")
    if not model:
        raise typer.BadParameter("give --model or specify it in --config")

    if isinstance(concurrency, (list, tuple)):
        concurrency = ",".join(str(x) for x in concurrency)
    elif concurrency is not None:
        concurrency = str(concurrency)

    if isinstance(rate, (list, tuple)):
        rate = ",".join(str(x) for x in rate)
    elif rate is not None:
        rate = str(rate)

    if (concurrency is None) == (rate is None):
        raise typer.BadParameter("give exactly one of --concurrency or --rate")
    mode = "concurrency" if concurrency is not None else "rate"
    levels = parse_levels(concurrency if concurrency is not None else rate)
    if mode == "concurrency" and any(math.isinf(v) or v != int(v) for v in levels):
        raise typer.BadParameter("--concurrency levels must be whole numbers")
    if duration and any(math.isinf(v) for v in levels):
        raise typer.BadParameter("--duration needs finite rates ('inf' sends all at once)")
    st = _setup(
        url=url,
        model=model,
        mode=mode,
        levels=levels,
        max_concurrency=max_concurrency,
        num_requests=num_requests,
        duration=duration,
        min_duration=min_duration,
        repeat=repeat,
        input_len=input_len,
        output_len=output_len,
        prompts=prompts,
        endpoint=endpoint,
        ignore_eos=ignore_eos,
        temperature=temperature,
        warmup=warmup,
        seed=seed,
        timeout=timeout,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        tokenizer=tokenizer,
        api_key=api_key,
        notes=notes,
        preset=preset,
        input_dist=input_dist,
        output_dist=output_dist,
        arrival_dist=arrival_dist,
        burstiness=burstiness,
        shared_prefix_tokens=shared_prefix_tokens,
        trace=trace,
        gpu_telemetry=gpu,
        scrape_metrics=scrape_metrics,
    )

    rep_txt = f" (x{repeat} repeats)" if repeat > 1 else ""
    console.print(
        f"servebench {__version__}: {mode} sweep {levels}{rep_txt} against {url} ({model})"
    )
    try:
        levels_out = asyncio.run(
            run_sweep(
                st.spec,
                st.prompts_for,
                mode,
                levels,
                num_requests,
                duration=duration,
                min_duration=min_duration,
                repeat=repeat,
                warmup_requests=warmup,
                timeout=timeout,
                seed=st.cfg.seed,
                max_concurrency=max_concurrency,
                slo_ttft_ms=slo_ttft_ms,
                slo_tpot_ms=slo_tpot_ms,
                counter=st.counter,
                on_level=_print_level,
                burstiness=burstiness,
                gpu_telemetry=gpu,
                gpu_index=gpu_index,
                metrics_url=st.metrics_url,
            )
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    st.after_run()
    run_res = RunResult(
        label=label or (out.stem if out else model),
        config=st.cfg,
        environment=st.env,
        levels=levels_out,
    )
    gpus_with_samples = [lv.gpu_telemetry for lv in levels_out if lv.gpu_telemetry]
    if gpus_with_samples:
        run_res.gpu_telemetry = gpus_with_samples[-1]
    diag = diagnose_run(run_res)
    run_res.bottleneck_diagnosis = diag.model_dump()
    _finish(run_res, out, csv, csv_requests)


@app.command("find-capacity")
def find_capacity_cmd(
    url: Url,
    model: Model,
    mode: Annotated[
        str, typer.Option(help="'concurrency' (in-flight requests) or 'rate' (Poisson req/s)")
    ] = "concurrency",
    start: Annotated[float, typer.Option(help="First load level to try")] = 1,
    max_load: Annotated[float, typer.Option("--max", help="Highest load level to try")] = 256,
    target: Annotated[
        float, typer.Option(help="Share of requests that must meet the SLOs for a level to pass")
    ] = 0.9,
    tolerance: Annotated[
        float,
        typer.Option(
            help="Stop when the pass/fail gap is within this fraction of the passing load"
        ),
    ] = 0.1,
    max_concurrency: MaxConcurrency = None,
    num_requests: NumRequests = None,
    duration: Duration = None,
    input_len: InputLen = 256,
    output_len: OutputLen = 128,
    prompts: Prompts = "synthetic",
    endpoint: Endpoint = "chat",
    ignore_eos: IgnoreEos = True,
    temperature: Temperature = 0.0,
    warmup: Warmup = 2,
    seed: Seed = None,
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
    st = _setup(
        url=url,
        model=model,
        mode=mode,
        levels=[],
        max_concurrency=max_concurrency,
        num_requests=num_requests,
        duration=duration,
        input_len=input_len,
        output_len=output_len,
        prompts=prompts,
        endpoint=endpoint,
        ignore_eos=ignore_eos,
        temperature=temperature,
        warmup=warmup,
        seed=seed,
        timeout=timeout,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        tokenizer=tokenizer,
        api_key=api_key,
        notes=notes,
    )

    console.print(
        f"servebench {__version__}: finding {mode} capacity against {url} ({model}); "
        f"a level passes with no errors and {target * 100:.0f}% of requests in SLO"
    )
    result = _capacity_search(
        st,
        label or (out.stem if out else model),
        start=start,
        max_load=max_load,
        target=target,
        tolerance=tolerance,
        num_requests=num_requests,
        duration=duration,
        max_concurrency=max_concurrency,
    )
    _finish(result, out)


def _capacity_search(
    st: Setup,
    label: str,
    *,
    start: float,
    max_load: float,
    target: float,
    tolerance: float,
    num_requests: int | None,
    duration: float | None,
    max_concurrency: int | None,
) -> RunResult:
    """Warm up, search for capacity, ask the server about the model; shared by
    `find-capacity` and `quick`."""
    c = st.cfg

    def on_level(lv: LevelResult) -> None:
        _print_level(lv, "  -> pass" if passes(lv, target) else "  -> FAIL")

    async def go():
        await check_warmup(st.spec, st.prompts_for, c.warmup, c.timeout, c.seed)
        measure = level_runner(
            st.spec,
            st.prompts_for,
            c.mode,
            num_requests,
            duration=duration,
            timeout=c.timeout,
            seed=c.seed,
            max_concurrency=max_concurrency,
            slo_ttft_ms=c.slo_ttft_ms,
            slo_tpot_ms=c.slo_tpot_ms,
            counter=st.counter,
            on_level=on_level,
        )
        return await find_capacity(
            measure,
            c.mode,
            start=start,
            max_value=max_load,
            target=target,
            tolerance=tolerance,
            slo_ttft_ms=c.slo_ttft_ms,
            slo_tpot_ms=c.slo_tpot_ms,
        )

    try:
        levels_out, cap = asyncio.run(go())
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    c.levels = [lv.value for lv in levels_out]
    st.after_run()
    return RunResult(label=label, config=c, environment=st.env, levels=levels_out, capacity=cap)


@app.command()
def quick(
    url: Annotated[
        str | None,
        typer.Option(
            help="OpenAI-compatible base URL; default: look for a local server (Ollama, "
            "llama.cpp, LM Studio, vLLM, SGLang on their usual ports)"
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Model to measure; default: the loaded one (Ollama) or the first listed"),
    ] = None,
    max_users: Annotated[
        int, typer.Option("--max", help="Most users to try")
    ] = quick_mod.QUICK_MAX_USERS,
    slo_ttft_ms: SloTtft = 500.0,
    slo_tpot_ms: SloTpot = 50.0,
    api_key: ApiKey = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="Where to save the run (default: servebench-quick-<engine>-<model>.json)"
        ),
    ] = None,
) -> None:
    """Find your local LLM server, measure it for about a minute, and say how many users it
    can serve. The no-setup way to start; use `find-capacity` / `run` for full control."""
    if url:
        server = detect.identify(url)
        if server is None:
            console.print(f"[red]No OpenAI-compatible server answered at {url}/models.[/red]")
            raise typer.Exit(1)
    else:
        found = detect.find_local()
        if not found:
            ports = ", ".join(f"{name} :{port}" for port, name in detect.KNOWN_PORTS)
            console.print(f"[red]No local LLM server found[/red] (looked for {ports}).")
            if detect.ollama_installed():
                console.print(
                    "Ollama is installed but not running: open the Ollama app, or "
                    "run `ollama serve` in another terminal, then run this again."
                )
            else:
                console.print(
                    "Start one, for example: ollama serve, then ollama pull "
                    "qwen2.5:0.5b. Or pass --url for a server elsewhere."
                )
            raise typer.Exit(1)
        server = found[0]
        if len(found) > 1:
            others = ", ".join(f"{s.engine} ({s.url})" for s in found[1:])
            console.print(f"[dim]Also found: {others}. Use --url to pick one.[/dim]")
    model = model or quick_mod.pick_model(server)
    if not model:
        console.print(
            f"[red]{server.engine} at {server.url} lists no models.[/red] "
            "For Ollama: ollama pull qwen2.5:0.5b"
        )
        raise typer.Exit(1)

    st = _setup(
        url=server.url,
        model=model,
        mode="concurrency",
        levels=[],
        max_concurrency=None,
        num_requests=quick_mod.QUICK_REQUESTS_PER_LEVEL,
        duration=None,
        input_len=quick_mod.QUICK_INPUT_LEN,
        output_len=quick_mod.QUICK_OUTPUT_LEN,
        prompts="synthetic",
        endpoint="chat",
        ignore_eos=True,
        temperature=0.0,
        warmup=2,
        seed=None,
        timeout=120.0,
        slo_ttft_ms=slo_ttft_ms,
        slo_tpot_ms=slo_tpot_ms,
        tokenizer=None,
        api_key=api_key,
        notes="servebench quick",
    )
    from .report import machine_text

    engine = " ".join(x for x in (server.engine, server.version) if x)
    console.print(f"servebench quick: {engine} at {server.url}, model {model}")
    result = _capacity_search(
        st,
        f"{server.engine} {model}",
        start=1,
        max_load=max_users,
        target=0.9,
        tolerance=0.1,
        num_requests=quick_mod.QUICK_REQUESTS_PER_LEVEL,
        duration=None,
        max_concurrency=None,
    )
    out = out or Path(quick_mod.default_out(server.engine, model))
    save_run(result, out)
    console.print()
    console.print(f"[bold]{engine}, {model}[/bold] on {machine_text(result)}")
    for line in quick_mod.summary(result):
        console.print(line)
    from .report import warnings_for

    for w in warnings_for(result):
        console.print(f"[yellow]! {w}[/yellow]")
    console.print(
        f"[dim]Saved {out}. Share it: servebench share {out}  |  full report: "
        f"servebench report {out} --out report.html[/dim]"
    )


@app.command()
def report(
    runs: Annotated[
        list[Path], typer.Argument(help="One or more run.json files", exists=True, dir_okay=False)
    ],
    out: Annotated[Path | None, typer.Option(help="Write an .html or .md report")] = None,
    csv: Annotated[Path | None, typer.Option(help="Write level summaries to this CSV file")] = None,
    csv_requests: Annotated[
        Path | None, typer.Option(help="Write per-request details to this CSV file (first run)")
    ] = None,
    slo_ttft_ms: Annotated[
        float | None, typer.Option(help="Recompute goodput with this TTFT SLO")
    ] = None,
    slo_tpot_ms: Annotated[
        float | None, typer.Option(help="Recompute goodput with this TPOT SLO")
    ] = None,
) -> None:
    """Print and compare saved runs; optionally write an HTML/Markdown report with charts or CSV."""
    try:
        loaded = [load_run(p) for p in runs]
    except RunFileError as e:
        raise typer.BadParameter(str(e)) from e
    if slo_ttft_ms is not None or slo_tpot_ms is not None:
        for r in loaded:
            resummarize(
                r,
                slo_ttft_ms if slo_ttft_ms is not None else r.config.slo_ttft_ms,
                slo_tpot_ms if slo_tpot_ms is not None else r.config.slo_tpot_ms,
            )
    for r in loaded:
        print_run(r, console)
    if out:
        try:
            write_report(loaded, out)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        console.print(f"Saved {out}")
    if csv:
        write_csv(loaded, csv)
        console.print(f"Saved CSV {csv}")
    if csv_requests and loaded:
        write_requests_csv(loaded[0], csv_requests)
        console.print(f"Saved requests CSV {csv_requests}")


@app.command()
def share(
    run_file: Annotated[
        Path,
        typer.Argument(
            help="A run.json (from quick, run or find-capacity)", exists=True, dir_okay=False
        ),
    ],
    card: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the PNG card (default: next to the run, <name>-card.png)"
        ),
    ] = None,
    no_card: Annotated[bool, typer.Option("--no-card", help="Only print the Markdown")] = False,
) -> None:
    """Print a Markdown summary to paste into a post or issue, and write a PNG card."""
    from . import share as share_mod

    try:
        run_ = load_run(run_file)
    except RunFileError as e:
        raise typer.BadParameter(str(e)) from e
    # plain print, not rich: the Markdown must come out exactly as written, ready to copy
    print(share_mod.markdown(run_))
    if not no_card:
        path = share_mod.card(run_, card or run_file.with_name(f"{run_file.stem}-card.png"))
        console.print(f"\n[dim]Card image: {path}[/dim]")


@app.command("mock-server")
def mock_server(
    port: int = 8000,
    host: str = "127.0.0.1",
    ttft_ms: float = 100.0,
    itl_ms: float = 20.0,
    tokens_per_chunk: int = 1,
    slowdown: Annotated[
        float,
        typer.Option(help="Extra latency per extra in-flight request, as a fraction (0.1 = +10%)"),
    ] = 0.0,
) -> None:
    """Run a mock OpenAI-compatible server with known timings (needs the mock extra)."""
    try:
        from .mock_server import MockConfig, serve
    except ImportError as e:
        console.print(r"[red]mock-server needs the servebench\[mock] extra[/red]")
        raise typer.Exit(1) from e
    console.print(
        f"Mock server on http://{host}:{port}/v1 (model 'mock-model'), "
        f"TTFT {ttft_ms} ms, ITL {itl_ms} ms. Ctrl+C to stop."
    )
    serve(
        MockConfig(
            ttft=ttft_ms / 1000,
            itl=itl_ms / 1000,
            tokens_per_chunk=tokens_per_chunk,
            slowdown=slowdown,
        ),
        host=host,
        port=port,
    )


@app.command("compare-models")
def compare_models_cmd(
    files: Annotated[list[Path], typer.Argument(help="Two or more run.json files to compare")],
) -> None:
    """Compare multiple models side-by-side on latency, throughput, and energy efficiency."""
    if len(files) < 2:
        raise typer.BadParameter("Provide at least two run.json files to compare.")
    runs = [load_run(f) for f in files]

    tbl = Table(
        title="Model Serving Comparison",
        box=None,
        pad_edge=False,
        header_style="bold underline",
    )
    tbl.add_column("Model / Label", justify="left")
    tbl.add_column("Engine", justify="left")
    tbl.add_column("Peak Tok/s", justify="right")
    tbl.add_column("TTFT p50 (ms)", justify="right")
    tbl.add_column("TPOT p50 (ms)", justify="right")
    tbl.add_column("Goodput", justify="right")
    tbl.add_column("Avg Power (W)", justify="right")
    tbl.add_column("Tok/Joule", justify="right")
    tbl.add_column("Primary Bottleneck", justify="left")

    base_tok_s = None
    for r in runs:
        engine = r.environment.engine or "-"
        peak_lv = max(r.levels, key=lambda lv: lv.summary.output_throughput) if r.levels else None
        if not peak_lv:
            continue
        sm = peak_lv.summary
        tok_s = sm.output_throughput
        if base_tok_s is None:
            base_tok_s = tok_s
            tok_s_str = f"{tok_s:.1f}"
        else:
            diff = ((tok_s - base_tok_s) / base_tok_s) * 100.0 if base_tok_s > 0 else 0.0
            sign = "+" if diff >= 0 else ""
            tok_s_str = f"{tok_s:.1f} ({sign}{diff:.0f}%)"

        ttft_str = f"{sm.ttft.p50 * 1000:.0f}" if sm.ttft.p50 is not None else "-"
        tpot_str = f"{sm.tpot.p50 * 1000:.1f}" if sm.tpot.p50 is not None else "-"
        good_str = f"{sm.goodput:.1f}"

        gt = peak_lv.gpu_telemetry or r.gpu_telemetry
        pow_str = f"{gt.avg_power_w:.1f}" if gt and gt.avg_power_w is not None else "-"
        eff_str = f"{gt.tokens_per_joule:.2f}" if gt and gt.tokens_per_joule is not None else "-"

        bn = (
            r.bottleneck_diagnosis.get("primary_bottleneck", "-") if r.bottleneck_diagnosis else "-"
        )
        tbl.add_row(r.label, engine, tok_s_str, ttft_str, tpot_str, good_str, pow_str, eff_str, bn)

    console.print(tbl)


@app.command("find-context-limit")
def find_context_limit_cmd(
    url: Url,
    model: Model,
    start: Annotated[int, typer.Option(help="Starting prompt length in tokens")] = 512,
    max_len: Annotated[int, typer.Option("--max", help="Maximum prompt length to test")] = 32768,
    output_len: OutputLen = 16,
    timeout: Timeout = 60.0,
    api_key: ApiKey = None,
) -> None:
    """Find maximum working context window before server OOM, context error, or timeout."""
    spec = RequestSpec(url=url, model=model, output_len=output_len, api_key=api_key)
    lengths = []
    curr = start
    while curr <= max_len:
        lengths.append(curr)
        curr *= 2

    console.print(f"Finding context limit for {model} @ {url} (lengths: {lengths})...")

    async def _test_limits():
        last_pass = None
        fail_reason = None
        for in_len in lengths:
            p = Prompt(" ".join(["test"] * in_len), "synthetic")
            async with make_session(1, timeout) as sess:
                res = await send_request(sess, spec, p)
                if res.ok:
                    last_pass = in_len
                    ttft_ms = (res.ttft * 1000.0) if res.ttft else 0.0
                    console.print(
                        f"  [green]✓ {in_len:>5} tokens: ok (TTFT {ttft_ms:.0f} ms)[/green]"
                    )
                else:
                    fail_reason = res.error
                    console.print(f"  [red]✗ {in_len:>5} tokens: failed ({res.error})[/red]")
                    break
        return last_pass, fail_reason

    last_pass, fail_reason = asyncio.run(_test_limits())
    console.print()
    if last_pass:
        console.print(f"[bold green]Maximum context supported: >= {last_pass} tokens[/bold green]")
        if fail_reason:
            console.print(f"[dim]Failed at next step due to: {fail_reason}[/dim]")
    else:
        console.print(f"[bold red]Failed even at initial {start} tokens: {fail_reason}[/bold red]")


@app.command("check")
def check_cmd(
    candidate: Annotated[
        Path,
        typer.Argument(
            help="Candidate run.json file to evaluate for regressions",
            exists=True,
            dir_okay=False,
        ),
    ],
    baseline: Annotated[
        Path,
        typer.Option(
            "--baseline",
            "-b",
            help="Baseline run.json file to compare against",
            exists=True,
            dir_okay=False,
        ),
    ],
    max_ttft_pct: Annotated[
        float,
        typer.Option("--max-ttft-degradation-pct", help="Max allowed TTFT p50 latency increase %"),
    ] = 15.0,
    max_tpot_pct: Annotated[
        float,
        typer.Option("--max-tpot-degradation-pct", help="Max allowed TPOT p50 latency increase %"),
    ] = 10.0,
    max_throughput_pct: Annotated[
        float,
        typer.Option(
            "--max-throughput-degradation-pct", help="Max allowed output throughput drop %"
        ),
    ] = 10.0,
    max_goodput_pct: Annotated[
        float,
        typer.Option("--max-goodput-degradation-pct", help="Max allowed goodput drop %"),
    ] = 10.0,
    max_error_rate_pct: Annotated[
        float,
        typer.Option("--max-error-rate-pct", help="Max allowed request error rate %"),
    ] = 1.0,
) -> None:
    """CI Performance Regression Gate: compare candidate run vs baseline and fail on regressions."""
    try:
        cand_run = load_run(candidate)
        base_run = load_run(baseline)
    except RunFileError as e:
        raise typer.BadParameter(str(e)) from e

    thresh = CheckThresholds(
        max_ttft_pct=max_ttft_pct,
        max_tpot_pct=max_tpot_pct,
        max_throughput_pct=max_throughput_pct,
        max_goodput_pct=max_goodput_pct,
        max_error_rate_pct=max_error_rate_pct,
    )

    report = evaluate_regression(cand_run, base_run, thresh)
    print_check_report(report, console)

    if not report.passed:
        raise typer.Exit(1)


@app.command("roofline")
def roofline_cmd(
    run_file: Annotated[
        Path | None,
        typer.Argument(
            help="Optional run.json file to compute empirical roofline from",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model name (e.g., qwen2.5:0.5b, llama3:8b-q4_k_m)"),
    ] = "qwen2.5:0.5b",
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            "--gpu-name",
            "-g",
            help="GPU device name override (e.g., 'rtx 3050', 'a100', 't4')",
        ),
    ] = None,
    bandwidth: Annotated[
        float | None,
        typer.Option(
            "--bandwidth",
            "--gpu-mem-bw",
            "-b",
            help="Custom peak memory bandwidth in GB/s",
        ),
    ] = None,
    peak_tflops: Annotated[
        float | None,
        typer.Option(
            "--peak-tflops",
            "--gpu-tflops",
            help="Custom peak FP16/BF16 Tensor TFLOPs",
        ),
    ] = None,
) -> None:
    """Hardware Roofline Model: compute MBU %, MFU %, and plot ASCII roofline curve."""
    levels_data = []
    effective_model = model
    effective_gpu = gpu

    if run_file is not None:
        try:
            run = load_run(run_file)
            effective_model = run.config.model or model
            if effective_gpu is None:
                if run.gpu_telemetry and run.gpu_telemetry.device_name:
                    effective_gpu = run.gpu_telemetry.device_name
                elif run.environment.gpus:
                    effective_gpu = run.environment.gpus[0]
            for lv in run.levels:
                out_t = (
                    getattr(lv.summary, "output_throughput", 0.0)
                    or getattr(lv.summary, "output_tokens_per_s", 0.0)
                    or 0.0
                )
                in_t = getattr(lv.summary, "input_throughput", None) or (out_t * 0.25)
                levels_data.append(
                    {
                        "concurrency": int(lv.value),
                        "output_tokens_per_s": out_t,
                        "prompt_tokens_per_s": in_t,
                    }
                )
        except Exception as e:
            raise typer.BadParameter(f"Could not load run file: {e}") from e
    else:
        # Default representative sweeps if no run file provided
        levels_data = [
            {"concurrency": 1, "output_tokens_per_s": 85.0},
            {"concurrency": 2, "output_tokens_per_s": 145.0},
            {"concurrency": 4, "output_tokens_per_s": 220.0},
            {"concurrency": 8, "output_tokens_per_s": 240.0},
        ]

    analysis = compute_roofline(
        model_name=effective_model,
        levels_data=levels_data,
        gpu_name=effective_gpu,
        custom_bandwidth_gb_s=bandwidth,
        custom_peak_tflops=peak_tflops,
    )

    console.print(render_ascii_roofline(analysis))

    t = Table(title=f"Roofline Breakdown: {analysis.gpu_name} ({analysis.model_name})", box=None)
    t.add_column("Concurrency", justify="center", style="cyan")
    t.add_column("Throughput (tok/s)", justify="right", style="green")
    t.add_column("Achieved TFLOPs", justify="right", style="white")
    t.add_column("MBU (Memory %)", justify="right", style="yellow")
    t.add_column("MFU (Compute %)", justify="right", style="magenta")
    t.add_column("Regime", style="bold")

    for pt in analysis.points:
        t.add_row(
            str(pt.concurrency),
            f"{pt.achieved_tok_s:.1f}",
            f"{pt.achieved_tflops:.3f}",
            f"{pt.mbu_percent:.1f}%",
            f"{pt.mfu_percent:.1f}%",
            f"[yellow]{pt.regime}[/yellow]" if "Memory" in pt.regime else f"[red]{pt.regime}[/red]",
        )
    console.print(t)


@app.command("tune")
def tune_cmd(
    run_file: Annotated[
        Path | None,
        typer.Argument(
            help="Optional completed run.json to derive tuning recommendations from",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    url: Annotated[
        str,
        typer.Option("--url", "-u", help="Base URL of OpenAI-compatible server"),
    ] = "http://127.0.0.1:11434/v1",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model name"),
    ] = "qwen2.5:0.5b",
    target_ttft_ms: Annotated[
        float,
        typer.Option("--target-ttft-ms", help="Target TTFT p50 latency threshold in ms"),
    ] = 150.0,
    target_itl_ms: Annotated[
        float,
        typer.Option("--target-itl-ms", help="Target ITL p50 latency threshold in ms"),
    ] = 40.0,
    engine: Annotated[
        str | None,
        typer.Option(
            "--engine",
            "-e",
            help="Target engine override (ollama, vllm, llamacpp, mini-infer)",
        ),
    ] = None,
) -> None:
    """Automated Server Tuning: evaluate concurrency knees and generate production config flags."""
    if run_file is not None:
        try:
            run = load_run(run_file)
        except Exception as e:
            raise typer.BadParameter(f"Could not load run file: {e}") from e
    else:
        console.print(
            f"[cyan]No run file provided. Running automated capacity sweep on {url}...[/cyan]"
        )
        try:
            detected = detect.identify(url)
            target_model = detected.models[0] if detected and detected.models else model
            cfg = RunConfig(
                url=url,
                model=target_model,
                levels=[1.0, 2.0, 4.0, 8.0],
                mode="concurrency",
                num_requests=16,
                warmup_requests=2,
            )
            run = asyncio.run(run_sweep(cfg, console=console))
        except Exception as e:
            raise typer.BadParameter(f"Failed to run automated tuning sweep: {e}") from e

    report = evaluate_tuning(
        run=run,
        engine_override=engine,
        target_ttft_p50_ms=target_ttft_ms,
        target_itl_p50_ms=target_itl_ms,
    )

    rec = report.recommendation
    console.print(
        f"\n[bold green]⚡ Optimal Concurrency Knee: "
        f"{rec.recommended_concurrency} concurrent streams[/bold green]"
    )
    console.print(
        f"[dim]Peak Sustainable Goodput:[/dim] "
        f"[bold white]{rec.knee_throughput_tok_s:.1f} tok/s[/bold white]\n"
    )

    t = Table(title=f"Recommended Configuration for {rec.engine.upper()}", box=None)
    t.add_column("Type", style="cyan", width=14)
    t.add_column("Parameter / Flag", style="bold white")

    for k, v in rec.recommended_env_vars.items():
        t.add_row("Environment", f"{k}={v}")
    for flag in rec.recommended_cli_flags:
        t.add_row("CLI Flag", flag)

    console.print(t)

    console.print("\n[bold]Copy-pasteable Launch Command:[/bold]")
    console.print(f"  [bold yellow]{rec.launch_command_example}[/bold yellow]\n")

    console.print("[bold]Engineering Rationale:[/bold]")
    for r in rec.rationale:
        console.print(f"  • {r}")


@app.command("top")
def top_cmd(
    url: Annotated[
        str,
        typer.Option("--url", "-u", help="Base URL of OpenAI-compatible server"),
    ] = "http://127.0.0.1:11434/v1",
    refresh: Annotated[
        float,
        typer.Option("--refresh", "-r", help="Dashboard refresh interval in seconds"),
    ] = 0.5,
    duration: Annotated[
        float | None,
        typer.Option("--duration", "-d", help="Exit after N seconds (useful for scripts/testing)"),
    ] = None,
    inline: Annotated[
        bool,
        typer.Option("--inline", help="Render inline without switching to alternate screen buffer"),
    ] = False,
) -> None:
    """Real-Time Interactive TUI: live GPU telemetry, wattage, tokens/sec, and queue monitor."""
    try:
        asyncio.run(
            run_top_monitor(
                url=url,
                refresh_rate_s=refresh,
                max_duration_s=duration,
                inline=inline,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
