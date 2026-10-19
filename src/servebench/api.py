"""Stable Python API for programmatic LLM serving benchmarks.

Allows executing benchmarks, running quick capacity discovery, evaluating regressions,
and comparing runs directly in Python scripts, test harnesses, or Jupyter notebooks.

Example:
    >>> import servebench
    >>> res = servebench.benchmark(
    ...     url="http://127.0.0.1:8000/v1",
    ...     model="llama-3-8b",
    ...     mode="concurrency",
    ...     levels=[1, 4, 8],
    ...     duration=10.0,
    ... )
    >>> peak = max(lv.summary.output_throughput for lv in res.levels)
    >>> print(f"Peak throughput: {peak:.1f} tok/s")
"""

from __future__ import annotations

import asyncio
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from . import __version__, detect, system
from .check import CheckReport, CheckThresholds, evaluate_regression
from .client import RequestSpec
from .load import run_sweep
from .prompts import Prompt, load_prompts
from .report import load_run, save_run
from .results import Environment, RunConfig, RunResult
from .tokens import make_counter
from .workloads import PRESETS, generate_workload_requests, load_trace_file


def _build_env(url: str, notes: str | None = None) -> Environment:
    server = detect.identify(url)
    host = urlparse(url).hostname or ""
    return Environment(
        servebench_version=__version__,
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        local_server=host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"),
        engine=server.engine if server else None,
        engine_version=server.version if server else None,
        notes=notes,
        **system.describe_for_env(),
    )


def benchmark(
    url: str,
    model: str,
    *,
    mode: Literal["concurrency", "rate"] = "concurrency",
    levels: list[float] | None = None,
    num_requests: int | None = 20,
    duration: float | None = None,
    input_len: int = 256,
    output_len: int = 128,
    preset: str | None = None,
    burstiness: float = 1.0,
    trace: str | Path | None = None,
    shared_prefix_tokens: int | None = None,
    gpu_telemetry: bool = True,
    scrape_metrics: str | None = None,
    endpoint: Literal["chat", "completions"] = "chat",
    slo_ttft_ms: float | None = 500.0,
    slo_tpot_ms: float | None = 50.0,
    warmup: int = 2,
    seed: int = 0,
    timeout: float = 300.0,
    api_key: str | None = None,
    label: str | None = None,
    save_path: str | Path | None = None,
) -> RunResult:
    """Run a serving benchmark synchronously and return a RunResult."""
    if preset and preset in PRESETS:
        input_len = PRESETS[preset]["input_len"]
        output_len = PRESETS[preset]["output_len"]

    if levels is None:
        levels = [1.0, 2.0, 4.0] if mode == "concurrency" else [5.0, 10.0]

    env = _build_env(url)
    label_val = label or f"{model} ({mode})"

    spec = RequestSpec(
        url=url,
        model=model,
        endpoint=endpoint,
        output_len=output_len,
        ignore_eos=True,
        temperature=0.0,
        api_key=api_key,
    )

    if trace:
        def prompts_for(n: int, _s: int) -> list[Prompt]:
            return load_trace_file(trace, n)
    elif shared_prefix_tokens:
        def prompts_for(n: int, s: int) -> list[Prompt]:
            return generate_workload_requests(
                n,
                input_len,
                output_len,
                shared_prefix_tokens=shared_prefix_tokens,
                seed=s,
            )
    else:
        def prompts_for(n: int, s: int) -> list[Prompt]:
            return load_prompts("synthetic", n, input_len, s)

    counter = make_counter(None)
    config = RunConfig(
        url=url,
        model=model,
        endpoint=endpoint,
        input_len=input_len,
        output_len=output_len,
        mode=mode,
        levels=levels,
        num_requests=num_requests,
        duration=duration,
        warmup=warmup,
        burstiness=burstiness,
        gpu_telemetry=gpu_telemetry,
        scrape_metrics=scrape_metrics,
    )

    levels_out = asyncio.run(
        run_sweep(
            spec,
            prompts_for,
            mode,
            levels,
            num_requests,
            duration=duration,
            warmup_requests=warmup,
            timeout=timeout,
            seed=seed,
            slo_ttft_ms=slo_ttft_ms,
            slo_tpot_ms=slo_tpot_ms,
            counter=counter,
            burstiness=burstiness,
            gpu_telemetry=gpu_telemetry,
            metrics_url=scrape_metrics,
        )
    )

    run_res = RunResult(
        label=label_val,
        config=config,
        environment=env,
        levels=levels_out,
    )

    if save_path:
        save_run(run_res, Path(save_path))

    return run_res


def check_regression(
    candidate: RunResult | Path | str,
    baseline: RunResult | Path | str,
    thresholds: CheckThresholds | None = None,
) -> CheckReport:
    """Evaluate performance regressions between candidate and baseline runs."""
    cand = load_run(Path(candidate)) if isinstance(candidate, (str, Path)) else candidate
    base = load_run(Path(baseline)) if isinstance(baseline, (str, Path)) else baseline
    return evaluate_regression(cand, base, thresholds)


def compare(runs: list[RunResult | Path | str]) -> list[RunResult]:
    """Load and normalize multiple benchmark runs for comparison."""
    loaded: list[RunResult] = []
    for r in runs:
        if isinstance(r, (str, Path)):
            loaded.append(load_run(Path(r)))
        elif isinstance(r, RunResult):
            loaded.append(r)
        else:
            raise TypeError(f"Expected RunResult, Path, or str, got {type(r)}")
    return loaded
