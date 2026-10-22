"""Terminal tables, charts and Markdown/HTML reports for one or more runs."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .metrics import percentile, summarize
from .results import SCHEMA_VERSION, LevelResult, RunResult, Summary


def _pyplot():
    """matplotlib is only imported when a chart is drawn: it costs ~0.4 s of startup on
    every command otherwise (`servebench --help`, `run`, ...)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# Categorical slots in fixed order (one per run), stepped separately for light and dark.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
        "series": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#33322f",
        "series": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],
    },
}
MAX_RUNS = 8


class RunFileError(ValueError):
    """A run.json that can't be read: newer format, or not a servebench file."""


def load_run(path: str | Path) -> RunResult:
    text = Path(path).read_text(encoding="utf-8")
    try:
        version = int(json.loads(text).get("schema_version", 1))
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
        raise RunFileError(f"{path} is not a servebench run file ({e})") from e
    if version > SCHEMA_VERSION:
        raise RunFileError(
            f"{path} was written by a newer servebench (file format "
            f"{version}, this version reads up to {SCHEMA_VERSION}); upgrade "
            "servebench to read it"
        )
    try:
        return RunResult.model_validate_json(text)
    except ValidationError as e:
        raise RunFileError(
            f"{path} is not a valid servebench run file: "
            f"{e.error_count()} problem(s), first: {e.errors()[0]['msg']} "
            f"at {'.'.join(map(str, e.errors()[0]['loc']))}"
        ) from e


def save_run(run: RunResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def to_csv(runs: list[RunResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "run",
            "label",
            "mode",
            "level",
            "repeat_count",
            "completed",
            "failed",
            "req_s",
            "req_s_std",
            "out_tok_s",
            "out_tok_s_std",
            "steady_out_tok_s",
            "ttft_p50_ms",
            "ttft_p50_std_ms",
            "ttft_p90_ms",
            "ttft_p99_ms",
            "tpot_p50_ms",
            "tpot_p50_std_ms",
            "tpot_p90_ms",
            "tpot_p99_ms",
            "itl_p50_ms",
            "e2e_p50_s",
            "goodput",
            "slo_attainment_pct",
            "server_prefill_p50_ms",
            "server_decode_p50_ms",
            "queue_overhead_p50_ms",
            "gpu_util_avg",
            "vram_used_avg_mb",
            "power_avg_w",
            "tokens_per_joule",
        ]
    )
    for run in runs:
        for lv in run.levels:
            s = lv.summary
            rm = lv.repeat_metrics or {}

            def _val(x: float | None, scale: float = 1.0, fmt: str = "{:.3f}") -> str:
                return "" if x is None else fmt.format(x * scale)

            req_std = _val(rm["request_throughput"].std) if "request_throughput" in rm else ""
            out_std = (
                _val(rm["output_throughput"].std, fmt="{:.1f}") if "output_throughput" in rm else ""
            )
            ttft_std = (
                _val(rm["ttft_p50"].std, scale=1000.0, fmt="{:.1f}") if "ttft_p50" in rm else ""
            )
            tpot_std = (
                _val(rm["tpot_p50"].std, scale=1000.0, fmt="{:.2f}") if "tpot_p50" in rm else ""
            )

            gt = lv.gpu_telemetry
            writer.writerow(
                [
                    run.label,
                    lv.label,
                    lv.mode,
                    lv.value,
                    lv.repeat_count,
                    s.completed,
                    s.failed,
                    _val(s.request_throughput, fmt="{:.2f}"),
                    req_std,
                    _val(s.output_throughput, fmt="{:.1f}"),
                    out_std,
                    _val(s.steady_state_output_throughput, fmt="{:.1f}"),
                    _val(s.ttft.p50, scale=1000.0, fmt="{:.1f}"),
                    ttft_std,
                    _val(s.ttft.p90, scale=1000.0, fmt="{:.1f}"),
                    _val(s.ttft.p99, scale=1000.0, fmt="{:.1f}"),
                    _val(s.tpot.p50, scale=1000.0, fmt="{:.2f}"),
                    tpot_std,
                    _val(s.tpot.p90, scale=1000.0, fmt="{:.2f}"),
                    _val(s.tpot.p99, scale=1000.0, fmt="{:.2f}"),
                    _val(s.itl.p50, scale=1000.0, fmt="{:.2f}"),
                    _val(s.e2e.p50, fmt="{:.2f}"),
                    _val(s.goodput, fmt="{:.2f}"),
                    _val(s.slo_attainment, scale=100.0, fmt="{:.1f}"),
                    _val(s.server_prefill_ms.p50, fmt="{:.1f}"),
                    _val(s.server_decode_ms.p50, fmt="{:.1f}"),
                    _val(s.queue_network_overhead_ms.p50, fmt="{:.1f}"),
                    _val(gt.avg_gpu_util, fmt="{:.1f}") if gt else "",
                    _val(gt.avg_vram_used_mb, fmt="{:.1f}") if gt else "",
                    _val(gt.avg_power_w, fmt="{:.2f}") if gt else "",
                    _val(gt.tokens_per_joule, fmt="{:.2f}") if gt else "",
                ]
            )
    return buf.getvalue()


def write_csv(runs: list[RunResult], path: str | Path) -> None:
    text = to_csv(runs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_requests_csv(run: RunResult, path: str | Path) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "level",
            "index",
            "ok",
            "error",
            "start_s",
            "ttft_ms",
            "e2e_s",
            "output_tokens",
            "prompt_tokens",
            "tpot_ms",
            "chunks",
            "token_source",
            "server_prefill_ms",
            "server_decode_ms",
            "queue_network_overhead_ms",
        ]
    )
    for lv in run.levels:
        for r in lv.requests:
            writer.writerow(
                [
                    lv.label,
                    r.index,
                    r.ok,
                    r.error or "",
                    f"{r.start:.4f}",
                    f"{r.ttft * 1000:.2f}" if r.ttft is not None else "",
                    f"{r.e2e:.4f}" if r.e2e is not None else "",
                    r.output_tokens,
                    r.prompt_tokens or "",
                    f"{r.tpot * 1000:.2f}" if r.tpot is not None else "",
                    r.num_chunks,
                    r.token_source,
                    f"{r.server_prefill_ms:.2f}" if r.server_prefill_ms is not None else "",
                    f"{r.server_decode_ms:.2f}" if r.server_decode_ms is not None else "",
                    f"{r.queue_network_overhead_ms:.2f}"
                    if r.queue_network_overhead_ms is not None
                    else "",
                ]
            )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(buf.getvalue(), encoding="utf-8")


def resummarize(run: RunResult, slo_ttft_ms: float | None, slo_tpot_ms: float | None) -> None:
    """Recompute summaries from raw requests, e.g. to apply different SLOs in a report."""
    for lv in run.levels:
        lv.summary = summarize(lv.requests, lv.duration, slo_ttft_ms, slo_tpot_ms)


# ---------- formatting ----------


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.0f}" if x * 1000 >= 10 else f"{x * 1000:.1f}"


def _num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


COLUMNS = [
    "load",
    "ok/fail",
    "req/s",
    "out tok/s",
    "TTFT p50",
    "TTFT p99",
    "TPOT p50",
    "TPOT p99",
    "ITL p50",
    "E2E p50 (s)",
    "goodput",
    "tok/chunk",
    "in SLO",
]
# The terminal table drops ITL p50 (equal to TPOT at one token per chunk) and "in SLO" (the
# run progress lines show it); both stay in run.json and the MD/HTML reports. That keeps it
# inside 80 columns even with 5-digit latencies.
TERMINAL_COLUMNS = [
    "load",
    "ok/\nfail",
    "req/s",
    "out\ntok/s",
    "TTFT\np50",
    "TTFT\np99",
    "TPOT\np50",
    "TPOT\np99",
    "E2E\np50 s",
    "good-\nput",
    "tok/\nchunk",
]
_TERMINAL_SKIP = {COLUMNS.index("ITL p50"), COLUMNS.index("in SLO")}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def level_row(lv: LevelResult) -> list[str]:
    s: Summary = lv.summary
    rm = lv.repeat_metrics or {}
    if "request_throughput" in rm:
        req_s = f"{rm['request_throughput'].mean:.2f}±{rm['request_throughput'].std:.2f}"
    else:
        req_s = _num(s.request_throughput, 2)

    if "output_throughput" in rm:
        out_tok = f"{rm['output_throughput'].mean:.0f}±{rm['output_throughput'].std:.0f}"
    else:
        out_tok = _num(s.output_throughput)

    if "ttft_p50" in rm:
        ttft_p50 = f"{rm['ttft_p50'].mean * 1000:.0f}±{rm['ttft_p50'].std * 1000:.0f}"
    else:
        ttft_p50 = _ms(s.ttft.p50)

    if "tpot_p50" in rm:
        tpot_p50 = f"{rm['tpot_p50'].mean * 1000:.1f}±{rm['tpot_p50'].std * 1000:.1f}"
    else:
        tpot_p50 = _ms(s.tpot.p50)

    lbl = lv.label + (f" (x{lv.repeat_count})" if lv.repeat_count > 1 else "")
    return [
        lbl,
        f"{s.completed}/{s.failed}",
        req_s,
        out_tok,
        ttft_p50,
        _ms(s.ttft.p99),
        tpot_p50,
        _ms(s.tpot.p99),
        _ms(s.itl.p50),
        _num(s.e2e.p50, 2),
        _num(s.goodput, 2),
        _num(s.tokens_per_chunk, 2),
        _pct(s.slo_attainment),
    ]


def _slo_desc(ttft_ms: float | None, tpot_ms: float | None) -> str:
    parts = []
    if ttft_ms is not None:
        parts.append(f"TTFT < {ttft_ms:g} ms")
    if tpot_ms is not None:
        parts.append(f"TPOT < {tpot_ms:g} ms")
    return " and ".join(parts) or "no SLO"


def slo_text(s: Summary) -> str:
    return _slo_desc(s.slo_ttft_ms, s.slo_tpot_ms)


def capacity_text(run: RunResult) -> str | None:
    """One-line answer from `find-capacity`, e.g. 'Capacity: 12 concurrent requests (...)'."""
    cap = run.capacity
    if cap is None:
        return None

    def load(v: float | None) -> str:
        if cap.mode == "rate":
            return f"{v:g} req/s"
        return f"{v:g} concurrent request" + ("" if v == 1 else "s")

    slo = _slo_desc(cap.slo_ttft_ms, cap.slo_tpot_ms)
    target = _pct(cap.target)
    by_value = {lv.value: lv.summary for lv in run.levels}

    def met(v: float | None) -> str:
        """'100%' or '23% (3 failed)': share of requests at load v that met the SLO."""
        s = by_value.get(v)
        if s is None:
            return "?"
        return _pct(s.slo_attainment) + (f" ({s.failed} failed)" if s.failed else "")

    rule = f"SLO: {slo} for {target} of requests"
    if cap.reason == "start_failed":
        return (
            f"No capacity found: even at {load(cap.first_fail)}, only "
            f"{met(cap.first_fail)} met the SLO. {rule}."
        )
    if cap.reason == "max_reached":
        return (
            f"Capacity: at least {load(cap.capacity)} ({met(cap.capacity)} met the SLO; "
            f"the search stopped at --max). {rule}."
        )
    return (
        f"Capacity: {load(cap.capacity)}. At {cap.capacity:g}, {met(cap.capacity)} met "
        f"the SLO; at {cap.first_fail:g}, only {met(cap.first_fail)}. {rule}."
    )


def warnings_for(run: RunResult) -> list[str]:
    out = []
    reasoning = [lv for lv in run.levels if lv.summary.reasoning_share]
    if reasoning:
        lv = reasoning[-1]
        out.append(
            f"Reasoning model: {lv.summary.reasoning_share * 100:.0f}% of output chunks "
            f"at {lv.label} were reasoning tokens. TTFT counts the first reasoning token; "
            f"the first answer token came at p50 {_ms(lv.summary.answer_start.p50)} ms."
        )
    sources = {k for lv in run.levels for k in lv.summary.token_sources}
    if "chunks" in sources:
        out.append(
            "Some token counts are chunk counts (server sent no `usage` and no "
            "--tokenizer was given); TPOT and tok/s may be off if a chunk holds "
            "several tokens."
        )
    cap = run.capacity
    if (
        cap
        and run.levels
        and (run.levels[0].summary.slo_ttft_ms, run.levels[0].summary.slo_tpot_ms)
        != (cap.slo_ttft_ms, cap.slo_tpot_ms)
    ):
        out.append(
            "The capacity was found with different SLOs than this report uses; the "
            "'in SLO' figures are rescored, the capacity is not. Rerun find-capacity."
        )
    for lv in run.levels:
        if lv.client_lag_p99_ms is not None and lv.client_lag_p99_ms > 10:
            out.append(
                f"{lv.label}: client event-loop lag p99 {lv.client_lag_p99_ms:.0f} ms; "
                "the client may be limiting the results."
            )
        if lv.summary.errors:
            errs = ", ".join(f"{k} x{v}" for k, v in lv.summary.errors.items())
            out.append(f"{lv.label}: errors: {errs}")
        if stall := find_stall(lv):
            index, worst, median = stall
            out.append(
                f"{lv.label}: request {index} stalled {worst:.1f} s between two tokens "
                f"(typical gap {median * 1000:.0f} ms). That is the server pausing "
                "(GPU/driver power state, another app on the GPU, memory swapping); "
                "this level's averages include it. Rerun to confirm."
            )
    small_levels = [lv.label for lv in run.levels if 0 < lv.summary.completed < 100]
    if small_levels:
        out.append(
            f"Small sample size at {', '.join(small_levels)} (<100 requests): "
            "p99 metrics approximate the sample maximum. Run >=100 requests for "
            "statistically confident p99s."
        )
    return out


def find_stall(lv: LevelResult) -> tuple[int, float, float] | None:
    """(request index, worst gap, median gap) if one token gap is far beyond normal: over 5 s
    and over 50x the median. Seen for real: Ollama paused 288 s mid-answer on a laptop GPU
    (its own log said 2280 ms/token for that request vs 5 ms for the rest)."""
    worst, index = 0.0, -1
    gaps: list[float] = []
    for r in lv.requests:
        if r.ok and r.itls:
            gaps += r.itls
            if (m := max(r.itls)) > worst:
                worst, index = m, r.index
    if not gaps:
        return None
    median = percentile(sorted(gaps), 50)
    if worst > 5.0 and worst > 50 * median:
        return index, worst, median
    return None


def print_run(run: RunResult, console: Console | None = None) -> None:
    console = console or Console()
    c = run.config
    # no borders and two-line headers keep it inside 80 characters; only headers may wrap,
    # numbers never get cut to "108…" or split across lines
    table = Table(
        title=f"{run.label}  ({c.model} @ {c.url})",
        title_justify="left",
        box=None,
        pad_edge=False,
        header_style="bold underline",
    )
    for col in TERMINAL_COLUMNS:
        table.add_column(col, justify="left" if col == "load" else "right", no_wrap=True)
    for lv in run.levels:
        row = level_row(lv)
        table.add_row(*(v for i, v in enumerate(row) if i not in _TERMINAL_SKIP))
    console.print(table)

    # Server-Side Timings breakdown if present
    timing_levels = [
        lv
        for lv in run.levels
        if (
            lv.summary.server_prefill_ms.count > 0 or lv.summary.queue_network_overhead_ms.count > 0
        )
    ]
    if timing_levels:
        t_table = Table(
            title="Server-Side Timings Breakdown (P11)",
            box=None,
            pad_edge=False,
            header_style="bold underline",
        )
        t_table.add_column("load", justify="left")
        t_table.add_column("Client TTFT\n(p50 ms)", justify="right")
        t_table.add_column("Server Prefill\n(p50 ms)", justify="right")
        t_table.add_column("Server Decode\n(p50 ms)", justify="right")
        t_table.add_column("Queue / Net\n(p50 ms)", justify="right")
        t_table.add_column("Queue Share\n(%)", justify="right")
        for lv in timing_levels:
            s = lv.summary
            c_ttft = f"{s.ttft.p50 * 1000:.1f}" if s.ttft.p50 is not None else "-"
            s_pre = f"{s.server_prefill_ms.p50:.1f}" if s.server_prefill_ms.p50 is not None else "-"
            s_dec = f"{s.server_decode_ms.p50:.1f}" if s.server_decode_ms.p50 is not None else "-"
            q_ovh = (
                f"{s.queue_network_overhead_ms.p50:.1f}"
                if s.queue_network_overhead_ms.p50 is not None
                else "-"
            )
            if s.ttft.p50 and s.queue_network_overhead_ms.p50 is not None and s.ttft.p50 > 0:
                q_pct = f"{(s.queue_network_overhead_ms.p50 / (s.ttft.p50 * 1000.0)) * 100:.0f}%"
            else:
                q_pct = "-"
            t_table.add_row(lv.label, c_ttft, s_pre, s_dec, q_ovh, q_pct)
        console.print()
        console.print(t_table)

    # GPU Telemetry table if present
    gpu_levels = [lv for lv in run.levels if lv.gpu_telemetry]
    if gpu_levels:
        gpu_table = Table(
            title="GPU Telemetry & Energy Efficiency",
            box=None,
            pad_edge=False,
            header_style="bold underline",
        )
        gpu_table.add_column("load", justify="left")
        gpu_table.add_column("GPU Util\n(avg/peak)", justify="right")
        gpu_table.add_column("VRAM MB\n(avg/peak)", justify="right")
        gpu_table.add_column("Power W\n(avg/peak)", justify="right")
        gpu_table.add_column("Temp\n(avg/peak)", justify="right")
        gpu_table.add_column("Tok/Joule\n(efficiency)", justify="right")
        for lv in gpu_levels:
            gt = lv.gpu_telemetry
            if gt:
                u_str = (
                    f"{gt.avg_gpu_util:.0f}%/{gt.peak_gpu_util:.0f}%"
                    if gt.avg_gpu_util is not None
                    else "-"
                )
                vr_str = (
                    f"{gt.avg_vram_used_mb:.0f}/{gt.peak_vram_used_mb:.0f}"
                    if gt.avg_vram_used_mb is not None
                    else "-"
                )
                p_str = (
                    f"{gt.avg_power_w:.1f}W/{gt.peak_power_w:.1f}W"
                    if gt.avg_power_w is not None
                    else "-"
                )
                t_str = (
                    f"{gt.avg_temp_c:.0f}C/{gt.peak_temp_c:.0f}C"
                    if gt.avg_temp_c is not None
                    else "-"
                )
                eff_str = f"{gt.tokens_per_joule:.2f}" if gt.tokens_per_joule is not None else "-"
                gpu_table.add_row(lv.label, u_str, vr_str, p_str, t_str, eff_str)
        console.print()
        console.print(gpu_table)

    # Prometheus server metrics if present
    prom_levels = [lv for lv in run.levels if lv.prometheus_metrics]
    if prom_levels:
        p_table = Table(
            title="Server Prometheus Telemetry",
            box=None,
            pad_edge=False,
            header_style="bold underline",
        )
        p_table.add_column("load", justify="left")
        p_table.add_column("Waiting Reqs", justify="right")
        p_table.add_column("Running Reqs", justify="right")
        p_table.add_column("KV Cache Util", justify="right")
        p_table.add_column("Prompt Tok Delta", justify="right")
        p_table.add_column("Gen Tok Delta", justify="right")
        for lv in prom_levels:
            pm = lv.prometheus_metrics or {}
            w = f"{pm.get('num_requests_waiting', 0):.0f}"
            r = f"{pm.get('num_requests_running', 0):.0f}"
            kv = f"{pm.get('gpu_cache_usage', 0) * 100:.1f}%" if "gpu_cache_usage" in pm else "-"
            pt = f"{pm.get('prompt_tokens_total', 0):.0f}" if "prompt_tokens_total" in pm else "-"
            gt = (
                f"{pm.get('generation_tokens_total', 0):.0f}"
                if "generation_tokens_total" in pm
                else "-"
            )
            p_table.add_row(lv.label, w, r, kv, pt, gt)
        console.print()
        console.print(p_table)

    # Bottleneck Diagnosis if present
    if run.bottleneck_diagnosis:
        bd = run.bottleneck_diagnosis
        findings = bd.get("findings", [])
        primary = bd.get("primary_bottleneck", "Balanced")
        health = bd.get("overall_health", "good")
        health_color = (
            "green"
            if health in ("excellent", "good")
            else ("yellow" if health == "degraded" else "red")
        )
        console.print()
        console.print(
            f"[bold {health_color}]Bottleneck Diagnosis: {primary}[/bold {health_color}] "
            f"(Health: {health})"
        )
        for f in findings:
            sev = f.get("severity", "info")
            if f.get("category") == "Balanced":
                console.print(f"  [dim]• {f.get('headline')}[/dim]")
            else:
                c_tag = "bold red" if sev == "high" else ("yellow" if sev == "medium" else "cyan")
                console.print(f"  [{c_tag}]• {f.get('headline')}[/{c_tag}]")
                for ev in f.get("evidence", []):
                    console.print(f"    [dim]- {ev}[/dim]")
                for rec in f.get("recommendations", []):
                    console.print(f"    [green]-> Recommendation: {rec}[/green]")

    if cap := capacity_text(run):
        console.print()
        console.print(f"[bold]{cap}[/bold]")
    if run.levels:
        console.print(
            f"[dim]Latencies in ms unless noted. Goodput = req/s meeting "
            f"{slo_text(run.levels[0].summary)}.[/dim]"
        )
    for w in warnings_for(run):
        console.print(f"[yellow]! {w}[/yellow]")


# ---------- charts ----------


def _style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["ink2"], labelsize=9)
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(theme["ink"])


def _x_values(lv: LevelResult, inf_x: float) -> float:
    return inf_x if math.isinf(lv.value) else lv.value


def _tick_labels(xs: list[float], inf_x: float | None, log_x: bool) -> list[str]:
    """Label every tick, except on a log axis where a value sits too close to the previous
    labelled one (find-capacity's 8, 9, 10 would overlap); its tick mark stays."""
    out: list[str] = []
    last: float | None = None
    for x in xs:
        if x == inf_x:
            out.append("inf")
        elif log_x and last is not None and x / last < 1.2:
            out.append("")
        else:
            out.append(f"{x:g}")
            last = x
    return out


PANELS = [
    ("TTFT p50 (ms)", lambda s: s.ttft.p50, 1000),
    ("TPOT p50 (ms)", lambda s: s.tpot.p50, 1000),
    ("Output throughput (tok/s)", lambda s: s.output_throughput, 1),
    ("E2E latency p50 (s)", lambda s: s.e2e.p50, 1),
]


def sweep_figure(runs: list[RunResult], theme_name: str = "light"):
    theme = THEMES[theme_name]
    mode = runs[0].config.mode
    finite = [lv.value for r in runs for lv in r.levels if not math.isinf(lv.value)]
    inf_x = (max(finite) * 2) if finite else 1.0
    xs_all = sorted({_x_values(lv, inf_x) for r in runs for lv in r.levels})
    log_x = len(xs_all) > 2 and xs_all[0] > 0 and xs_all[-1] / xs_all[0] >= 4

    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=110)
    fig.patch.set_facecolor(theme["surface"])
    direct = len(runs) <= 4
    for ax, (title, get, scale) in zip(axes.flat, PANELS, strict=True):
        _style_axes(ax, theme)
        ax.set_title(title, fontsize=11, loc="left")
        for i, run in enumerate(runs):
            color = theme["series"][i]
            pts = [(_x_values(lv, inf_x), get(lv.summary)) for lv in run.levels]
            pts = [(x, y * scale) for x, y in pts if y is not None]
            if not pts:
                continue
            x, y = zip(*pts, strict=True)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.5,
                label=run.label,
            )
            if direct and len(runs) > 1:
                ax.annotate(
                    run.label,
                    (x[-1], y[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=theme["ink2"],
                )
        if log_x:
            ax.set_xscale("log", base=2)
        has_inf = any(math.isinf(lv.value) for r in runs for lv in r.levels)
        ax.set_xticks(xs_all)
        ax.set_xticklabels(_tick_labels(xs_all, inf_x if has_inf else None, log_x))
        ax.set_ylim(bottom=0)
        ax.set_xlabel(
            "concurrent requests" if mode == "concurrency" else "request rate (req/s)",
            color=theme["ink2"],
            fontsize=9,
        )
    if len(runs) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        leg = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(runs), 4),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.0),
        )
        for t in leg.get_texts():
            t.set_color(theme["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def distribution_figure(runs: list[RunResult], theme_name: str = "light"):
    """CDFs of TTFT and TPOT at each run's highest load level."""
    theme = THEMES[theme_name]
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=110)
    fig.patch.set_facecolor(theme["surface"])
    for ax, (title, get) in zip(
        axes, [("TTFT (ms)", lambda r: r.ttft), ("TPOT (ms)", lambda r: r.tpot)], strict=True
    ):
        _style_axes(ax, theme)
        ax.set_title(f"{title}, cumulative share of requests", fontsize=11, loc="left")
        for i, run in enumerate(runs):
            if not run.levels:
                continue
            lv = run.levels[-1]
            vals = sorted(v * 1000 for r in lv.requests if r.ok and (v := get(r)) is not None)
            if len(vals) == 0:
                continue
            frac = [(i + 1) / len(vals) for i in range(len(vals))]
            ax.step(
                vals,
                frac,
                where="post",
                color=theme["series"][i],
                linewidth=2,
                label=f"{run.label} ({lv.label})",
            )
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.5, 0.9, 0.99])
        ax.set_yticklabels(["0", "p50", "p90", "p99"])
        leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
        for t in leg.get_texts():
            t.set_color(theme["ink"])
    fig.tight_layout()
    return fig


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    _pyplot().close(fig)
    return buf.getvalue()


def _check(runs: list[RunResult]) -> None:
    if not runs:
        raise ValueError("no runs given")
    if len(runs) > MAX_RUNS:
        raise ValueError(f"at most {MAX_RUNS} runs per report (got {len(runs)}); split them")
    modes = {r.config.mode for r in runs}
    if len(modes) > 1:
        raise ValueError("cannot compare a concurrency sweep with a rate sweep in one report")


# ---------- markdown / html ----------


def _config_rows(run: RunResult) -> list[tuple[str, str]]:
    c, e = run.config, run.environment
    return [
        ("model", c.model),
        ("url", c.url),
        ("endpoint", c.endpoint),
        ("prompts", f"{c.prompts}, input ~{c.input_len} tok, output {c.output_len} tok"),
        ("ignore_eos", str(c.ignore_eos)),
        ("seed", str(c.seed)),
        ("server", server_text(run)),
        ("machine", machine_text(run)),
        ("client", f"servebench {e.servebench_version}, Python {e.python}, {e.platform}"),
        ("when", e.timestamp),
        ("notes", e.notes or "-"),
    ]


def machine_text(run: RunResult) -> str:
    """'AMD Ryzen 5 6600H, 15.3 GB RAM, NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB'"""
    e = run.environment
    parts = [e.cpu or e.client_cpu or "?"]
    if e.ram_gb:
        parts.append(f"{e.ram_gb:g} GB RAM")
    parts += [g.split(", driver")[0] for g in e.gpus or []]
    where = "" if e.local_server is not False else " (client machine; server is remote)"
    return ", ".join(parts) + where


def server_text(run: RunResult) -> str:
    """'Ollama 0.35.1, qwen2.5:0.5b (494M, Q4_K_M, 100% on GPU)'"""
    e, c = run.environment, run.config
    engine = " ".join(x for x in (e.engine, e.engine_version) if x) or "?"
    info = e.model_info or {}
    facts = [str(info[k]) for k in ("parameters", "quantization") if info.get(k)]
    if info.get("on_gpu"):
        facts.append(f"{info['on_gpu']} on GPU")
    if info.get("total_slots"):
        facts.append(f"{info['total_slots']} slots")
    return f"{engine}, {c.model}" + (f" ({', '.join(facts)})" if facts else "")


def markdown(runs: list[RunResult], image_dir: Path, image_prefix: str = "") -> str:
    _check(runs)
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "sweep.png").write_bytes(_png(sweep_figure(runs)))
    (image_dir / "distribution.png").write_bytes(_png(distribution_figure(runs)))
    lines = ["# servebench report", ""]
    lines += [
        f"![Latency and throughput vs load]({image_prefix}sweep.png)",
        "",
        f"![TTFT and TPOT distributions]({image_prefix}distribution.png)",
        "",
    ]
    for run in runs:
        lines += [f"## {run.label}", ""]
        if cap := capacity_text(run):
            lines += [f"**{cap}**", ""]
        lines += ["| | |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in _config_rows(run)]
        lines += ["", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(level_row(lv)) + " |" for lv in run.levels]
        if run.levels:
            lines += [
                "",
                f"Latencies in ms unless noted. Goodput = req/s meeting "
                f"{slo_text(run.levels[0].summary)}.",
            ]
        for w in warnings_for(run):
            lines.append(f"> {w}")
        lines.append("")
    return "\n".join(lines)


_CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df;
        --card:#f4f3f0; --warn:#8a5a00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark; --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#33322f;
  --card:#232321; --warn:#e0a542; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --line:#33322f; --card:#232321; --warn:#e0a542; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; }
.sub { color:var(--ink2); margin:0 0 24px; }
img { max-width:100%; height:auto; display:block; border-radius:8px; }
figure { margin:0 0 16px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:right;
         white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--ink2); font-weight:600; }
.cfg td { text-align:left; white-space:normal; } .cfg td:first-child { color:var(--ink2); }
.note { color:var(--ink2); font-size:13px; }
.cap { font-weight:600; margin:0 0 12px; }
.warn { color:var(--warn); font-size:13px; margin:4px 0; }
"""


def _picture(fig_fn, runs: list[RunResult], alt: str) -> str:
    light = base64.b64encode(_png(fig_fn(runs, "light"))).decode()
    dark = base64.b64encode(_png(fig_fn(runs, "dark"))).decode()
    return (
        f'<figure><picture><source srcset="data:image/png;base64,{dark}" '
        f'media="(prefers-color-scheme: dark)"><img alt="{html.escape(alt)}" '
        f'src="data:image/png;base64,{light}"></picture></figure>'
    )


def html_report(runs: list[RunResult]) -> str:
    _check(runs)
    esc = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>servebench report</title><style>{_CSS}</style></head><body><main>",
        "<h1>servebench report</h1>",
        f"<p class='sub'>{esc(', '.join(r.label for r in runs))}</p>",
        _picture(
            sweep_figure,
            runs,
            "TTFT, TPOT, output throughput and end-to-end latency against load, one line per run",
        ),
        _picture(
            distribution_figure,
            runs,
            "Cumulative distributions of TTFT and TPOT at the highest load level of each run",
        ),
    ]
    for run in runs:
        parts.append(f"<h2>{esc(run.label)}</h2>")
        if cap := capacity_text(run):
            parts.append(f"<p class='cap'>{esc(cap)}</p>")

        if run.bottleneck_diagnosis:
            bd = run.bottleneck_diagnosis
            primary = esc(bd.get("primary_bottleneck", "Balanced"))
            health = esc(bd.get("overall_health", "good"))
            parts.append(
                "<div style='margin:16px 0; padding:12px 16px; background:var(--card); "
                "border-radius:6px;'>"
            )
            parts.append(f"<strong>Bottleneck Diagnosis: {primary}</strong> (Health: {health})")
            for f in bd.get("findings", []):
                head = esc(f.get("headline", ""))
                parts.append(f"<p style='margin:4px 0;'>• {head}</p>")
                for rec in f.get("recommendations", []):
                    parts.append(
                        "<p style='margin:2px 0 2px 16px; color:#1baf7a; font-size:13px;'>"
                        f"→ Recommendation: {esc(rec)}</p>"
                    )
            parts.append("</div>")

        parts.append("<div class='scroll'><table class='cfg'>")
        parts += [f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in _config_rows(run)]
        parts.append("</table></div><div class='scroll'><table><tr>")
        parts += [f"<th>{esc(c)}</th>" for c in COLUMNS]
        parts.append("</tr>")
        for lv in run.levels:
            parts.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in level_row(lv)) + "</tr>")
        parts.append("</table></div>")

        gpu_levels = [lv for lv in run.levels if lv.gpu_telemetry]
        if gpu_levels:
            parts.append(
                "<h3>GPU Telemetry & Energy Efficiency</h3><div class='scroll'><table><tr>"
            )
            parts.append(
                "<th>load</th><th>GPU Util</th><th>VRAM MB</th>"
                "<th>Power W</th><th>Temp</th><th>Tok/Joule</th></tr>"
            )
            for lv in gpu_levels:
                gt = lv.gpu_telemetry
                if gt:
                    u_s = (
                        f"{gt.avg_gpu_util:.0f}% / {gt.peak_gpu_util:.0f}%"
                        if gt.avg_gpu_util is not None
                        else "-"
                    )
                    v_s = (
                        f"{gt.avg_vram_used_mb:.0f} / {gt.peak_vram_used_mb:.0f}"
                        if gt.avg_vram_used_mb is not None
                        else "-"
                    )
                    p_s = (
                        f"{gt.avg_power_w:.1f}W / {gt.peak_power_w:.1f}W"
                        if gt.avg_power_w is not None
                        else "-"
                    )
                    t_s = f"{gt.avg_temp_c:.0f}°C" if gt.avg_temp_c is not None else "-"
                    e_s = f"{gt.tokens_per_joule:.2f}" if gt.tokens_per_joule is not None else "-"
                    parts.append(
                        f"<tr><td>{esc(lv.label)}</td><td>{u_s}</td><td>{v_s}</td>"
                        f"<td>{p_s}</td><td>{t_s}</td><td>{e_s}</td></tr>"
                    )
            parts.append("</table></div>")

        timing_levels = [
            lv
            for lv in run.levels
            if (
                lv.summary.server_prefill_ms.count > 0
                or lv.summary.queue_network_overhead_ms.count > 0
            )
        ]
        if timing_levels:
            parts.append("<h3>Server-Side Timings Breakdown</h3><div class='scroll'><table><tr>")
            parts.append(
                "<th>load</th><th>Client TTFT (p50 ms)</th><th>Server Prefill (p50 ms)</th>"
                "<th>Server Decode (p50 ms)</th><th>Queue / Net (p50 ms)</th></tr>"
            )
            for lv in timing_levels:
                s = lv.summary
                c_ttft = f"{s.ttft.p50 * 1000:.1f}" if s.ttft.p50 is not None else "-"
                s_pre = (
                    f"{s.server_prefill_ms.p50:.1f}" if s.server_prefill_ms.p50 is not None else "-"
                )
                s_dec = (
                    f"{s.server_decode_ms.p50:.1f}" if s.server_decode_ms.p50 is not None else "-"
                )
                q_ovh = (
                    f"{s.queue_network_overhead_ms.p50:.1f}"
                    if s.queue_network_overhead_ms.p50 is not None
                    else "-"
                )
                parts.append(
                    f"<tr><td>{esc(lv.label)}</td><td>{c_ttft}</td><td>{s_pre}</td>"
                    f"<td>{s_dec}</td><td>{q_ovh}</td></tr>"
                )
            parts.append("</table></div>")

        if run.levels:
            parts.append(
                f"<p class='note'>Latencies in ms unless noted. Goodput = req/s "
                f"meeting {esc(slo_text(run.levels[0].summary))}.</p>"
            )
        parts += [f"<p class='warn'>! {esc(w)}</p>" for w in warnings_for(run)]
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_report(runs: list[RunResult], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".html", ".htm"):
        out.write_text(html_report(runs), encoding="utf-8")
    elif out.suffix.lower() == ".md":
        out.write_text(
            markdown(runs, out.parent / f"{out.stem}_files", f"{out.stem}_files/"), encoding="utf-8"
        )
    else:
        raise ValueError("report output must end in .html or .md")
    return out
