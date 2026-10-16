"""Terminal tables, charts and Markdown/HTML reports for one or more runs."""

from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .metrics import summarize
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
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                   "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#33322f",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767"],
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
        raise RunFileError(f"{path} was written by a newer servebench (file format "
                           f"{version}, this version reads up to {SCHEMA_VERSION}); upgrade "
                           "servebench to read it")
    try:
        return RunResult.model_validate_json(text)
    except ValidationError as e:
        raise RunFileError(f"{path} is not a valid servebench run file: "
                           f"{e.error_count()} problem(s), first: {e.errors()[0]['msg']} "
                           f"at {'.'.join(map(str, e.errors()[0]['loc']))}") from e


def save_run(run: RunResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def resummarize(run: RunResult, slo_ttft_ms: float | None, slo_tpot_ms: float | None) -> None:
    """Recompute summaries from raw requests, e.g. to apply different SLOs in a report."""
    for lv in run.levels:
        lv.summary = summarize(lv.requests, lv.duration, slo_ttft_ms, slo_tpot_ms)


# ---------- formatting ----------

def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.0f}" if x * 1000 >= 10 else f"{x * 1000:.1f}"


def _num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


COLUMNS = ["load", "ok/fail", "req/s", "out tok/s", "TTFT p50", "TTFT p99", "TPOT p50",
           "TPOT p99", "ITL p50", "E2E p50 (s)", "goodput", "tok/chunk", "in SLO"]
# The terminal table drops ITL p50 (equal to TPOT at one token per chunk) and "in SLO" (the
# run progress lines show it); both stay in run.json and the MD/HTML reports. That keeps it
# inside 80 columns even with 5-digit latencies.
TERMINAL_COLUMNS = ["load", "ok/\nfail", "req/s", "out\ntok/s", "TTFT\np50", "TTFT\np99",
                    "TPOT\np50", "TPOT\np99", "E2E\np50 s", "good-\nput", "tok/\nchunk"]
_TERMINAL_SKIP = {COLUMNS.index("ITL p50"), COLUMNS.index("in SLO")}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def level_row(lv: LevelResult) -> list[str]:
    s: Summary = lv.summary
    return [
        lv.label, f"{s.completed}/{s.failed}", _num(s.request_throughput, 2),
        _num(s.output_throughput), _ms(s.ttft.p50), _ms(s.ttft.p99), _ms(s.tpot.p50),
        _ms(s.tpot.p99), _ms(s.itl.p50), _num(s.e2e.p50, 2), _num(s.goodput, 2),
        _num(s.tokens_per_chunk, 2), _pct(s.slo_attainment),
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
        return (f"No capacity found: even at {load(cap.first_fail)}, only "
                f"{met(cap.first_fail)} met the SLO. {rule}.")
    if cap.reason == "max_reached":
        return (f"Capacity: at least {load(cap.capacity)} ({met(cap.capacity)} met the SLO; "
                f"the search stopped at --max). {rule}.")
    return (f"Capacity: {load(cap.capacity)}. At {cap.capacity:g}, {met(cap.capacity)} met "
            f"the SLO; at {cap.first_fail:g}, only {met(cap.first_fail)}. {rule}.")


def warnings_for(run: RunResult) -> list[str]:
    out = []
    reasoning = [lv for lv in run.levels if lv.summary.reasoning_share]
    if reasoning:
        lv = reasoning[-1]
        out.append(f"Reasoning model: {lv.summary.reasoning_share * 100:.0f}% of output chunks "
                   f"at {lv.label} were reasoning tokens. TTFT counts the first reasoning token; "
                   f"the first answer token came at p50 {_ms(lv.summary.answer_start.p50)} ms.")
    sources = {k for lv in run.levels for k in lv.summary.token_sources}
    if "chunks" in sources:
        out.append("Some token counts are chunk counts (server sent no `usage` and no "
                   "--tokenizer was given); TPOT and tok/s may be off if a chunk holds "
                   "several tokens.")
    cap = run.capacity
    if cap and run.levels and (run.levels[0].summary.slo_ttft_ms, run.levels[0].summary.slo_tpot_ms
                               ) != (cap.slo_ttft_ms, cap.slo_tpot_ms):
        out.append("The capacity was found with different SLOs than this report uses; the "
                   "'in SLO' figures are rescored, the capacity is not. Rerun find-capacity.")
    for lv in run.levels:
        if lv.client_lag_p99_ms is not None and lv.client_lag_p99_ms > 10:
            out.append(f"{lv.label}: client event-loop lag p99 {lv.client_lag_p99_ms:.0f} ms; "
                       "the client may be limiting the results.")
        if lv.summary.errors:
            errs = ", ".join(f"{k} x{v}" for k, v in lv.summary.errors.items())
            out.append(f"{lv.label}: errors: {errs}")
    return out


def print_run(run: RunResult, console: Console | None = None) -> None:
    console = console or Console()
    c = run.config
    # no borders and two-line headers keep it inside 80 characters; only headers may wrap,
    # numbers never get cut to "108…" or split across lines
    table = Table(title=f"{run.label}  ({c.model} @ {c.url})", title_justify="left",
                  box=None, pad_edge=False, header_style="bold underline")
    for col in TERMINAL_COLUMNS:
        table.add_column(col, justify="left" if col == "load" else "right", no_wrap=True)
    for lv in run.levels:
        row = level_row(lv)
        table.add_row(*(v for i, v in enumerate(row) if i not in _TERMINAL_SKIP))
    console.print(table)
    if cap := capacity_text(run):
        console.print(f"[bold]{cap}[/bold]")
    if run.levels:
        console.print(f"[dim]Latencies in ms unless noted. Goodput = req/s meeting "
                      f"{slo_text(run.levels[0].summary)}.[/dim]")
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
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=theme["surface"], markeredgewidth=1.5, label=run.label)
            if direct and len(runs) > 1:
                ax.annotate(run.label, (x[-1], y[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color=theme["ink2"])
        if log_x:
            ax.set_xscale("log", base=2)
        has_inf = any(math.isinf(lv.value) for r in runs for lv in r.levels)
        ax.set_xticks(xs_all)
        ax.set_xticklabels(_tick_labels(xs_all, inf_x if has_inf else None, log_x))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("concurrent requests" if mode == "concurrency" else "request rate (req/s)",
                      color=theme["ink2"], fontsize=9)
    if len(runs) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="upper center", ncol=min(len(runs), 4),
                         frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
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
    for ax, (title, get) in zip(axes, [("TTFT (ms)", lambda r: r.ttft),
                                       ("TPOT (ms)", lambda r: r.tpot)], strict=True):
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
            ax.step(vals, frac, where="post", color=theme["series"][i], linewidth=2,
                    label=f"{run.label} ({lv.label})")
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
        ("model", c.model), ("url", c.url), ("endpoint", c.endpoint),
        ("prompts", f"{c.prompts}, input ~{c.input_len} tok, output {c.output_len} tok"),
        ("ignore_eos", str(c.ignore_eos)), ("seed", str(c.seed)),
        ("client", f"{e.platform}, Python {e.python}, {e.client_cpu or '?'}"),
        ("when", e.timestamp), ("notes", e.notes or "-"),
    ]


def markdown(runs: list[RunResult], image_dir: Path, image_prefix: str = "") -> str:
    _check(runs)
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "sweep.png").write_bytes(_png(sweep_figure(runs)))
    (image_dir / "distribution.png").write_bytes(_png(distribution_figure(runs)))
    lines = ["# servebench report", ""]
    lines += [f"![Latency and throughput vs load]({image_prefix}sweep.png)", "",
              f"![TTFT and TPOT distributions]({image_prefix}distribution.png)", ""]
    for run in runs:
        lines += [f"## {run.label}", ""]
        if cap := capacity_text(run):
            lines += [f"**{cap}**", ""]
        lines += ["| | |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in _config_rows(run)]
        lines += ["", "| " + " | ".join(COLUMNS) + " |",
                  "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(level_row(lv)) + " |" for lv in run.levels]
        if run.levels:
            lines += ["", f"Latencies in ms unless noted. Goodput = req/s meeting "
                          f"{slo_text(run.levels[0].summary)}."]
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
    return (f'<figure><picture><source srcset="data:image/png;base64,{dark}" '
            f'media="(prefers-color-scheme: dark)"><img alt="{html.escape(alt)}" '
            f'src="data:image/png;base64,{light}"></picture></figure>')


def html_report(runs: list[RunResult]) -> str:
    _check(runs)
    esc = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>servebench report</title><style>{_CSS}</style></head><body><main>",
        "<h1>servebench report</h1>",
        f"<p class='sub'>{esc(', '.join(r.label for r in runs))}</p>",
        _picture(sweep_figure, runs, "TTFT, TPOT, output throughput and end-to-end latency "
                 "against load, one line per run"),
        _picture(distribution_figure, runs, "Cumulative distributions of TTFT and TPOT at "
                 "the highest load level of each run"),
    ]
    for run in runs:
        parts.append(f"<h2>{esc(run.label)}</h2>")
        if cap := capacity_text(run):
            parts.append(f"<p class='cap'>{esc(cap)}</p>")
        parts.append("<div class='scroll'><table class='cfg'>")
        parts += [f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in _config_rows(run)]
        parts.append("</table></div><div class='scroll'><table><tr>")
        parts += [f"<th>{esc(c)}</th>" for c in COLUMNS]
        parts.append("</tr>")
        for lv in run.levels:
            parts.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in level_row(lv)) + "</tr>")
        parts.append("</table></div>")
        if run.levels:
            parts.append(f"<p class='note'>Latencies in ms unless noted. Goodput = req/s "
                         f"meeting {esc(slo_text(run.levels[0].summary))}.</p>")
        parts += [f"<p class='warn'>! {esc(w)}</p>" for w in warnings_for(run)]
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_report(runs: list[RunResult], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".html", ".htm"):
        out.write_text(html_report(runs), encoding="utf-8")
    elif out.suffix.lower() == ".md":
        out.write_text(markdown(runs, out.parent / f"{out.stem}_files", f"{out.stem}_files/"),
                       encoding="utf-8")
    else:
        raise ValueError("report output must end in .html or .md")
    return out
