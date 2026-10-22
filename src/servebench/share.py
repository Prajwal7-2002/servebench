"""`servebench share`: a result people can paste into a post or an issue.

Two outputs from one run.json: a Markdown block (table + capacity + how it was measured,
so others can compare like with like) and a 1200x630 PNG card (the size social sites
preview).
"""

from __future__ import annotations

from pathlib import Path

from . import __version__
from .report import THEMES, _pyplot, machine_text, server_text
from .results import LevelResult, RunResult

REPO = "https://github.com/Prajwal7-2002/servebench"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.0f}"


def _per_user(lv: LevelResult) -> str:
    tpot = lv.summary.tpot.p50
    return f"{1 / tpot:.0f}" if tpot else "-"


def capacity_line(run: RunResult) -> str | None:
    cap = run.capacity
    if cap is None:
        return None
    rule = []
    if cap.slo_ttft_ms is not None:
        rule.append(f"first token < {cap.slo_ttft_ms:g} ms")
    if cap.slo_tpot_ms is not None:
        rule.append(f">= {1000 / cap.slo_tpot_ms:.0f} tok/s each")
    rule_text = " and ".join(rule) + f", {cap.target * 100:.0f}% of requests"
    if cap.reason == "start_failed":
        return f"**Capacity: below 1 user** ({rule_text})"
    if cap.mode == "concurrency":
        unit = "user" if cap.capacity == 1 else "users"
    else:
        unit = "req/s"
    prefix = "at least " if cap.reason == "max_reached" else ""
    return f"**Capacity: {prefix}{cap.capacity:g} {unit}** ({rule_text})"


def markdown(run: RunResult) -> str:
    c = run.config
    load = "Users" if c.mode == "concurrency" else "Req/s offered"
    lines = [
        f"**{server_text(run)}**  ",
        f"Machine: {machine_text(run)}",
        "",
        f"| {load} | Total tok/s | Per user tok/s | First token p50 / p99 (ms) | In target |",
        "|---|---|---|---|---|",
    ]
    for lv in run.levels:
        s = lv.summary
        value = "inf" if lv.value == float("inf") else f"{lv.value:g}"
        in_slo = "-" if s.slo_attainment is None else f"{s.slo_attainment * 100:.0f}%"
        lines.append(
            f"| {value} | {s.output_throughput:.0f} | {_per_user(lv)} | "
            f"{_ms(s.ttft.p50)} / {_ms(s.ttft.p99)} | {in_slo} |"
        )
    if cap := capacity_line(run):
        lines += ["", cap]
    lines += [
        "",
        f"<sub>Measured with [servebench]({REPO}) {run.environment.servebench_version}"
        f": {c.prompts} prompts of ~{c.input_len} tokens, {c.output_len}-token "
        f"answers, {c.endpoint} endpoint.</sub>",
    ]
    return "\n".join(lines)


def card(run: RunResult, path: Path) -> Path:
    """1200x630 PNG: what was measured, the headline numbers, and throughput vs load."""
    plt = _pyplot()
    theme = THEMES["light"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(theme["surface"])
    ink, ink2, accent = theme["ink"], theme["ink2"], theme["series"][0]

    fig.text(0.04, 0.9, server_text(run), fontsize=20, color=ink, weight="bold")
    fig.text(0.04, 0.84, machine_text(run), fontsize=12, color=ink2)

    one = next((lv for lv in run.levels if lv.value == 1 and lv.summary.completed), None)
    cap = run.capacity
    at = next((lv for lv in run.levels if cap and lv.value == cap.capacity), None)
    stats = []
    if cap and cap.reason != "start_failed":
        unit = "users at once" if cap.mode == "concurrency" else "req/s"
        prefix = ">= " if cap.reason == "max_reached" else ""
        stats.append((f"{prefix}{cap.capacity:g}", unit))
    if one:
        stats.append((_per_user(one), "tokens/s for one user"))
        stats.append((_ms(one.summary.ttft.p50), "ms to first token"))
    if at:
        stats.append((f"{at.summary.output_throughput:.0f}", "tokens/s total at capacity"))
    for i, (big, small) in enumerate(stats[:4]):
        y = 0.66 - i * 0.16
        fig.text(0.04, y, big, fontsize=34, color=ink, weight="bold")
        fig.text(0.04, y - 0.045, small, fontsize=12, color=ink2)

    ax = fig.add_axes((0.42, 0.14, 0.54, 0.62))
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=ink2, labelsize=10)
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    pts = sorted(
        (lv.value, lv.summary.output_throughput) for lv in run.levels if lv.value != float("inf")
    )
    if pts:
        xs, ys = zip(*pts, strict=True)
        ax.plot(
            xs,
            ys,
            color=accent,
            linewidth=2.5,
            marker="o",
            markersize=7,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
        )
        if at:
            ax.axvline(at.value, color=ink2, linewidth=1, linestyle="--")
            ax.annotate(
                "capacity",
                (at.value, max(ys)),
                xytext=(6, -4),
                textcoords="offset points",
                fontsize=10,
                color=ink2,
                va="top",
            )
        ax.set_ylim(bottom=0)
    ax.set_title(
        "Total tokens/s vs " + ("users" if run.config.mode == "concurrency" else "offered req/s"),
        loc="left",
        fontsize=12,
        color=ink,
    )
    fig.text(
        0.04,
        0.04,
        f"servebench {__version__}  ·  {REPO.removeprefix('https://')}",
        fontsize=10,
        color=ink2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
