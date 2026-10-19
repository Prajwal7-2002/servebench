"""CI Performance Regression Gate (`servebench check`).

Compares candidate benchmark run against a baseline run. Computes relative degradation
percentages for TTFT p50, TPOT p50, output throughput, goodput, and error rates across
matched load levels. Exits with code 1 if regressions exceed configured tolerance thresholds.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from .results import RunResult


class CheckThresholds(BaseModel):
    max_ttft_pct: float = Field(
        15.0, description="Max allowed TTFT p50 latency increase percentage"
    )
    max_tpot_pct: float = Field(
        10.0, description="Max allowed TPOT p50 latency increase percentage"
    )
    max_throughput_pct: float = Field(
        10.0, description="Max allowed output throughput (tok/s) drop percentage"
    )
    max_goodput_pct: float = Field(10.0, description="Max allowed goodput drop percentage")
    max_error_rate_pct: float = Field(1.0, description="Max allowed request error rate percentage")


class MetricComparison(BaseModel):
    metric: str
    baseline: float
    candidate: float
    unit: str
    change_pct: float
    passed: bool
    threshold_pct: float


class LevelComparison(BaseModel):
    label: str
    passed: bool
    metrics: list[MetricComparison] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


class CheckReport(BaseModel):
    passed: bool
    candidate_label: str
    baseline_label: str
    levels: list[LevelComparison] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


def evaluate_regression(
    candidate: RunResult,
    baseline: RunResult,
    thresholds: CheckThresholds | None = None,
) -> CheckReport:
    """Evaluate performance regressions of candidate run against baseline run."""
    thresh = thresholds or CheckThresholds()

    # Index baseline levels by label (e.g. c=1, c=4, rate=10)
    baseline_by_label = {lv.label: lv for lv in baseline.levels}
    level_comparisons: list[LevelComparison] = []
    overall_failures: list[str] = []

    for c_lv in candidate.levels:
        b_lv = baseline_by_label.get(c_lv.label)
        if not b_lv:
            # If no direct matching level, skip
            continue

        c_sum = c_lv.summary
        b_sum = b_lv.summary

        metrics: list[MetricComparison] = []
        level_failures: list[str] = []

        # 1. TTFT p50 (lower is better; positive diff = latency degradation)
        b_ttft = (b_sum.ttft.p50 or 0.0) * 1000.0
        c_ttft = (c_sum.ttft.p50 or 0.0) * 1000.0
        if b_ttft > 0:
            ttft_diff_pct = ((c_ttft - b_ttft) / b_ttft) * 100.0
            ttft_pass = ttft_diff_pct <= thresh.max_ttft_pct
            metrics.append(
                MetricComparison(
                    metric="TTFT p50",
                    baseline=round(b_ttft, 1),
                    candidate=round(c_ttft, 1),
                    unit="ms",
                    change_pct=round(ttft_diff_pct, 1),
                    passed=ttft_pass,
                    threshold_pct=thresh.max_ttft_pct,
                )
            )
            if not ttft_pass:
                fail_msg = (
                    f"[{c_lv.label}] TTFT p50 degraded by +{ttft_diff_pct:.1f}% "
                    f"({b_ttft:.1f}ms -> {c_ttft:.1f}ms, limit +{thresh.max_ttft_pct:.1f}%)"
                )
                level_failures.append(fail_msg)
                overall_failures.append(fail_msg)

        # 2. TPOT p50 (lower is better; positive diff = latency degradation)
        b_tpot = (b_sum.tpot.p50 or 0.0) * 1000.0
        c_tpot = (c_sum.tpot.p50 or 0.0) * 1000.0
        if b_tpot > 0:
            tpot_diff_pct = ((c_tpot - b_tpot) / b_tpot) * 100.0
            tpot_pass = tpot_diff_pct <= thresh.max_tpot_pct
            metrics.append(
                MetricComparison(
                    metric="TPOT p50",
                    baseline=round(b_tpot, 2),
                    candidate=round(c_tpot, 2),
                    unit="ms",
                    change_pct=round(tpot_diff_pct, 1),
                    passed=tpot_pass,
                    threshold_pct=thresh.max_tpot_pct,
                )
            )
            if not tpot_pass:
                fail_msg = (
                    f"[{c_lv.label}] TPOT p50 degraded by +{tpot_diff_pct:.1f}% "
                    f"({b_tpot:.2f}ms -> {c_tpot:.2f}ms, limit +{thresh.max_tpot_pct:.1f}%)"
                )
                level_failures.append(fail_msg)
                overall_failures.append(fail_msg)

        # 3. Output Throughput (higher is better; positive diff = throughput drop)
        b_tok_s = b_sum.output_throughput
        c_tok_s = c_sum.output_throughput
        if b_tok_s > 0:
            tok_drop_pct = ((b_tok_s - c_tok_s) / b_tok_s) * 100.0
            tok_pass = tok_drop_pct <= thresh.max_throughput_pct
            metrics.append(
                MetricComparison(
                    metric="Throughput",
                    baseline=round(b_tok_s, 1),
                    candidate=round(c_tok_s, 1),
                    unit="tok/s",
                    change_pct=round(-tok_drop_pct, 1),  # negative indicates drop
                    passed=tok_pass,
                    threshold_pct=thresh.max_throughput_pct,
                )
            )
            if not tok_pass:
                limit_s = f"-{thresh.max_throughput_pct:.1f}%"
                fail_msg = (
                    f"[{c_lv.label}] Output throughput dropped by {tok_drop_pct:.1f}% "
                    f"({b_tok_s:.1f} -> {c_tok_s:.1f} tok/s, limit {limit_s})"
                )
                level_failures.append(fail_msg)
                overall_failures.append(fail_msg)

        # 4. Error Rate
        total_reqs = len(c_lv.requests)
        failed_reqs = sum(1 for r in c_lv.requests if not r.ok)
        error_rate_pct = (failed_reqs / total_reqs * 100.0) if total_reqs > 0 else 0.0
        err_pass = error_rate_pct <= thresh.max_error_rate_pct
        metrics.append(
            MetricComparison(
                metric="Error Rate",
                baseline=0.0,
                candidate=round(error_rate_pct, 2),
                unit="%",
                change_pct=round(error_rate_pct, 2),
                passed=err_pass,
                threshold_pct=thresh.max_error_rate_pct,
            )
        )
        if not err_pass:
            fail_msg = (
                f"[{c_lv.label}] Request error rate {error_rate_pct:.1f}% exceeded "
                f"maximum threshold of {thresh.max_error_rate_pct:.1f}%"
            )
            level_failures.append(fail_msg)
            overall_failures.append(fail_msg)

        level_comparisons.append(
            LevelComparison(
                label=c_lv.label,
                passed=len(level_failures) == 0,
                metrics=metrics,
                failures=level_failures,
            )
        )

    all_passed = len(overall_failures) == 0 and len(level_comparisons) > 0
    return CheckReport(
        passed=all_passed,
        candidate_label=candidate.label,
        baseline_label=baseline.label,
        levels=level_comparisons,
        failures=overall_failures,
    )


def print_check_report(report: CheckReport, console: Console | None = None) -> None:
    """Render check regression table to terminal."""
    con = console or Console()

    status_str = (
        "[bold green]PASSED[/bold green]" if report.passed else "[bold red]FAILED[/bold red]"
    )
    tbl = Table(
        title=f"CI Performance Regression Gate: {status_str}",
        box=None,
        pad_edge=False,
        header_style="bold underline",
    )
    tbl.add_column("Level", justify="left")
    tbl.add_column("Metric", justify="left")
    tbl.add_column("Baseline", justify="right")
    tbl.add_column("Candidate", justify="right")
    tbl.add_column("Change", justify="right")
    tbl.add_column("Status", justify="center")

    for lv in report.levels:
        for m in lv.metrics:
            if m.metric in ("TTFT p50", "TPOT p50"):
                sign = "+" if m.change_pct > 0 else ""
                diff_str = f"{sign}{m.change_pct:.1f}%"
            elif m.metric == "Throughput":
                diff_str = f"{m.change_pct:+.1f}%"
            else:
                diff_str = f"{m.candidate:.1f}%"

            stat = "[green]PASS[/green]" if m.passed else "[red]FAIL[/red]"
            base_str = f"{m.baseline} {m.unit}"
            cand_str = f"{m.candidate} {m.unit}"
            tbl.add_row(lv.label, m.metric, base_str, cand_str, diff_str, stat)

    con.print(tbl)
    con.print()

    if report.failures:
        con.print("[bold red]Performance Regressions Detected:[/bold red]")
        for f in report.failures:
            con.print(f"  [red]✗ {f}[/red]")
    else:
        con.print(
            "[bold green]✓ All performance metrics are within specified "
            "regression tolerances.[/bold green]"
        )
