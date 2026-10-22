"""Real-Time Interactive Terminal UI Dashboard (`servebench top`).

Monitors local or remote LLM servers in real time using rich.live:
- Live GPU Wattage, VRAM usage, Temperature, and Tokens/Joule via NVML
- Active request concurrency and queue depth
- Real-time rolling throughput sparklines
- Server-side Prometheus metrics reconciliation (cache hit rate, waiting queue)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .detect import identify
from .prometheus import extract_standard_metrics, resolve_metrics_url, scrape_prometheus
from .telemetry import NvmlHandle

SPARK_CHARS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def render_sparkline(values: list[float], length: int = 24) -> str:
    """Render a sequence of floats as an ASCII sparkline."""
    if not values:
        return " " * length
    recent = list(values)[-length:]
    if len(recent) < length:
        recent = [0.0] * (length - len(recent)) + recent

    min_v = min(recent)
    max_v = max(recent)
    span = max_v - min_v if max_v > min_v else 1.0

    chars = []
    for v in recent:
        idx = int(((v - min_v) / span) * (len(SPARK_CHARS) - 1))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def render_progress_bar(fraction: float, width: int = 16) -> str:
    """Render a compact bracketed progress bar: [██████░░░░░░]."""
    f = max(0.0, min(1.0, fraction))
    filled = int(round(f * width))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


@dataclass
class TopState:
    url: str
    model: str = "auto"
    engine: str = "detecting..."
    start_time: float = field(default_factory=time.time)
    fps: float = 2.0
    history_tok_s: deque[float] = field(default_factory=lambda: deque(maxlen=40))
    history_power_w: deque[float] = field(default_factory=lambda: deque(maxlen=40))
    history_vram_mb: deque[float] = field(default_factory=lambda: deque(maxlen=40))

    # Current telemetry
    gpu_name: str = "GPU Unavailable"
    gpu_util_pct: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 1.0
    power_w: float = 0.0
    temp_c: float = 0.0
    current_tok_s: float = 0.0
    tokens_per_joule: float = 0.0

    # Server Prometheus state
    req_running: int = 0
    req_waiting: int = 0
    kv_cache_usage_pct: float = 0.0
    cache_hit_rate_pct: float = 0.0
    total_tokens_generated: int = 0


def generate_top_layout(state: TopState) -> Layout:
    """Build the multi-panel Rich Layout for servebench top."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="telemetry", ratio=1),
        Layout(name="metrics", ratio=1),
    )

    elapsed_s = int(time.time() - state.start_time)
    elapsed_str = f"{elapsed_s // 60:02d}:{elapsed_s % 60:02d}"

    # Header
    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left", ratio=1)
    header_table.add_column(justify="right", ratio=1)
    header_table.add_row(
        f"[bold cyan]⚡ servebench top[/bold cyan] [dim]•[/dim] "
        f"[bold white]{state.url}[/bold white] [dim]({state.engine} | {state.model})[/dim]",
        f"[dim]Uptime:[/dim] [green]{elapsed_str}[/green] [dim]•[/dim] "
        f"[dim]Press [bold white]Ctrl+C / Q[/bold white] to exit[/dim]",
    )
    layout["header"].update(Panel(header_table, style="bold blue"))

    # Hardware & Telemetry Panel
    vram_pct = (state.vram_used_mb / state.vram_total_mb) if state.vram_total_mb > 0 else 0.0
    vram_bar = render_progress_bar(vram_pct, width=14)
    gpu_bar = render_progress_bar(state.gpu_util_pct / 100.0, width=14)

    telem_table = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    telem_table.add_column("Key", style="dim", width=14)
    telem_table.add_column("Value", style="bold white")

    telem_table.add_row("Device", f"[cyan]{state.gpu_name}[/cyan]")
    telem_table.add_row("GPU Load", f"{gpu_bar} {state.gpu_util_pct:5.1f}%")
    used_gb = state.vram_used_mb / 1024
    total_gb = state.vram_total_mb / 1024
    telem_table.add_row(
        "VRAM Usage",
        f"{vram_bar} {used_gb:.2f} / {total_gb:.2f} GB ({vram_pct * 100:.1f}%)",
    )
    telem_table.add_row(
        "Power Draw",
        f"[yellow]{state.power_w:5.1f}W[/yellow] [dim]•[/dim] [red]{state.temp_c:.0f}°C[/red]",
    )
    telem_table.add_row(
        "Efficiency", f"[bold green]{state.tokens_per_joule:5.2f} Tokens / Joule[/bold green]"
    )
    telem_table.add_row(
        "Power Curve",
        f"[yellow]{render_sparkline(list(state.history_power_w), length=20)}[/yellow]",
    )

    layout["telemetry"].update(
        Panel(telem_table, title="[bold]Hardware & GPU Telemetry[/bold]", border_style="cyan")
    )

    # Performance & Serving Metrics Panel
    spark_tok = render_sparkline(list(state.history_tok_s), length=20)
    perf_table = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    perf_table.add_column("Key", style="dim", width=16)
    perf_table.add_column("Value", style="bold white")

    perf_table.add_row("Throughput", f"[bold green]{state.current_tok_s:6.1f} tok/s[/bold green]")
    perf_table.add_row("Throughput Curve", f"[green]{spark_tok}[/green]")
    perf_table.add_row(
        "Requests Active",
        f"[cyan]{state.req_running}[/cyan] running [dim]•[/dim] "
        f"[magenta]{state.req_waiting}[/magenta] queued",
    )
    cache_bar = render_progress_bar(state.kv_cache_usage_pct / 100.0, width=12)
    perf_table.add_row(
        "KV Cache Usage",
        f"{cache_bar} {state.kv_cache_usage_pct:.1f}%",
    )
    perf_table.add_row(
        "Cache Hit Rate", f"[cyan]{state.cache_hit_rate_pct:5.1f}%[/cyan] prefix reuse"
    )
    perf_table.add_row("Tokens Total", f"[white]{state.total_tokens_generated:,}[/white]")

    layout["metrics"].update(
        Panel(perf_table, title="[bold]Serving Throughput & Queue[/bold]", border_style="green")
    )

    # Footer
    footer_text = Text.from_markup(
        " [bold cyan]servebench[/bold cyan] [dim]v1.1[/dim]  │  "
        "[dim]Keys: [/dim][bold white]q[/bold white] [dim]quit[/dim]  •  "
        "[bold white]s[/bold white] [dim]save snapshot[/dim]  •  "
        "[bold white]r[/bold white] [dim]reset stats[/dim]"
    )
    layout["footer"].update(Panel(Align.center(footer_text), style="dim"))

    return layout


async def run_top_monitor(
    url: str,
    refresh_rate_s: float = 0.5,
    max_duration_s: float | None = None,
    inline: bool = False,
    headless_frames: int | None = None,
) -> TopState:
    """Run the live TUI dashboard event loop."""
    console = Console()
    state = TopState(url=url)

    # Initial server detection
    try:
        detected = identify(url)
        if detected:
            state.engine = detected.engine
            state.model = detected.models[0] if detected.models else "unknown"
    except Exception:
        state.engine = "generic"

    prom_url = resolve_metrics_url(url, "auto")

    frame_count = 0
    prev_tok_counter: float | None = None
    prev_time: float = time.time()

    with Live(
        generate_top_layout(state),
        console=console,
        refresh_per_second=int(1.0 / refresh_rate_s),
        screen=(headless_frames is None and not inline),
    ) as live:
        try:
            while True:
                now = time.time()
                dt = max(0.001, now - prev_time)

                # 1. Collect GPU Telemetry
                if NvmlHandle.is_available():
                    sample = NvmlHandle.sample(0)
                    if sample:
                        state.gpu_name = NvmlHandle.device_name(0) or state.gpu_name
                        state.gpu_util_pct = float(sample.gpu_util)
                        state.vram_used_mb = float(sample.vram_used_mb)
                        state.vram_total_mb = float(sample.vram_total_mb)
                        state.power_w = float(sample.power_w or 0.0)
                        state.temp_c = float(sample.temp_c or 0.0)

                        state.history_power_w.append(state.power_w)
                        state.history_vram_mb.append(state.vram_used_mb)

                # 2. Collect Prometheus Metrics (if server exposes /metrics)
                if prom_url:
                    try:
                        raw = await scrape_prometheus(prom_url)
                        if raw:
                            metrics = extract_standard_metrics(None, raw)
                            state.req_running = int(metrics.get("num_requests_running") or 0)
                            state.req_waiting = int(metrics.get("num_requests_waiting") or 0)
                            state.kv_cache_usage_pct = (
                                float(metrics.get("gpu_cache_usage") or 0.0) * 100.0
                            )

                            gen_tokens = raw.get("vllm:generation_tokens_total") or raw.get(
                                "llamacpp:tokens_predicted_total"
                            )
                            if gen_tokens is not None:
                                state.total_tokens_generated = int(gen_tokens)
                                if prev_tok_counter is not None:
                                    delta_tokens = max(0.0, gen_tokens - prev_tok_counter)
                                    state.current_tok_s = delta_tokens / dt
                                    state.history_tok_s.append(state.current_tok_s)
                                    if state.power_w > 0:
                                        state.tokens_per_joule = state.current_tok_s / state.power_w
                                prev_tok_counter = gen_tokens
                    except Exception:
                        pass

                prev_time = now
                live.update(generate_top_layout(state))

                frame_count += 1
                if headless_frames is not None and frame_count >= headless_frames:
                    break

                if max_duration_s is not None and (now - state.start_time) >= max_duration_s:
                    break

                await asyncio.sleep(refresh_rate_s)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

    return state
