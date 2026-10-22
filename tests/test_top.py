"""Tests for live TUI monitor and layout rendering."""

import pytest

from servebench.top import (
    TopState,
    generate_top_layout,
    render_progress_bar,
    render_sparkline,
    run_top_monitor,
)


def test_render_sparkline():
    assert len(render_sparkline([])) == 24
    assert len(render_sparkline([1.0, 2.0, 3.0], length=10)) == 10
    spark = render_sparkline([10.0, 50.0, 100.0], length=3)
    assert len(spark) == 3


def test_render_progress_bar():
    b0 = render_progress_bar(0.0, width=10)
    assert b0 == "[░░░░░░░░░░]"

    b100 = render_progress_bar(1.0, width=10)
    assert b100 == "[██████████]"

    b50 = render_progress_bar(0.5, width=10)
    assert b50 == "[█████░░░░░]"


def test_generate_top_layout():
    state = TopState(
        url="http://127.0.0.1:11434/v1",
        model="qwen2.5:0.5b",
        engine="Ollama",
        gpu_name="RTX 3050 Laptop",
        gpu_util_pct=65.0,
        vram_used_mb=3500.0,
        vram_total_mb=6144.0,
        power_w=58.0,
        temp_c=62.0,
        current_tok_s=240.0,
        tokens_per_joule=4.14,
        req_running=4,
        req_waiting=1,
    )

    layout = generate_top_layout(state)
    assert layout is not None
    assert layout["header"] is not None
    assert layout["body"]["telemetry"] is not None
    assert layout["body"]["metrics"] is not None


@pytest.mark.asyncio
async def test_run_top_monitor_headless():
    # Test headless frame execution without blocking
    final_state = await run_top_monitor(
        url="http://127.0.0.1:11434/v1",
        refresh_rate_s=0.01,
        headless_frames=2,
    )
    assert final_state is not None
    assert final_state.url == "http://127.0.0.1:11434/v1"
