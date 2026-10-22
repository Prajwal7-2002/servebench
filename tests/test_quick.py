"""`servebench quick`: picking a model, the plain-language summary, stall warnings, and the
command end to end against the mock server."""

from typer.testing import CliRunner

from servebench import detect, quick
from servebench.cli import app
from servebench.metrics import summarize
from servebench.report import find_stall, load_run
from servebench.results import (
    CapacityResult,
    Environment,
    LevelResult,
    RequestResult,
    RunConfig,
    RunResult,
)


def level(value, ttft, e2e, n=16, tokens=128):
    reqs = [
        RequestResult(
            index=i,
            start=0,
            ttft=ttft,
            e2e=e2e,
            output_tokens=tokens,
            chunk_times=[ttft + k * (e2e - ttft) / (tokens - 1) for k in range(tokens)],
        )
        for i in range(n)
    ]
    return LevelResult(
        mode="concurrency",
        value=value,
        duration=n * e2e / value,
        requests=reqs,
        summary=summarize(reqs, n * e2e / value, 500, 50),
    )


def run_with(levels, capacity, first_fail, reason="converged", engine="Ollama"):
    return RunResult(
        label="x",
        config=RunConfig(url="http://127.0.0.1:11434/v1", model="m"),
        environment=Environment(
            servebench_version="0", python="3", platform="t", timestamp="t", engine=engine
        ),
        levels=levels,
        capacity=CapacityResult(
            mode="concurrency",
            target=0.9,
            slo_ttft_ms=500,
            slo_tpot_ms=50,
            capacity=capacity,
            first_fail=first_fail,
            reason=reason,
        ),
    )


def test_summary_explains_queueing_and_gives_the_ollama_tip():
    # one slot: at 2 users the second waits for the whole first answer
    lines = quick.summary(run_with([level(1, 0.05, 0.67), level(2, 0.69, 1.3)], 1, 2))
    text = "\n".join(lines)
    assert "One user:    first token in 50 ms, then 205 tokens/s" in text
    assert "Capacity:    1 user at once (first token < 500 ms and >= 20 tokens/s each" in text
    assert "requests queue: the server works on about 1 at once" in text
    assert "OLLAMA_NUM_PARALLEL=4" in text


def test_summary_without_queueing_has_no_tip():
    # TTFT at the failing level is small compared to an answer: slower, not queued
    lines = quick.summary(
        run_with(
            [level(1, 0.05, 0.67), level(4, 0.1, 1.3), level(5, 0.15, 1.5)], 4, 5, engine="vLLM"
        )
    )
    text = "\n".join(lines)
    assert "Capacity:    4 users at once" in text
    assert "queue" not in text and "OLLAMA" not in text


def test_summary_start_failed_and_max_reached():
    failed = "\n".join(quick.summary(run_with([level(1, 0.9, 30.0)], None, 1, "start_failed")))
    assert "even 1 user misses the target" in failed and "--slo-ttft-ms 2000" in failed
    maxed = "\n".join(
        quick.summary(run_with([level(1, 0.05, 0.6), level(32, 0.2, 1.0)], 32, None, "max_reached"))
    )
    assert "Capacity:    at least 32 users at once" in maxed


def test_pick_model_and_default_out(monkeypatch):
    s = detect.Server(
        url="http://127.0.0.1:8080/v1",
        engine="llama.cpp",
        models=["nomic-embed-text", "qwen2.5-0.5b"],
    )
    assert quick.pick_model(s) == "qwen2.5-0.5b"  # embedding models can't chat
    ollama = detect.Server(
        url="http://127.0.0.1:11434/v1", engine="Ollama", models=["llama3:latest", "qwen2.5:0.5b"]
    )
    monkeypatch.setattr(detect, "_get_json", lambda url, t: {"models": [{"name": "qwen2.5:0.5b"}]})
    assert quick.pick_model(ollama) == "qwen2.5:0.5b"  # already loaded: no load wait
    assert (
        quick.default_out("Ollama", "qwen2.5:0.5b") == "servebench-quick-ollama-qwen2.5-0.5b.json"
    )


def test_find_stall():
    normal = level(1, 0.05, 0.67)
    assert find_stall(normal) is None
    stalled = level(1, 0.05, 0.67)
    r = stalled.requests[13]
    r.chunk_times = r.chunk_times[:60] + [t + 288.0 for t in r.chunk_times[60:]]
    index, worst, median = find_stall(stalled)
    assert index == 13 and 287 < worst < 289 and median < 0.01


def test_quick_end_to_end_on_mock_server(server, tmp_path):
    # TTFT = 40 ms x requests in flight; with a 150 ms target, 3 users pass and 4 fail
    server.cfg.ttft, server.cfg.itl, server.cfg.slowdown = 0.04, 0.002, 1.0
    out = tmp_path / "q.json"
    res = CliRunner().invoke(
        app,
        [
            "quick",
            "--url",
            server.url,
            "--max",
            "8",
            "--slo-ttft-ms",
            "150",
            "--slo-tpot-ms",
            "1000",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "servebench quick: servebench mock" in res.output
    assert "One user:" in res.output and "Capacity:    3 users at once" in res.output
    run = load_run(out)
    assert run.capacity.capacity == 3 and run.environment.engine == "servebench mock"


def test_quick_with_nothing_running(monkeypatch):
    monkeypatch.setattr(detect, "find_local", lambda **k: [])
    monkeypatch.setattr(detect, "ollama_installed", lambda: False)
    res = CliRunner().invoke(app, ["quick"])
    assert res.exit_code == 1
    assert "No local LLM server found" in res.output and "ollama serve" in res.output


def test_share_markdown_and_card(tmp_path):
    from servebench import share

    r = run_with([level(1, 0.05, 0.67), level(4, 0.1, 1.3), level(5, 0.7, 1.6)], 4, 5)
    md = share.markdown(r)
    assert "| Users | Total tok/s | Per user tok/s |" in md
    assert "| 4 |" in md and "**Capacity: 4 users**" in md
    assert "servebench" in md and "synthetic prompts of ~256 tokens" in md
    path = share.card(r, tmp_path / "card.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 20_000


def test_share_command(tmp_path):
    from servebench.report import save_run

    f = tmp_path / "r.json"
    save_run(run_with([level(1, 0.05, 0.67), level(2, 0.7, 1.3)], 1, 2), f)
    res = CliRunner().invoke(app, ["share", str(f)])
    assert res.exit_code == 0, res.output
    assert "**Capacity: 1 user** (" in res.output
    assert (tmp_path / "r-card.png").exists()
    res = CliRunner().invoke(app, ["share", str(f), "--no-card", "--card", str(tmp_path / "x.png")])
    assert res.exit_code == 0 and not (tmp_path / "x.png").exists()


def test_quick_says_when_ollama_is_installed_but_not_running(monkeypatch):
    monkeypatch.setattr(detect, "find_local", lambda **k: [])
    monkeypatch.setattr(detect, "ollama_installed", lambda: True)
    res = CliRunner().invoke(app, ["quick"])
    assert res.exit_code == 1 and "Ollama is installed but not running" in res.output
