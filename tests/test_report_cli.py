"""M3: run.json round-trips, reports render, and the CLI works end to end."""

import json
import math

import pytest
from typer.testing import CliRunner

from servebench.cli import app, parse_levels
from servebench.metrics import summarize
from servebench.report import html_report, load_run, markdown, save_run, write_report
from servebench.results import Environment, LevelResult, RequestResult, RunConfig, RunResult

runner = CliRunner()


def sample_run(label, mode="concurrency", levels=(1, 2, 4), slow=1.0) -> RunResult:
    lvs = []
    for v in levels:
        reqs = []
        for i in range(10):
            ttft = 0.1 * slow * (1 if math.isinf(v) else v) + 0.001 * i
            e2e = ttft + 0.02 * 31
            reqs.append(RequestResult(index=i, start=0.0, ttft=ttft, e2e=e2e, output_tokens=32,
                                      chunk_times=[ttft + 0.02 * k for k in range(32)]))
        lvs.append(LevelResult(mode=mode, value=v, duration=2.0, requests=reqs,
                               summary=summarize(reqs, 2.0, 500, 50)))
    return RunResult(
        label=label,
        config=RunConfig(url="http://x/v1", model="m", mode=mode, levels=list(levels)),
        environment=Environment(servebench_version="0", python="3", platform="test",
                                timestamp="2026-10-16T00:00:00+00:00"),
        levels=lvs,
    )


def test_run_json_round_trip_with_inf(tmp_path):
    run = sample_run("a", mode="rate", levels=(1, math.inf))
    path = tmp_path / "a.json"
    save_run(run, path)
    json.loads(path.read_text())  # strict JSON, no bare Infinity
    back = load_run(path)
    assert math.isinf(back.levels[1].value)
    assert back.levels[1].label == "rate=inf"
    assert back.levels[0].summary == run.levels[0].summary


def test_markdown_and_html_reports(tmp_path):
    runs = [sample_run("ollama"), sample_run("llamacpp", slow=2)]
    md = markdown(runs, tmp_path / "img", "img/")
    assert (tmp_path / "img" / "sweep.png").stat().st_size > 10_000
    assert "## ollama" in md and "## llamacpp" in md and "| c=4 |" in md
    page = html_report(runs)
    assert page.count("data:image/png;base64,") == 4  # light + dark for two figures
    assert "prefers-color-scheme: dark" in page


def test_report_refuses_mixed_modes(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        write_report([sample_run("a"), sample_run("b", mode="rate")], tmp_path / "r.html")


def test_parse_levels():
    assert parse_levels("1, 2,4") == [1, 2, 4]
    assert parse_levels("0.5,inf") == [0.5, math.inf]


def test_cli_run_and_report_against_mock_server(server, tmp_path):
    server.cfg.ttft, server.cfg.itl = 0.03, 0.005
    out = tmp_path / "mock.json"
    res = runner.invoke(app, [
        "run", "--url", server.url, "--model", "mock-model", "--concurrency", "1,2",
        "--num-requests", "4", "--input-len", "32", "--output-len", "8", "--warmup", "1",
        "--out", str(out),
    ])
    assert res.exit_code == 0, res.output
    run = load_run(out)
    assert run.label == "mock"
    assert [lv.value for lv in run.levels] == [1, 2]
    assert all(lv.summary.completed == 4 for lv in run.levels)
    assert run.environment.server_models == ["mock-model"]
    assert run.levels[0].requests[0].prompt_tokens == 32  # synthetic prompt length

    html = tmp_path / "report.html"
    res = runner.invoke(app, ["report", str(out), str(out), "--out", str(html),
                              "--slo-ttft-ms", "1"])
    assert res.exit_code == 0, res.output
    assert html.exists()


def test_cli_needs_exactly_one_mode():
    res = runner.invoke(app, ["run", "--url", "http://x/v1", "--model", "m"])
    assert res.exit_code != 0


@pytest.mark.parametrize("width", [80, 88])
def test_terminal_table_never_truncates_numbers(width):
    from rich.console import Console

    from servebench.report import print_run

    # big values on purpose: 5-digit latencies and throughput, like a server under overload
    run = sample_run("a-fairly-long-run-label", levels=(1, 16, 128))
    run.levels[-1].summary.output_throughput = 12345.6
    console = Console(width=width, record=True)
    print_run(run, console)
    text = console.export_text()
    assert "…" not in text  # no ellipsis anywhere
    lines = text.splitlines()
    assert any("c=128" in ln and "12345.6" in ln and "12809" in ln for ln in lines)


def test_cli_warns_about_unknown_model(server, tmp_path):
    server.cfg.ttft, server.cfg.itl = 0.01, 0.002
    res = runner.invoke(app, [
        "run", "--url", server.url, "--model", "not-pulled:0.5b", "--concurrency", "1",
        "--num-requests", "1", "--output-len", "2", "--warmup", "0",
    ])
    assert res.exit_code == 0, res.output
    assert "does not list model 'not-pulled:0.5b'" in res.output
    assert "mock-model" in res.output


def test_cli_warns_about_localhost(server):
    server.cfg.ttft, server.cfg.itl = 0.01, 0.002
    url = server.url.replace("127.0.0.1", "localhost")
    res = runner.invoke(app, ["run", "--url", url, "--model", "mock-model", "--concurrency",
                              "1", "--num-requests", "1", "--output-len", "2", "--warmup", "0"])
    assert res.exit_code == 0, res.output
    assert "Use 127.0.0.1" in res.output


def test_tick_labels_skip_crowded_values_on_log_axis():
    from servebench.report import _tick_labels

    assert _tick_labels([1, 2, 4, 8, 9, 10, 12, 16], None, True) == \
        ["1", "2", "4", "8", "", "10", "12", "16"]
    assert _tick_labels([1, 2, 3], None, False) == ["1", "2", "3"]
    assert _tick_labels([1, 2, 4], 4, True) == ["1", "2", "inf"]


def test_report_missing_file_is_a_clean_error(tmp_path):
    res = runner.invoke(app, ["report", str(tmp_path / "nope.json")])
    assert res.exit_code != 0
    assert "does not exist" in res.output
