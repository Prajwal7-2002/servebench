"""Tests for statistical rigor: sample counts, repeat trials, config files, and CSV export."""

import csv
import json

from typer.testing import CliRunner

from servebench.cli import app, load_config_file
from servebench.metrics import stats, summarize, summarize_repeats
from servebench.report import load_run, warnings_for, write_csv, write_requests_csv
from servebench.results import LevelResult, RequestResult, RunConfig, RunResult

runner = CliRunner()


def test_stats_count_and_p99_approx():
    s_empty = stats([])
    assert s_empty.count == 0
    assert not s_empty.p99_is_approx

    s_small = stats([float(i) for i in range(20)])
    assert s_small.count == 20
    assert s_small.p99_is_approx
    assert s_small.max == 19.0

    s_large = stats([float(i) for i in range(120)])
    assert s_large.count == 120
    assert not s_large.p99_is_approx


def test_small_sample_warning():
    reqs = [
        RequestResult(
            index=i, start=0.0, ttft=0.05, e2e=0.1, output_tokens=10, chunk_times=[0.05, 0.1]
        )
        for i in range(10)
    ]
    lv = LevelResult(
        mode="concurrency",
        value=1.0,
        duration=1.0,
        requests=reqs,
        summary=summarize(reqs, 1.0, 500, 50),
    )
    run = RunResult(
        label="test",
        config=RunConfig(url="http://x/v1", model="m", mode="concurrency", levels=[1.0]),
        environment=load_run("results/learn/prompt-16.json").environment,
        levels=[lv],
    )
    warns = warnings_for(run)
    assert any("Small sample size" in w and "<100 requests" in w for w in warns)


def test_summarize_repeats():
    reqs1 = [
        RequestResult(
            index=0, start=0.0, ttft=0.04, e2e=0.1, output_tokens=20, chunk_times=[0.04, 0.1]
        )
    ]
    reqs2 = [
        RequestResult(
            index=0, start=0.0, ttft=0.06, e2e=0.1, output_tokens=22, chunk_times=[0.06, 0.1]
        )
    ]
    s1 = summarize(reqs1, 1.0, 500, 50)
    s2 = summarize(reqs2, 1.0, 500, 50)

    rm = summarize_repeats([s1, s2])
    assert "output_throughput" in rm
    assert rm["output_throughput"].mean == 21.0
    assert rm["output_throughput"].min == 20.0
    assert rm["output_throughput"].max == 22.0
    assert rm["output_throughput"].std > 0
    assert rm["output_throughput"].ci95 is not None
    assert rm["ttft_p50"].mean == 0.05


def test_csv_export(tmp_path):
    run = load_run("results/learn/prompt-16.json")
    csv_file = tmp_path / "test.csv"
    write_csv([run], csv_file)
    assert csv_file.is_file()

    rows = list(csv.reader(csv_file.read_text(encoding="utf-8").splitlines()))
    assert rows[0][0] == "run"
    assert "out_tok_s" in rows[0]
    assert len(rows) == 2  # header + 1 level

    reqs_file = tmp_path / "reqs.csv"
    write_requests_csv(run, reqs_file)
    assert reqs_file.is_file()
    req_rows = list(csv.reader(reqs_file.read_text(encoding="utf-8").splitlines()))
    assert req_rows[0][0] == "level"
    assert len(req_rows) == 1 + len(run.levels[0].requests)


def test_load_config_yaml_json_toml(tmp_path):
    # JSON config
    json_path = tmp_path / "bench.json"
    json_path.write_text(
        json.dumps({"url": "http://127.0.0.1:8000/v1", "model": "test-m", "concurrency": "1,2"})
    )
    cfg_json = load_config_file(json_path)
    assert cfg_json["model"] == "test-m"

    # YAML config
    yaml_path = tmp_path / "bench.yaml"
    yaml_path.write_text(
        "url: http://127.0.0.1:8000/v1\nmodel: yaml-m\nrepeat: 3\nconcurrency: [1, 2]\n"
    )
    cfg_yaml = load_config_file(yaml_path)
    assert cfg_yaml["model"] == "yaml-m"
    assert cfg_yaml["repeat"] == 3

    # TOML config
    toml_path = tmp_path / "bench.toml"
    toml_path.write_text("url = 'http://127.0.0.1:8000/v1'\nmodel = 'toml-m'\nnum_requests = 16\n")
    cfg_toml = load_config_file(toml_path)
    assert cfg_toml["model"] == "toml-m"
    assert cfg_toml["num_requests"] == 16


def test_cli_repeat_and_config(server, tmp_path):
    server.cfg.ttft, server.cfg.itl = 0.01, 0.002

    # Test running with --repeat 2
    out_json = tmp_path / "repeat_run.json"
    out_csv = tmp_path / "repeat_run.csv"
    res = runner.invoke(app, [
        "run", "--url", server.url, "--model", "mock-model",
        "--concurrency", "1", "--num-requests", "4", "--repeat", "2",
        "--warmup", "0", "--out", str(out_json), "--csv", str(out_csv)
    ])
    assert res.exit_code == 0, res.output
    assert "repeats" in res.output
    run = load_run(out_json)
    assert run.config.repeat == 2
    assert run.levels[0].repeat_count == 2
    assert run.levels[0].repeats is not None and len(run.levels[0].repeats) == 2
    assert run.levels[0].repeat_metrics is not None
    assert "output_throughput" in run.levels[0].repeat_metrics
    assert out_csv.is_file()

    # Test running via -c bench.yaml
    bench_yaml = tmp_path / "bench.yaml"
    bench_yaml.write_text(f"""
url: {server.url}
model: mock-model
concurrency: 1
num_requests: 2
warmup: 0
""")
    res_yaml = runner.invoke(app, ["run", "-c", str(bench_yaml)])
    assert res_yaml.exit_code == 0, res_yaml.output
    assert "c=1" in res_yaml.output
