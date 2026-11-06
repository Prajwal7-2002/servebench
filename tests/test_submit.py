"""`servebench submit`: standard test, RunsOnMine result v1, and nothing private in it."""

import getpass
import json
import socket

import pytest
from typer.testing import CliRunner

from servebench import submit
from servebench.cli import app


def test_submit_end_to_end_on_mock_server(server, tmp_path):
    server.cfg.ttft, server.cfg.itl = 0.005, 0.0005
    out = tmp_path / "result.json"
    res = CliRunner().invoke(
        app, ["submit", "--url", server.url, "--model", "mock-model", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "runsonmine/result/v1"
    assert payload["test"]["name"] == "standard-v1"
    assert [r["users"] for r in payload["results"]] == [1, 2, 4]
    assert all(r["ok"] == 16 and r["failed"] == 0 for r in payload["results"])
    assert payload["engine"]["name"] == "servebench mock"
    assert payload["model"]["name"] == "mock-model"
    assert set(payload) == {
        "schema",
        "date",
        "tool",
        "test",
        "hardware",
        "engine",
        "model",
        "results",
        "conditions",
    }


def test_submit_payload_contains_nothing_private(server, tmp_path):
    server.cfg.ttft, server.cfg.itl = 0.005, 0.0005
    out = tmp_path / "result.json"
    res = CliRunner().invoke(
        app, ["submit", "--url", server.url, "--model", "mock-model", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    text = out.read_text(encoding="utf-8").lower()
    assert "127.0.0.1" not in text and "http" not in text and str(server.port) not in text
    assert socket.gethostname().lower() not in text
    user = getpass.getuser().lower()
    if len(user) >= 4:  # very short usernames can occur by chance in model names
        assert user not in text
    assert str(tmp_path).lower() not in text


def test_submit_refuses_a_remote_server(monkeypatch):
    from servebench import cli, detect

    remote = detect.Server(url="http://10.0.0.5:8000/v1", engine="vLLM", models=["m"])
    monkeypatch.setattr(cli, "_find_server_and_model", lambda url, model: (remote, "m"))
    res = CliRunner().invoke(app, ["submit", "--url", remote.url])
    assert res.exit_code == 1
    assert "this machine" in res.output


def test_build_payload_requires_a_local_server(server, tmp_path):
    from servebench.report import load_run

    server.cfg.ttft, server.cfg.itl = 0.005, 0.0005
    out = tmp_path / "r.json"
    args = [
        "run",
        "--url",
        server.url,
        "--model",
        "mock-model",
        "--concurrency",
        "1",
        "--num-requests",
        "2",
        "--output-len",
        "2",
        "--warmup",
        "0",
        "--out",
        str(out),
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    run = load_run(out)
    run.environment.local_server = False
    with pytest.raises(ValueError, match="this machine"):
        submit.build_payload(run)


def test_parse_gpu():
    assert submit._parse_gpu("NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB, driver 551.76") == {
        "name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "vram_gb": 6.0,
        "driver": "551.76",
    }
    assert submit._parse_gpu("Apple M3 Max GPU (unified memory)") == {
        "name": "Apple M3 Max GPU (unified memory)"
    }


def test_probes_never_raise():
    assert submit.power_source() in ("ac", "battery", None)
    limits = submit.gpu_limits()
    assert set(limits) <= {"gpu_clock_max_mhz", "gpu_power_limit_w"}
