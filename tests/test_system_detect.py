"""Machine info (parsed from sample tool output, so it's tested on every OS) and server
detection (against the mock server and fake engine fingerprints)."""

import json

from servebench import detect, system

NVIDIA = ("NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144, 551.76\n"
          "NVIDIA A100-SXM4-80GB, 81920, 550.54\n")


def test_parse_nvidia_smi():
    assert system.parse_nvidia_smi(NVIDIA) == [
        "NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB, driver 551.76",
        "NVIDIA A100-SXM4-80GB, 80.0 GB, driver 550.54",
    ]
    assert system.parse_nvidia_smi("") == []
    assert system.parse_nvidia_smi("garbage") == []


def test_parse_linux_proc_files():
    assert system.parse_meminfo("MemTotal:       16243756 kB\nMemFree: 1 kB\n") == 15.5
    assert system.parse_meminfo("nothing") is None
    cpuinfo = "processor\t: 0\nmodel name\t: AMD Ryzen 7 7840U w/ Radeon 780M Graphics\n"
    assert system.parse_cpuinfo(cpuinfo) == "AMD Ryzen 7 7840U w/ Radeon 780M Graphics"


def test_probes_never_raise_when_tools_are_missing(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda _: None)
    assert system._run(["nvidia-smi"]) is None
    info = system.describe_for_env()  # must not raise on any OS
    assert set(info) == {"cpu", "ram_gb", "gpus"}


def test_identify_mock_server(server):
    s = detect.identify(server.url)
    assert s.engine == "servebench mock"
    assert s.models == ["mock-model"]
    assert s.url == server.url


def test_identify_nothing_listening():
    assert detect.identify("http://127.0.0.1:9/v1", timeout=0.5) is None


def test_identify_engine_fingerprints(monkeypatch):
    def fake_server(routes):
        def get(url, timeout):  # exact paths, like a real server (others are 404 -> None)
            return routes.get(url.split("127.0.0.1:1", 1)[1])
        return get

    base = "http://127.0.0.1:1/v1"
    models = {"/v1/models": {"data": [{"id": "m"}]}}
    cases = {
        "Ollama": {"/api/version": {"version": "0.35.1"}},
        "llama.cpp": {"/props": {"total_slots": 4, "build_info": "b6000",
                                 "default_generation_settings": {"n_ctx": 4096}}},
        "vLLM": {"/version": {"version": "0.11.0"}},
        "SGLang": {"/get_model_info": {"model_path": "Qwen"}},
        "LM Studio": {"/api/v0/models": {"data": []}},
        "OpenAI-compatible": {},
    }
    for engine, routes in cases.items():
        monkeypatch.setattr(detect, "_get_json", fake_server(models | routes))
        s = detect.identify(base)
        assert s.engine == engine, engine
    monkeypatch.setattr(detect, "_get_json", fake_server(models | cases["llama.cpp"]))
    s = detect.identify(base)
    assert s.version == "b6000" and s.details == {"total_slots": 4, "n_ctx": 4096}


def test_ollama_model_info(monkeypatch):
    ps = {"models": [{"name": "qwen2.5:0.5b", "size": 481810185, "size_vram": 481810185,
                      "context_length": 4096,
                      "details": {"parameter_size": "494.03M", "quantization_level": "Q4_K_M"}}]}
    monkeypatch.setattr(detect, "_get_json", lambda url, timeout: ps)
    s = detect.Server(url="http://127.0.0.1:11434/v1", engine="Ollama")
    assert detect.model_info(s, "qwen2.5:0.5b") == {
        "parameters": "494.03M", "quantization": "Q4_K_M", "context_length": 4096,
        "size_gb": 0.45, "on_gpu": "100%"}
    assert detect.model_info(s, "other") is None


def test_find_local_on_custom_ports(server):
    found = detect.find_local(ports=[(9, "nothing"), (server.port, "mock")], timeout=0.5)
    assert [s.engine for s in found] == ["servebench mock"]


def test_run_records_machine_and_server(server, tmp_path):
    from typer.testing import CliRunner

    from servebench.cli import app
    from servebench.report import load_run, machine_text, server_text

    server.cfg.ttft, server.cfg.itl = 0.01, 0.002
    out = tmp_path / "r.json"
    res = CliRunner().invoke(app, ["run", "--url", server.url, "--model", "mock-model",
                                   "--concurrency", "1", "--num-requests", "2",
                                   "--output-len", "2", "--warmup", "0", "--out", str(out)])
    assert res.exit_code == 0, res.output
    run = load_run(out)
    e = run.environment
    assert e.engine == "servebench mock" and e.local_server is True
    assert e.cpu  # every CI OS can name its CPU
    assert server_text(run).startswith("servebench mock, mock-model")
    assert machine_text(run).startswith(e.cpu)
    json.loads(out.read_text())  # still plain JSON
