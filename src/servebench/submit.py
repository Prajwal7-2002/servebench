"""`servebench submit`: a standard test whose result can be shared with RunsOnMine.

The payload follows the RunsOnMine result format v1
(https://github.com/Prajwal7-2002/runsonmine/blob/main/docs/format.md). It holds only
hardware model names, engine and model facts, test settings, results and run conditions:
no hostnames, usernames, IP addresses, URLs, prompts or file paths.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .results import LevelResult, RunResult

SCHEMA = "runsonmine/result/v1"
STANDARD_TEST = {
    "name": "standard-v1",
    "users": [1, 2, 4],
    "requests_per_level": 16,
    "input_tokens": 256,
    "output_tokens": 128,
    "warmup_requests": 2,
    "seed": 1,
    "ignore_eos": True,
    "temperature": 0.0,
}

# Only these engine-reported model facts leave the machine.
_MODEL_FIELDS = ("parameter_count", "parameters", "quantization", "weights_gb", "on_gpu")


def power_source() -> str | None:
    """'ac', 'battery', or None if unknown (desktops without a battery report 'ac')."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            s = _Status()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
                return None
            return {0: "battery", 1: "ac"}.get(s.ACLineStatus)
        if sys.platform == "darwin":
            out = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
            ).stdout
            if "AC Power" in out:
                return "ac"
            if "Battery Power" in out:
                return "battery"
            return None
        supplies = list(Path("/sys/class/power_supply").glob("*"))
        mains = [
            p
            for p in supplies
            if (p / "type").exists() and (p / "type").read_text().strip() == "Mains"
        ]
        if mains:
            online = any((p / "online").read_text().strip() == "1" for p in mains)
            return "ac" if online else "battery"
        if supplies and not any(
            (p / "type").read_text().strip() == "Battery" for p in supplies if (p / "type").exists()
        ):
            return "ac"
        return None
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None


def load_model(url: str, model: str, api_key: str | None = None, timeout: float = 600.0) -> None:
    """Send one 1-token request so the engine loads the model onto a GPU before we look."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions", json.dumps(body).encode(), headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except OSError:
        pass  # the sweep's own warm-up will report real problems


def pick_serving_gpu(nvidia_smi_csv: str) -> int | None:
    """Index of the GPU using the most memory, from `index, memory.used` CSV lines."""
    best: tuple[float, int] | None = None
    for line in nvidia_smi_csv.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            index, used = int(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            continue
        if best is None or used > best[0]:
            best = (used, index)
    return best[1] if best else None


def serving_gpu() -> tuple[int, int]:
    """(index of the GPU holding the loaded model, number of NVIDIA GPUs); (0, 0) if none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0, 0
    count = len([line for line in out.strip().splitlines() if line.strip()])
    index = pick_serving_gpu(out)
    return (index if index is not None else 0), count


def gpu_limits(index: int = 0) -> dict:
    """Max SM clock and enforced power limit from nvidia-smi ({} if unavailable)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=clocks.max.sm,enforced.power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    parts = [p.strip() for p in out.split(",")]
    limits: dict = {}
    if len(parts) == 2:
        for key, raw in zip(("gpu_clock_max_mhz", "gpu_power_limit_w"), parts, strict=True):
            try:
                limits[key] = float(raw)
            except ValueError:
                pass
    return limits


def _parse_gpu(text: str) -> dict:
    """'NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6.0 GB, driver 551.76' -> fields."""
    parts = [p.strip() for p in text.split(",")]
    gpu: dict = {"name": parts[0]}
    for p in parts[1:]:
        if m := re.fullmatch(r"([\d.]+) GB", p):
            gpu["vram_gb"] = float(m.group(1))
        elif p.startswith("driver "):
            gpu["driver"] = p.removeprefix("driver ")
    return gpu


def _ms(x: float | None) -> float | None:
    return round(x * 1000.0, 1) if x is not None else None


def _level(lv: LevelResult) -> dict:
    s = lv.summary
    tpot = s.tpot.p50 if s.tpot else None
    return {
        "users": int(lv.value),
        "ok": s.completed,
        "failed": s.failed,
        "output_tok_s": round(s.output_throughput, 2),
        "per_user_tok_s": round(1.0 / tpot, 2) if tpot else None,
        "ttft_ms_p50": _ms(s.ttft.p50 if s.ttft else None),
        "ttft_ms_p90": _ms(s.ttft.p90 if s.ttft else None),
        "tpot_ms_p50": _ms(tpot),
    }


def _conditions(levels: list[LevelResult], extra: dict) -> dict:
    tel = [lv.gpu_telemetry for lv in levels if lv.gpu_telemetry]

    def avg(attr: str) -> float | None:
        vals = [getattr(t, attr) for t in tel if getattr(t, attr, None) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    def peak(attr: str) -> float | None:
        vals = [getattr(t, attr) for t in tel if getattr(t, attr, None) is not None]
        return round(max(vals), 1) if vals else None

    cond = {
        "power_source": extra.get("power_source"),
        "gpu_clock_avg_mhz": avg("avg_sm_clock_mhz"),
        "gpu_clock_max_mhz": extra.get("gpu_clock_max_mhz"),
        "gpu_power_avg_w": avg("avg_power_w"),
        "gpu_power_limit_w": extra.get("gpu_power_limit_w"),
        "gpu_util_avg_pct": avg("avg_gpu_util"),
        "gpu_temp_max_c": peak("peak_temp_c"),
    }
    return {k: v for k, v in cond.items() if v is not None}


def build_payload(run: RunResult, conditions: dict | None = None, serving_gpu: int = 0) -> dict:
    """RunsOnMine result v1 from a finished standard-test run. Pure: no I/O."""
    env = run.environment
    if not env.local_server:
        raise ValueError(
            "submit needs a server on this machine: the hardware recorded is this computer's"
        )
    info = env.model_info or {}
    gpus = [_parse_gpu(g) for g in (env.gpus or [])]
    hardware: dict = {"gpus": gpus}
    if len(gpus) > 1:  # which GPU served the model and was monitored
        hardware["serving_gpu"] = serving_gpu
    hardware |= {"cpu": env.cpu, "ram_gb": env.ram_gb, "os": env.platform}
    return {
        "schema": SCHEMA,
        "date": datetime.now(UTC).date().isoformat(),
        "tool": {"name": "servebench", "version": __version__},
        "test": STANDARD_TEST,
        "hardware": hardware,
        "engine": {"name": env.engine, "version": env.engine_version},
        "model": {"name": run.config.model} | {k: info[k] for k in _MODEL_FIELDS if k in info},
        "results": [_level(lv) for lv in run.levels],
        "conditions": _conditions(run.levels, conditions or {}),
    }
