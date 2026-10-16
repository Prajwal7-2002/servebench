"""Find local LLM servers and identify the engine behind an OpenAI-compatible URL.

Each engine has its own fingerprint next to the OpenAI API (`/api/version` for Ollama,
`/props` for llama.cpp, `/version` for vLLM, ...). Detection only reads small JSON
endpoints with short timeouts, in parallel, so `servebench quick` starts in about a second.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# Default ports of the servers local users run. Order = preference when several are up.
KNOWN_PORTS: list[tuple[int, str]] = [
    (11434, "Ollama"),
    (8080, "llama.cpp"),
    (1234, "LM Studio"),
    (8000, "vLLM"),
    (30000, "SGLang"),
]


@dataclass
class Server:
    url: str  # OpenAI base URL, e.g. http://127.0.0.1:11434/v1
    engine: str  # "Ollama", "llama.cpp", ..., or "OpenAI-compatible" if unknown
    version: str | None = None
    models: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)  # engine extras, e.g. llama.cpp slots


def _get_json(url: str, timeout: float) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _root(base_url: str) -> str:
    """http://host:port/v1 -> http://host:port"""
    u = urlparse(base_url)
    return f"{u.scheme}://{u.netloc}"


def identify(base_url: str, timeout: float = 1.0) -> Server | None:
    """Which engine serves this OpenAI base URL? None if nothing answers /models."""
    base_url = base_url.rstrip("/")
    listing = _get_json(f"{base_url}/models", timeout)
    if not isinstance(listing, dict):
        return None
    models = [m.get("id", "?") for m in listing.get("data", []) if isinstance(m, dict)]
    root = _root(base_url)
    server = Server(url=base_url, engine="OpenAI-compatible", models=models)

    if isinstance(v := _get_json(f"{root}/api/version", timeout), dict) and "version" in v:
        server.engine, server.version = "Ollama", str(v["version"])
    elif isinstance(p := _get_json(f"{root}/props", timeout), dict) and (
            "default_generation_settings" in p or "total_slots" in p):
        server.engine = "llama.cpp"
        server.version = p.get("build_info")
        server.details = {k: p[k] for k in ("total_slots", "model_path") if k in p}
        n_ctx = (p.get("default_generation_settings") or {}).get("n_ctx")
        if n_ctx:
            server.details["n_ctx"] = n_ctx
    elif isinstance(v := _get_json(f"{root}/version", timeout), dict) and "version" in v:
        server.engine, server.version = "vLLM", str(v["version"])
    elif isinstance(_get_json(f"{root}/get_model_info", timeout), dict):
        server.engine = "SGLang"
    elif isinstance(_get_json(f"{root}/api/v0/models", timeout), dict):
        server.engine = "LM Studio"
    elif isinstance(s := _get_json(f"{root}/stats", timeout), dict) and "connections" in s:
        server.engine = "servebench mock"
    return server


def ollama_installed() -> bool:
    """Ollama on PATH, or in its default Windows/macOS install location."""
    if shutil.which("ollama"):
        return True
    candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
                  Path("/Applications/Ollama.app")]
    return any(p.exists() for p in candidates if str(p) not in ("", "."))


def find_local(host: str = "127.0.0.1", ports: list[tuple[int, str]] | None = None,
               timeout: float = 1.0) -> list[Server]:
    """Every server answering on the usual local ports, in preference order."""
    ports = ports or KNOWN_PORTS
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        found = list(pool.map(lambda p: identify(f"http://{host}:{p[0]}/v1", timeout), ports))
    return [s for s in found if s is not None]


def model_info(server: Server, model: str, timeout: float = 2.0) -> dict | None:
    """Engine-specific facts about the model, read after a run (when it's loaded)."""
    if server.engine != "Ollama":
        return dict(server.details) or None
    ps = _get_json(f"{_root(server.url)}/api/ps", timeout)
    if not isinstance(ps, dict):
        return None
    for m in ps.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            d = m.get("details") or {}
            return {
                "parameters": d.get("parameter_size"),
                "quantization": d.get("quantization_level"),
                "context_length": m.get("context_length"),
                "size_gb": round(size / 2**30, 2) if size else None,
                "on_gpu": f"{vram / size * 100:.0f}%" if size else None,
            }
    return None
