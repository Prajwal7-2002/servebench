"""The mock server runs in its own process, like a real inference server.

Running it on a thread in the test process made client and server share one GIL: the mock's
precise-timing spin could delay the client reading chunks, so under load ITL came out as
bursts (5 ms instead of 15 ms). Tests change its behaviour per test over a small HTTP
control API (`/_mock/config`, `/_mock/reset`); `server.cfg.ttft = 0.2` still works.
"""

import os
import re
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import pytest

# CLI output in tests must not depend on the runner's terminal: macOS CI drew error messages
# in an 80-column colored box, which wrapped "newer servebench" across two lines.
os.environ["COLUMNS"] = "400"
os.environ["NO_COLOR"] = "1"
os.environ.pop("FORCE_COLOR", None)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")  # terminal color codes
_BOX = re.compile(r"[\u2500-\u257f|+]")  # box-drawing characters, ASCII fallback


def plain(text: str) -> str:
    """CLI output as plain words: no colors, no box borders, wrapped lines joined."""
    return " ".join(_BOX.sub(" ", _ANSI.sub("", text)).split())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Settings:
    """`server.cfg.x = v` sends the change to the mock process."""

    def __init__(self, control: httpx.Client):
        object.__setattr__(self, "_control", control)

    def __setattr__(self, name: str, value) -> None:
        self._control.post("/_mock/config", json={name: value}).raise_for_status()


class _Stats:
    def __init__(self, data: dict):
        self.active, self.peak, self.total = data["active"], data["peak"], data["total"]
        self.connections = data["connections"]


class MockServer:
    def __init__(self):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.url = f"{self.base}/v1"
        # Log to a file, never a pipe nobody reads: when the mock drops a connection on
        # purpose, uvicorn logs a traceback, a full pipe blocks the whole mock process, and
        # every later request times out.
        self.log = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "servebench.cli", "mock-server", "--port", str(self.port)],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 30
        while True:
            if self.proc.poll() is not None:
                self.log.seek(0)
                raise RuntimeError(f"mock server exited: {self.log.read().decode()}")
            try:
                httpx.get(f"{self.url}/models", timeout=1).raise_for_status()
                break
            except httpx.HTTPError:
                if time.time() > deadline:
                    raise RuntimeError("mock server did not start") from None
                time.sleep(0.1)
        # one long-lived client for control calls: a new httpx client per call builds an SSL
        # context each time (~0.1-0.2 s on Windows), which added ~1 s to every test
        self.control = httpx.Client(base_url=self.base, timeout=10)
        self.cfg = _Settings(self.control)
        self._warm_up()

    def _warm_up(self) -> None:
        # The first request through the app pays one-off costs (lazy imports, FastAPI
        # building its handlers). On a cold CI runner that added ~50 ms to the first test's
        # TTFT. Pay it here, like servebench's own warm-up requests do for real servers.
        body = {
            "model": "mock-model",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for path, extra in (
            ("chat/completions", {"messages": [{"role": "user", "content": "warm up"}]}),
            ("completions", {"prompt": "warm up"}),
        ):
            with httpx.stream("POST", f"{self.url}/{path}", json=body | extra, timeout=10) as r:
                r.read()
        self.reset()

    @property
    def stats(self) -> _Stats:
        return _Stats(self.control.get("/stats").json())

    def reset(self) -> None:
        """Default timings (TTFT 100 ms, 20 ms per token) and zeroed stats."""
        self.control.post("/_mock/reset", json={}).raise_for_status()

    def stop(self) -> None:
        self.control.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


@pytest.fixture(scope="session")
def _server():
    s = MockServer()
    yield s
    s.stop()


@pytest.fixture
def server(_server):
    """The shared mock server, reset to default timings for each test."""
    _server.reset()
    yield _server
    _server.reset()
