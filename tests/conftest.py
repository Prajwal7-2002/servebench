import socket
import threading
import time
from dataclasses import replace

import httpx
import pytest
import uvicorn

from servebench.fake_server import FakeConfig, create_app

DEFAULT = FakeConfig(ttft=0.1, itl=0.02)


class FakeServer:
    def __init__(self):
        self.app = create_app(replace(DEFAULT))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.server = uvicorn.Server(uvicorn.Config(self.app, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]},
                                       daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("fake server did not start")
            time.sleep(0.02)
        self._warm_up()

    def _warm_up(self) -> None:
        # The first request through the app pays one-off costs (lazy imports, FastAPI
        # building its handlers). On a cold CI runner that added ~50 ms to the first test's
        # TTFT. Pay it here, like servebench's own warm-up requests do for real servers.
        body = {"model": "fake-model", "max_tokens": 2, "stream": True,
                "stream_options": {"include_usage": True}}
        for path, extra in (("chat/completions", {"messages": [{"role": "user",
                                                                "content": "warm up"}]}),
                            ("completions", {"prompt": "warm up"})):
            with httpx.stream("POST", f"{self.url}/{path}", json=body | extra, timeout=10) as r:
                r.read()
        self.app.state.stats.reset()

    @property
    def cfg(self) -> FakeConfig:
        return self.app.state.cfg

    @property
    def stats(self):
        return self.app.state.stats

    def reset(self) -> None:
        self.app.state.cfg = replace(DEFAULT)
        self.app.state.stats.reset()

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture(scope="session")
def _server():
    s = FakeServer()
    yield s
    s.stop()


@pytest.fixture
def fake(_server):
    """The shared fake server, reset to default timings for each test."""
    _server.reset()
    yield _server
    _server.reset()
