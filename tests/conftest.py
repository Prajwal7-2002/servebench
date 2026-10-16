import socket
import threading
import time
from dataclasses import replace

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
