"""Local teacher client: host / licence guards, caching, retries (against a loopback test server)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dsm.data.synth.client import LocalTeacher, TeacherError, is_private_host


@pytest.mark.parametrize(
    ("host", "private"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("10.1.2.3", True),
        ("192.168.0.10", True),
        ("gpu-box", True),
        ("teacher.internal", True),
        ("::1", True),
        ("8.8.8.8", False),
        ("api.example.com", False),
    ],
)
def test_private_hosts(host, private):
    assert is_private_host(host) is private


def test_remote_and_licence_guards():
    with pytest.raises(TeacherError, match="non-local"):
        LocalTeacher("https://api.example.com/v1", "m", license="apache-2.0")
    with pytest.raises(TeacherError, match="allowlist"):
        LocalTeacher("http://127.0.0.1:9/v1", "m", license="llama3")
    with pytest.raises(TeacherError, match="allowlist"):
        LocalTeacher("http://127.0.0.1:9/v1", "m", license=None)
    teacher = LocalTeacher("https://api.example.com/v1", "m", license="MIT", allow_remote=True)
    assert teacher.generated_by == "model:m"


class _Handler(BaseHTTPRequestHandler):
    calls: list[dict] = []
    fail_first = 0

    def do_POST(self):  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(body)
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(500)
            self.end_headers()
            return
        reply = {
            "choices": [
                {"message": {"role": "assistant", "content": "echo: " + body["messages"][-1]["content"]}}
            ]
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def server():
    _Handler.calls = []
    _Handler.fail_first = 0
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_chat_cache_and_map(server, tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:1")  # must be ignored for a local teacher
    cache = tmp_path / "cache.jsonl"
    teacher = LocalTeacher(
        server, "local", license="apache-2.0", cache=cache, generation={"temperature": 0.1}
    )
    messages = [{"role": "user", "content": "hi"}]
    assert teacher.chat(messages) == "echo: hi"
    assert teacher.chat(messages) == "echo: hi"
    assert len(_Handler.calls) == 1 and _Handler.calls[0]["temperature"] == 0.1
    assert teacher.chat(messages, temperature=0.5) == "echo: hi"  # different params -> new request
    assert len(_Handler.calls) == 2
    again = LocalTeacher(server, "local", license="apache-2.0", cache=cache, generation={"temperature": 0.1})
    assert again.chat(messages) == "echo: hi" and len(_Handler.calls) == 2  # served from the disk cache
    answers = teacher.map([[{"role": "user", "content": str(i)}] for i in range(5)], workers=3)
    assert answers == [f"echo: {i}" for i in range(5)]


def test_retries_then_none(server, monkeypatch):
    monkeypatch.setattr("dsm.data.synth.client.time.sleep", lambda s: None)
    _Handler.fail_first = 1
    teacher = LocalTeacher(server, "local", license="mit", max_retries=2)
    assert teacher.chat([{"role": "user", "content": "x"}]) == "echo: x"
    _Handler.fail_first = 5
    assert teacher.chat([{"role": "user", "content": "y"}]) is None
