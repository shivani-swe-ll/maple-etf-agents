"""Checks the HTTP format for both providers using a local fake server (no internet)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from maple_etf.llm import LLMClient, extract_json


@pytest.fixture
def server():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["path"], seen["body"], seen["headers"] = self.path, body, {k.lower(): v for k, v in self.headers.items()}
            if self.path.endswith("/messages"):
                reply = {"content": [{"type": "text", "text": "hi from anthropic"}],
                         "usage": {"input_tokens": 11, "output_tokens": 3}}
            else:
                reply = {"choices": [{"message": {"content": "hi from openai"}}],
                         "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", seen
    httpd.shutdown()


def test_openai_compatible(server):
    url, seen = server
    client = LLMClient("openai", "test-model", "key123", url)
    resp = client.complete("sys", "hello")
    assert resp.text == "hi from openai"
    assert seen["path"] == "/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer key123"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert client.usage == {"calls": 1, "input_tokens": 9, "output_tokens": 4}


def test_anthropic(server):
    url, seen = server
    client = LLMClient("anthropic", "test-model", "key456", url)
    resp = client.complete("sys", "hello")
    assert resp.text == "hi from anthropic"
    assert seen["path"] == "/messages"
    assert seen["headers"]["x-api-key"] == "key456"
    assert seen["body"]["system"] == "sys"


def test_from_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert LLMClient.from_env() is None
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    c = LLMClient.from_env()
    assert c.base_url == "http://localhost:11434/v1"


def test_extract_json_handles_code_fences():
    assert extract_json('```json\n{"action": "BUY"}\n```') == {"action": "BUY"}
