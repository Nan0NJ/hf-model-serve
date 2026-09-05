from model_server import cli
from model_server.config import ServerConfig


def test_full_test_command_against_tiny_fake_backend(monkeypatch):
    import argparse
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": "ok", "loaded": True})
            else:
                self._send({"object": "list", "data": [{"id": "fake"}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            json.loads(self.rfile.read(length))
            self._send({
                "id": "chatcmpl-fake", "object": "chat.completion", "created": 1, "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Štiri."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            })

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setattr(cli, "_endpoint", lambda: (base, {"served_model_name": "fake"}))
        monkeypatch.setattr(cli, "saved_config", lambda root: ServerConfig(model="fake", served_model_name="fake"))
        assert cli.cmd_test(argparse.Namespace(prompt="2+2?", max_tokens=8)) == 0
    finally:
        server.shutdown()
        server.server_close()


def test_generated_tokens_decoding_empty_is_detected(monkeypatch):
    body = {
        "choices": [{"message": {"content": "   "}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    monkeypatch.setattr(cli, "_request", lambda *a, **k: (body, 0.5))
    monkeypatch.setattr(cli, "saved_config", lambda root: ServerConfig(model="fake"))
    result = cli._one_completion("http://127.0.0.1:8000", {"served_model_name": "fake"}, "hi", 4)
    assert result["empty"] is True
    assert result["generated_but_empty"] is True
    assert result["tokens_per_second"] == 6


def test_request_installs_explicit_no_proxy_handler(monkeypatch):
    observed = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"status":"ok"}'

    class FakeOpener:
        def open(self, request, timeout): return FakeResponse()

    def fake_build(handler):
        observed["proxies"] = handler.proxies
        return FakeOpener()

    monkeypatch.setattr(cli.urllib.request, "build_opener", fake_build)
    body, _ = cli._request("http://127.0.0.1:8000/health")
    assert body == {"status": "ok"}
    assert observed["proxies"] == {}
