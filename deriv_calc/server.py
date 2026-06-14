"""Dependency-free HTTP server for the derivative calculator.

This uses only the Python standard library, so the backend runs without
installing anything::

    python server.py            # serves on http://127.0.0.1:5000

It exposes the same endpoints as the Flask app (``app.py``):

    GET  /api/health
    POST /api/derivative
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from api import handle_differentiate, health


class Handler(BaseHTTPRequestHandler):
    server_version = "deriv_calc/1.0"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # noqa: N802 (stdlib naming)
        self._send(204, {})

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/api/health":
            self._send(200, health())
        else:
            self._send(404, {"ok": False, "error": "Not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/api/derivative":
            self._send(404, {"ok": False, "error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "Invalid JSON body."})
            return
        status, body = handle_differentiate(payload)
        self._send(status, body)

    def log_message(self, fmt, *args):  # keep the console quiet by default
        pass


def main(host: str = "127.0.0.1", port: int = 5000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"deriv_calc backend listening on http://{host}:{port}")
    print("  GET  /api/health")
    print("  POST /api/derivative   body: {\"expression\": \"x^2*sin(x)\"}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
