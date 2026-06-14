"""Flask application exposing the derivative calculator backend.

Run with::

    pip install -r requirements.txt
    python app.py            # development server on http://127.0.0.1:5000

Endpoints
---------
GET  /api/health          service info and supported functions
POST /api/derivative      compute a derivative with step-by-step working

If Flask is not installed, use ``server.py`` instead, which provides the same
endpoints using only the Python standard library.
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from api import handle_differentiate, health

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # Allow a separately-hosted front-end to call the API during development.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/health", methods=["GET"])
def health_endpoint():
    return jsonify(health())


@app.route("/api/derivative", methods=["POST", "OPTIONS"])
def derivative_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True) or {}
    status, body = handle_differentiate(payload)
    return jsonify(body), status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
