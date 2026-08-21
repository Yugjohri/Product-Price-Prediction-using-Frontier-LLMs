"""WSGI entrypoint for the deployed demo.

Vercel's Python runtime looks for a single application object in one of a few
default filenames (app.py, index.py, server.py, main.py, wsgi.py, asgi.py). The
older "each file under api/ is its own function" convention is no longer
detected, so this module is the one entrypoint and routes internally.

Still zero-dependency: stdlib only, no framework. All request handling logic
lives in api/predict.py; this file only adapts it to WSGI and serves the two
static assets.
"""

import json
import mimetypes
from pathlib import Path

from api.predict import DEFAULT_MODEL, MODELS, handle

ROOT = Path(__file__).resolve().parent

# The only files the demo serves. An explicit allowlist rather than a directory
# walk, so no request can reach anything else in the deployment.
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/showcase.json": ("showcase.json", "application/json"),
    "/favicon.ico": (None, None),  # answered 204 rather than 404 noise
}

API_PATH = "/api/predict"


def _json(start_response, status_code, payload, extra_headers=()):
    body = json.dumps(payload).encode("utf-8")
    status = {
        200: "200 OK",
        204: "204 No Content",
        400: "400 Bad Request",
        401: "401 Unauthorized",
        404: "404 Not Found",
        405: "405 Method Not Allowed",
        429: "429 Too Many Requests",
        502: "502 Bad Gateway",
    }.get(status_code, f"{status_code} Status")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    headers.extend(extra_headers)
    start_response(status, headers)
    return [body]


def _model_registry():
    return {
        "models": [
            {"id": k, "label": v["label"], "tier": v["tier"], "provider": v["provider"]}
            for k, v in MODELS.items()
        ],
        "default": DEFAULT_MODEL,
    }


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/") or "/"
    method = environ.get("REQUEST_METHOD", "GET").upper()

    # ---- API ----
    if path == API_PATH:
        if method == "GET":
            return _json(start_response, 200, _model_registry())
        if method != "POST":
            return _json(start_response, 405, {"error": "method_not_allowed"})
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
            raw = environ["wsgi.input"].read(length) if length else b"{}"
            payload = json.loads(raw or b"{}")
        except Exception:
            return _json(start_response, 400, {"error": "bad_json", "message": "Malformed request."})
        status_code, body = handle(payload)
        return _json(start_response, status_code, body)

    # ---- static ----
    entry = STATIC.get(path)
    if entry is None:
        return _json(start_response, 404, {"error": "not_found"})

    filename, content_type = entry
    if filename is None:
        start_response("204 No Content", [("Content-Length", "0")])
        return [b""]

    file_path = ROOT / filename
    try:
        data = file_path.read_bytes()
    except OSError:
        return _json(start_response, 404, {"error": "not_found"})

    if not content_type:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    start_response(
        "200 OK",
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(data))),
            # The page and its data change together on redeploy; don't let a
            # browser pair a new page with a stale showcase.
            ("Cache-Control", "no-cache"),
        ],
    )
    return [data]


# Vercel looks for a module-level callable; some runtimes look for `handler`.
handler = app
