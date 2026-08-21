"""Local stand-in for Vercel: serves index.html and routes /api/predict.

Mirrors production routing so the demo can be driven end to end before deploying.

    python dev_server.py          -> http://127.0.0.1:8000
"""

import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env", override=True)

from api.predict import handler as ApiHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
API_PATH = "/api/predict"


class Router(SimpleHTTPRequestHandler):
    """Static by default; delegate the API path to the real serverless handler."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def _delegate(self, verb: str):
        # Reuse the production handler's logic against this live connection.
        bound = ApiHandler.__new__(ApiHandler)
        bound.rfile, bound.wfile = self.rfile, self.wfile
        bound.headers, bound.path = self.headers, self.path
        bound.request_version, bound.client_address = self.request_version, self.client_address
        bound.send_response = self.send_response
        bound.send_header = self.send_header
        bound.end_headers = self.end_headers
        getattr(bound, verb)()

    def do_GET(self):
        if self.path.split("?")[0] == API_PATH:
            return self._delegate("do_GET")
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] == API_PATH:
            return self._delegate("do_POST")
        self.send_error(404)

    def end_headers(self):
        # Never let a browser cache showcase.json between regenerations.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    print(f"serving {ROOT} on http://127.0.0.1:{PORT}  (api at {API_PATH})")
    HTTPServer(("127.0.0.1", PORT), Router).serve_forever()
