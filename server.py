#!/usr/bin/env python
"""
Standalone WebUI server for Orthrus training metrics.

Usage:
    python webui/server.py                        # serves metrics.jsonl from ./checkpoints
    python webui/server.py --file path/to/metrics.jsonl --port 8080
    python webui/server.py --file path/to/logs/   # auto-finds metrics.jsonl in dir

Open http://localhost:8080 in browser.
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_metrics_path = None
_html_cache = None


def _load_html():
    """Load and cache the HTML template from this package's directory."""
    global _html_cache
    if _html_cache is not None:
        return _html_cache
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Template not found: {html_path}")
    _html_cache = html_path.read_text(encoding="utf-8")
    return _html_cache


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            data = []
            if _metrics_path and os.path.exists(_metrics_path):
                with open(_metrics_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_load_html().encode("utf-8"))

    def log_message(self, format, *args):
        pass


def main():
    global _metrics_path

    parser = argparse.ArgumentParser(description="Orthrus Training WebUI")
    parser.add_argument("--file", type=str, default="checkpoints/metrics.jsonl",
                        help="Path to metrics JSONL file or directory containing it")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    path = Path(args.file)
    if path.is_dir():
        path = path / "metrics.jsonl"
    _metrics_path = str(path.resolve())

    if not os.path.exists(_metrics_path):
        print(f"⚠ Metrics file not found: {_metrics_path}")
        print(f"  UI will start but show no data until training logs metrics.")
    else:
        print(f"✓ Reading metrics from: {_metrics_path}")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"✓ WebUI → http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Shutdown")


if __name__ == "__main__":
    main()
