"""Lightweight local HTTP server for the TowerMind Interactive Dashboard."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from towermind.curriculum.maps import MAP_FACTORIES
from towermind.ui.generator import build_dashboard_html, export_dashboard

logger = logging.getLogger(__name__)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handles REST requests and serves the TowerMind interactive dashboard."""

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_text: str, status: int = 200) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            # Load real benchmark data if available
            comp_data = None
            gen_data = None
            comp_path = Path("benchmark_results/comparison_summary.json")
            gen_path = Path("benchmark_results/generalization/generalization_summary.json")
            if comp_path.exists():
                try:
                    with open(comp_path, encoding="utf-8") as f:
                        comp_data = json.load(f)
                except Exception:
                    pass
            if gen_path.exists():
                try:
                    with open(gen_path, encoding="utf-8") as f:
                        gen_data = json.load(f)
                except Exception:
                    pass

            html = build_dashboard_html(comp_data, gen_data)
            self._send_html(html)

        elif path == "/api/maps":
            self._send_json({"maps": sorted(MAP_FACTORIES.keys())})

        elif path == "/api/benchmarks":
            results_dir = Path("benchmark_results")
            files = list(results_dir.glob("*.json")) if results_dir.exists() else []
            self._send_json({"files": [f.name for f in files]})

        elif path == "/api/benchmark/latest":
            comp_path = Path("benchmark_results/comparison_summary.json")
            if comp_path.exists():
                with open(comp_path, encoding="utf-8") as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"error": "No benchmark results found"}, status=404)

        elif path == "/api/generalization":
            gen_path = Path("benchmark_results/generalization/generalization_summary.json")
            if gen_path.exists():
                with open(gen_path, encoding="utf-8") as f:
                    self._send_json(json.load(f))
            else:
                self._send_json({"error": "No generalization results found"}, status=404)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard request logging in production/CLI
        return


def run_dashboard_server(
    port: int = 8080,
    open_browser: bool = False,
    host: str = "127.0.0.1",
) -> None:
    """Start local web server serving the dashboard."""
    # Ensure exported dashboard HTML is written
    export_path = export_dashboard()
    print("=" * 70)
    print("TowerMind Web Visualizer & Dashboard")
    print(f"Server running at: http://{host}:{port}/")
    print(f"Exported static HTML: {export_path.resolve()}")
    print("=" * 70)

    server = HTTPServer((host, port), DashboardRequestHandler)

    if open_browser:
        url = f"http://{host}:{port}/"
        try:
            # On Linux (Fedora/Debian/Ubuntu), silence Chromium/Chrome zygote broken pipe stderr
            if sys.platform.startswith("linux"):
                import subprocess

                try:
                    subprocess.Popen(
                        ["xdg-open", url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception:
                    webbrowser.open(url)
            else:
                webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TowerMind server...")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run TowerMind Interactive Web Dashboard.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--open", action="store_true", help="Automatically open in browser")
    parser.add_argument(
        "--export-only", action="store_true", help="Only export static dashboard.html without running server"
    )
    args = parser.parse_args()

    if args.export_only:
        p = export_dashboard()
        print(f"Dashboard exported to: {p.resolve()}")
        sys.exit(0)

    run_dashboard_server(port=args.port, open_browser=args.open, host=args.host)


if __name__ == "__main__":
    main()
