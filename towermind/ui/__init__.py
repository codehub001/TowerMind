"""TowerMind UI: interactive web visualizer, dashboard generator, and local server."""

from towermind.ui.generator import build_dashboard_html, export_dashboard

__all__ = [
    "DashboardRequestHandler",
    "build_dashboard_html",
    "export_dashboard",
    "run_dashboard_server",
]


def __getattr__(name: str):
    if name in ("DashboardRequestHandler", "run_dashboard_server"):
        from towermind.ui import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
