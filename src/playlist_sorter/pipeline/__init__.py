"""Canonical, local-first pipeline services used by both CLI and UI."""

from .service import build_features, discover_run, evaluate_run, scan_to_run

__all__ = ["build_features", "discover_run", "evaluate_run", "scan_to_run"]
