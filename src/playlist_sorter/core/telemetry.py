"""Private-by-default run manifest and resource telemetry helpers."""

from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_manifest(
    run_id: str, *, command: str = "", config: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": utc_now(),
        "command": command,
        "config": config or {},
        "status": "running",
        "telemetry": [],
    }


def record_telemetry(manifest: dict[str, Any], event: str, **measurements: Any) -> dict[str, Any]:
    item = {"event": event, "timestamp": utc_now(), **measurements}
    manifest.setdefault("telemetry", []).append(item)
    return item


def write_run_manifest(manifest: dict[str, Any], output: str | Path) -> Path:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return p
