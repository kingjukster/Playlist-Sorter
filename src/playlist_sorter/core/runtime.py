"""Fail-closed, side-effect-free runtime readiness checks."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable


@dataclass(frozen=True)
class ResourceSnapshot:
    ram_gib: float | None = None
    vram_gib: float | None = None
    free_disk_gib: float | None = None
    contention: bool = False


def _vram_from_nvidia_smi(runner: Callable[..., Any] = subprocess.run) -> float | None:
    """Read free VRAM without importing torch or initializing CUDA."""
    try:
        result = runner(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        values = [
            float(line.strip()) / 1024 for line in str(result.stdout).splitlines() if line.strip()
        ]
        return max(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def resource_snapshot(
    path: str | os.PathLike[str] = ".",
    *,
    vram_reader: Callable[[], float | None] | None = None,
    contention_reader: Callable[[], bool] | None = None,
) -> ResourceSnapshot:
    """Collect cheap measurements; unknown values deliberately fail readiness."""
    ram = None
    try:
        import psutil

        ram = psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    try:
        disk = shutil.disk_usage(Path(path)).free / 2**30
    except OSError:
        disk = None
    return ResourceSnapshot(
        ram,
        (vram_reader or _vram_from_nvidia_smi)(),
        disk,
        (contention_reader or (lambda: False))(),
    )


def readiness(snapshot: ResourceSnapshot, *, min_disk_gib: float = 2.0) -> dict:
    checks = {
        "ram": snapshot.ram_gib is not None and snapshot.ram_gib >= 8.0,
        "vram": snapshot.vram_gib is not None and snapshot.vram_gib >= 12.0,
        "disk": snapshot.free_disk_gib is not None and snapshot.free_disk_gib >= min_disk_gib,
        "contention": not snapshot.contention,
    }
    return {"ready": all(checks.values()), "checks": checks, "snapshot": asdict(snapshot)}


def check_model_registry(registry: str | os.PathLike[str]) -> dict:
    p = Path(registry)
    try:
        import tomllib

        models = tomllib.loads(p.read_text(encoding="utf-8")).get("models", [])
        valid = len(models) == 3 and all(
            {"name", "repository", "revision", "license"} <= set(item) for item in models
        )
        reason = None if valid else "expected-three-pinned-models"
    except (OSError, ValueError):
        valid, reason = False, "missing-or-invalid-toml"
    return {"path": str(p), "exists": p.exists(), "valid": valid, "reason": reason}


def doctor(
    path: str | os.PathLike[str] = ".",
    *,
    snapshot: ResourceSnapshot | None = None,
    registry: str | os.PathLike[str] = "configs/license_registry.toml",
) -> dict:
    target = Path(path)
    gate = readiness(snapshot or resource_snapshot(target))
    try:
        target.mkdir(parents=True, exist_ok=True)
        writable = os.access(target, os.W_OK)
    except OSError:
        writable = False
    ffmpeg = shutil.which("ffmpeg") is not None
    registry_check = check_model_registry(registry)
    gate["checks"].update(
        {
            "python": sys.version_info >= (3, 12),
            "ffmpeg": ffmpeg,
            "writable_output": writable,
            "model_registry": registry_check["valid"],
        }
    )
    gate.update(
        {
            "ready": gate["ready"] and all(gate["checks"].values()),
            "wsl": bool(
                os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower()
            ),
            "python_version": platform.python_version(),
            "ffmpeg": ffmpeg,
            "writable_output": writable,
            "model_registry": registry_check,
        }
    )
    return gate
