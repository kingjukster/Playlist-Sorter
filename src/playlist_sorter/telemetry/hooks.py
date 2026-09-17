from __future__ import annotations
from collections.abc import Callable
from typing import Any

TelemetryHook = Callable[[str, dict[str, Any]], None]
