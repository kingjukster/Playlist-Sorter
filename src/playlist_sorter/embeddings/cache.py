"""Content-addressed vector cache with complete inference provenance."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(provenance: dict[str, Any]) -> str:
    raw = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class EmbeddingCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
    def get(self, provenance: dict[str, Any]) -> list[float] | None:
        path = self.root / f"{cache_key(provenance)}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["vector"] if value.get("provenance") == provenance else None
    def put(self, provenance: dict[str, Any], vector: list[float]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{cache_key(provenance)}.json"
        path.write_text(json.dumps({"provenance": provenance, "vector": vector}, sort_keys=True), encoding="utf-8")
        return path
