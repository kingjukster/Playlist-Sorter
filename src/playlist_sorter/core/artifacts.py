"""Canonical content-addressed artifact naming helpers."""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def artifact_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def artifact_key(kind: str, identity: Any, version: str = "1.0") -> str:
    if not kind or "/" in kind or "\\" in kind:
        raise ValueError("invalid artifact kind")
    return f"{kind}/v{version}/{artifact_fingerprint(identity)}.json"


def canonical_artifact_path(root: str, kind: str, identity: Any, version: str = "1.0") -> str:
    return str(PurePosixPath(root.replace("\\", "/")) / artifact_key(kind, identity, version))
