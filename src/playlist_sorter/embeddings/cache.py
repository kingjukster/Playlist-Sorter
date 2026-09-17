"""Content-addressed vector cache with complete inference provenance."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any


_REQUIRED_PROVENANCE_KEYS = {
    "model_repository",
    "model_revision",
    "preprocessing",
    "pooling",
    "feature_view",
}
_SOURCE_PROVENANCE_KEYS = {"source_fingerprint", "source_sha256"}
_OPTIONAL_PROVENANCE_KEYS = {"segment_id"}


def _json_value(value: Any, *, field: str) -> Any:
    """Return a JSON-compatible copy, rejecting lossy or ambiguous values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{field} cannot contain a non-finite float")
        return value
    if isinstance(value, list):
        return [_json_value(item, field=field) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field} mapping keys must be strings")
        return {key: _json_value(item, field=field) for key, item in value.items()}
    raise TypeError(f"{field} must contain only JSON values")


def validate_embedding_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the complete, canonical embedding cache identity.

    Canonical records identify source bytes, the pinned model, preprocessing,
    pooling, and the resulting feature view.  ``segment_id`` optionally
    distinguishes a segment vector from a pooled vector.  Generic cache users
    may continue to use :class:`EmbeddingCache` directly with their own
    provenance dictionaries; this validator is the fail-closed r5 boundary.
    """
    if not isinstance(provenance, Mapping):
        raise TypeError("embedding provenance must be a mapping")
    if not all(isinstance(key, str) for key in provenance):
        raise TypeError("embedding provenance keys must be strings")

    keys = set(provenance)
    allowed = _REQUIRED_PROVENANCE_KEYS | _SOURCE_PROVENANCE_KEYS | _OPTIONAL_PROVENANCE_KEYS
    missing = _REQUIRED_PROVENANCE_KEYS - keys
    unknown = keys - allowed
    sources = keys & _SOURCE_PROVENANCE_KEYS
    if missing or unknown or len(sources) != 1:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        if len(sources) != 1:
            details.append("exactly one of source_fingerprint or source_sha256 is required")
        raise ValueError("invalid embedding provenance: " + "; ".join(details))

    result = _json_value(provenance, field="embedding provenance")
    assert isinstance(result, dict)
    for field in (*sources, "model_repository", "model_revision", "feature_view"):
        if not isinstance(result[field], str) or not result[field]:
            raise TypeError(f"{field} must be a non-empty string")
    if "segment_id" in result and (
        not isinstance(result["segment_id"], str) or not result["segment_id"]
    ):
        raise TypeError("segment_id must be a non-empty string when supplied")
    for field in ("preprocessing", "pooling"):
        if not isinstance(result[field], dict):
            raise TypeError(f"{field} must be a mapping")
    return result


def embedding_provenance(
    *,
    source_fingerprint: str | None = None,
    source_sha256: str | None = None,
    model_repository: str,
    model_revision: str,
    preprocessing: Mapping[str, Any],
    pooling: Mapping[str, Any],
    feature_view: str,
    segment_id: str | None = None,
) -> dict[str, Any]:
    """Build a validated canonical embedding cache provenance record."""
    provenance: dict[str, Any] = {
        "model_repository": model_repository,
        "model_revision": model_revision,
        "preprocessing": preprocessing,
        "pooling": pooling,
        "feature_view": feature_view,
    }
    if source_fingerprint is not None:
        provenance["source_fingerprint"] = source_fingerprint
    if source_sha256 is not None:
        provenance["source_sha256"] = source_sha256
    if segment_id is not None:
        provenance["segment_id"] = segment_id
    return validate_embedding_provenance(provenance)


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
        path.write_text(
            json.dumps({"provenance": provenance, "vector": vector}, sort_keys=True),
            encoding="utf-8",
        )
        return path


class CacheBackendUnavailable(RuntimeError):
    """Raised only when an explicitly requested artifact backend is absent."""


class SafetensorsParquetCache:
    """Canonical vector payload and provenance index cache.

    Imports occur only at read/write time, keeping discovery and unit tests free
    of model or heavyweight serialization imports.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def paths(self, provenance: dict[str, Any]) -> tuple[Path, Path]:
        key = cache_key(provenance)
        return self.root / f"{key}.safetensors", self.root / f"{key}.parquet"

    def put(self, provenance: dict[str, Any], vector: list[float]) -> tuple[Path, Path]:
        try:
            import numpy as np
            import pyarrow as pa
            import pyarrow.parquet as pq
            from safetensors.numpy import save_file
        except ImportError as exc:
            raise CacheBackendUnavailable(
                "Safetensors/Parquet cache requires optional numpy, pyarrow, and safetensors backends"
            ) from exc
        vector_path, index_path = self.paths(provenance)
        self.root.mkdir(parents=True, exist_ok=True)
        save_file(
            {"vector": np.asarray(vector, dtype=np.float32)},
            str(vector_path),
            metadata={"key": vector_path.stem},
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "cache_key": vector_path.stem,
                        "provenance": json.dumps(provenance, sort_keys=True),
                    }
                ]
            ),
            index_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=False,
        )
        return vector_path, index_path

    def get(self, provenance: dict[str, Any]) -> list[float] | None:
        try:
            import pyarrow.parquet as pq
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise CacheBackendUnavailable(
                "Safetensors/Parquet cache requires optional pyarrow and safetensors backends"
            ) from exc
        vector_path, index_path = self.paths(provenance)
        if not vector_path.is_file() or not index_path.is_file():
            return None
        index = pq.read_table(index_path).to_pylist()
        if index != [
            {"cache_key": vector_path.stem, "provenance": json.dumps(provenance, sort_keys=True)}
        ]:
            return None
        return load_file(str(vector_path))["vector"].tolist()
