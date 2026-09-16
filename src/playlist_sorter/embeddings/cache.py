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
