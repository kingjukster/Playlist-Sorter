"""Pinned, lazy embedding adapter registry and cache."""

from .registry import MODEL_REVISIONS, adapter_for
from .cache import CacheBackendUnavailable, EmbeddingCache, SafetensorsParquetCache, cache_key

__all__ = [
    "MODEL_REVISIONS",
    "CacheBackendUnavailable",
    "EmbeddingCache",
    "SafetensorsParquetCache",
    "adapter_for",
    "cache_key",
]
