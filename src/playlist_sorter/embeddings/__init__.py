"""Pinned, lazy embedding adapter registry and cache."""

from .registry import (
    MODEL_REVISIONS,
    InferenceResult,
    InferenceTelemetry,
    MuQAdapter,
    ProductionAdapter,
    QwenLyricsAdapter,
    adapter_for,
)
from .cache import (
    CacheBackendUnavailable,
    EmbeddingCache,
    SafetensorsParquetCache,
    cache_key,
    embedding_provenance,
    validate_embedding_provenance,
)

__all__ = [
    "MODEL_REVISIONS",
    "CacheBackendUnavailable",
    "EmbeddingCache",
    "InferenceResult",
    "InferenceTelemetry",
    "MuQAdapter",
    "ProductionAdapter",
    "QwenLyricsAdapter",
    "SafetensorsParquetCache",
    "adapter_for",
    "cache_key",
    "embedding_provenance",
    "validate_embedding_provenance",
]
