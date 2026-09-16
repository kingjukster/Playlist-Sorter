"""Pinned, lazy embedding adapter registry and cache."""

from .registry import MODEL_REVISIONS, adapter_for
from .cache import EmbeddingCache, cache_key

__all__ = ["MODEL_REVISIONS", "EmbeddingCache", "adapter_for", "cache_key"]
