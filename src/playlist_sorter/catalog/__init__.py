"""Immutable read-only audio catalog."""

from .scan import CatalogFailure, CatalogSong, scan_library, write_catalog_parquet

__all__ = ["CatalogFailure", "CatalogSong", "scan_library", "write_catalog_parquet"]
