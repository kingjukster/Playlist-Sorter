"""Immutable read-only audio catalog."""

from .scan import (
    CatalogFailure,
    CatalogSong,
    group_musical_variants,
    scan_library,
    write_catalog_parquet,
)

__all__ = [
    "CatalogFailure",
    "CatalogSong",
    "group_musical_variants",
    "scan_library",
    "write_catalog_parquet",
]
