"""Immutable read-only audio catalog."""

from .scan import (
    CatalogFailure,
    CatalogSong,
    group_musical_variants,
    map_catalog_song,
    resolve_acquisition_provenance,
    scan_library,
    write_catalog_parquet,
)

__all__ = [
    "CatalogFailure",
    "CatalogSong",
    "group_musical_variants",
    "map_catalog_song",
    "resolve_acquisition_provenance",
    "scan_library",
    "write_catalog_parquet",
]
