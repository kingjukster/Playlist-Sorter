"""Deterministic descriptors and lyric chunk planning."""

from .lyrics import chunk_lyrics, pooled_spherical_mean, remove_duplicate_lines
from .descriptors import describe_samples

__all__ = ["chunk_lyrics", "pooled_spherical_mean", "remove_duplicate_lines", "describe_samples"]
