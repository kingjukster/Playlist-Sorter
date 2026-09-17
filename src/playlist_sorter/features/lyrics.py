"""Pure lyric preprocessing; tokenization is supplied by the pinned adapter."""

from __future__ import annotations
import math
from typing import Sequence


def remove_duplicate_lines(text: str) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for line in text.splitlines():
        key = line.strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(line)
    return "\n".join(kept)


def chunk_lyrics(tokens: Sequence[int], size: int = 384, overlap: int = 64) -> list[list[int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    stride = size - overlap
    return [
        list(tokens[index : index + size])
        for index in range(0, len(tokens), stride)
        if tokens[index : index + size]
    ]


def pooled_spherical_mean(
    vectors: Sequence[Sequence[float]], weights: Sequence[int]
) -> list[float]:
    if len(vectors) != len(weights) or not vectors or any(weight <= 0 for weight in weights):
        raise ValueError("vectors and positive token weights are required")
    dims = len(vectors[0])
    total = [0.0] * dims
    for vector, weight in zip(vectors, weights, strict=True):
        if len(vector) != dims:
            raise ValueError("embedding dimensions differ")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            for index, value in enumerate(vector):
                total[index] += weight * value / norm
    norm = math.sqrt(sum(value * value for value in total))
    return [value / norm for value in total] if norm else total
