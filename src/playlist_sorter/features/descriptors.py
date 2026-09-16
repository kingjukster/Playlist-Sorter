"""Portable deterministic descriptors; replaceable by an Essentia/librosa adapter."""

from __future__ import annotations
import math
from statistics import fmean


def describe_samples(samples: list[float], sample_rate: int) -> dict[str, float]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not samples:
        return {"duration_seconds": 0.0, "rms": 0.0, "peak": 0.0, "zero_crossing_rate": 0.0}
    rms = math.sqrt(fmean(value * value for value in samples))
    crossings = sum(a * b < 0 for a, b in zip(samples, samples[1:]))
    return {"duration_seconds": len(samples) / sample_rate, "rms": rms, "peak": max(map(abs, samples)),
            "zero_crossing_rate": crossings / max(1, len(samples) - 1)}
