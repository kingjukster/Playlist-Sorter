"""Generate the redistributable, deterministic tiny PCM WAV fixture on demand."""

from __future__ import annotations
import math
from pathlib import Path
import struct
import wave


def write_tiny_wav(path: str | Path, seconds: float = 2.0, rate: int = 24_000) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = [int(10_000 * math.sin(2 * math.pi * 440 * index / rate)) for index in range(round(seconds * rate))]
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return target
