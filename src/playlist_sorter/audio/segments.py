"""Small, dependency-free WAV path plus deterministic window selection."""

from __future__ import annotations

from dataclasses import dataclass
import audioop
from pathlib import Path
import shutil
import struct
import subprocess
import wave

TARGET_RATE = 24_000
WINDOW_SECONDS = 10.0


class AudioDecodeError(ValueError):
    """A structured, non-destructive decode failure."""

    def __init__(self, code: str, path: Path, detail: str):
        self.code, self.path, self.detail = code, path, detail
        super().__init__(f"{code}: {path}: {detail}")


@dataclass(frozen=True)
class Segment:
    start_seconds: float
    end_seconds: float
    reason: str


def decode_wav_mono_24khz(path: str | Path) -> tuple[list[float], int]:
    """Decode PCM WAV to mono float32-like Python floats at 24 kHz.

    Other formats deliberately need an installed FFmpeg/librosa adapter; this
    keeps scanning harmless and imports optional libraries only on use.
    """
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels, width, rate, frames = (
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getframerate(),
                handle.getnframes(),
            )
            if handle.getcomptype() != "NONE" or width not in (1, 2, 3, 4):
                raise AudioDecodeError(
                    "unsupported_wav", source, "only PCM 8/16/24/32-bit WAV is supported"
                )
            raw = handle.readframes(frames)
    except AudioDecodeError:
        raise
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioDecodeError("corrupt_audio", source, str(exc)) from exc
    try:
        if channels > 1:
            raw = audioop.tomono(raw, width, 0.5, 0.5)
        if rate != TARGET_RATE:
            raw, _ = audioop.ratecv(raw, width, 1, rate, TARGET_RATE, None)
        scale = float(1 << (width * 8 - 1))
        values = [
            audioop.getsample(raw, width, index) / scale for index in range(len(raw) // width)
        ]
    except audioop.error as exc:
        raise AudioDecodeError("decode_failed", source, str(exc)) from exc
    return values, TARGET_RATE


def decode_audio_mono_24khz(
    path: str | Path, *, ffmpeg: str | None = None
) -> tuple[list[float], int]:
    """Decode a supported format with FFmpeg, without importing media packages."""
    source = Path(path)
    if source.suffix.lower() == ".wav":
        return decode_wav_mono_24khz(source)
    executable = ffmpeg or shutil.which("ffmpeg")
    if executable is None:
        raise AudioDecodeError(
            "ffmpeg_unavailable",
            source,
            "install FFmpeg and ensure its executable is on PATH to decode MP3/OGG/FLAC/M4A",
        )
    command = [
        executable,
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except OSError as exc:
        raise AudioDecodeError("ffmpeg_failed", source, str(exc)) from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip() or "FFmpeg failed"
        raise AudioDecodeError("decode_failed", source, detail)
    if len(completed.stdout) % 4:
        raise AudioDecodeError("decode_failed", source, "FFmpeg produced incomplete float32 PCM")
    return list(struct.unpack(f"<{len(completed.stdout) // 4}f", completed.stdout)), TARGET_RATE


def select_segments(
    samples: list[float], sample_rate: int, window_seconds: float = WINDOW_SECONDS
) -> list[Segment]:
    """Choose percent-centered windows and one best non-overlapping RMS window."""
    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("sample_rate and window_seconds must be positive")
    duration = len(samples) / sample_rate
    if not samples:
        return [Segment(0.0, window_seconds, "padded-short")]
    threshold = max(max(abs(value) for value in samples) * 0.02, 1e-4)
    active = [index for index, value in enumerate(samples) if abs(value) >= threshold]
    trim_start = active[0] / sample_rate if active else 0.0
    trim_end = active[-1] / sample_rate if active else duration
    usable = max(0.0, trim_end - trim_start)
    half = window_seconds / 2

    def bounds(center: float) -> tuple[float, float]:
        if duration <= window_seconds:
            return 0.0, window_seconds
        start = min(max(center - half, trim_start), max(trim_start, trim_end - window_seconds))
        return round(start, 6), round(start + window_seconds, 6)

    selected: list[Segment] = []
    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        center = trim_start + usable * fraction
        start, end = bounds(center)
        if not any(abs(start - prior.start_seconds) < 1e-6 for prior in selected):
            selected.append(Segment(start, end, f"percent-{int(fraction * 100)}"))

    window = max(1, round(window_seconds * sample_rate))
    candidates = range(0, max(1, len(samples) - window + 1), max(1, sample_rate))
    best = max(
        candidates,
        key=lambda start: sum(value * value for value in samples[start : start + window]),
    )
    start, end = bounds((best + window / 2) / sample_rate)
    if not any(max(start, item.start_seconds) < min(end, item.end_seconds) for item in selected):
        selected.append(Segment(start, end, "highest-rms"))
    return selected
