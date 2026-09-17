"""Read-only audio decoding and deterministic segment selection."""

from .segments import (
    AudioDecodeError,
    Segment,
    decode_audio_mono_24khz,
    decode_wav_mono_24khz,
    select_segments,
)

__all__ = [
    "AudioDecodeError",
    "Segment",
    "decode_audio_mono_24khz",
    "decode_wav_mono_24khz",
    "select_segments",
]
