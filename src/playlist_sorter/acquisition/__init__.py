"""Rights-aware acquisition of explicitly authorized source media."""

from .youtube import (
    SUPPORTED_AUDIO_FORMATS,
    AcquisitionError,
    AcquisitionEntry,
    AcquisitionManifest,
    acquire_youtube_audio,
    build_ytdlp_command,
    load_acquisition_manifest,
)

__all__ = [
    "SUPPORTED_AUDIO_FORMATS",
    "AcquisitionEntry",
    "AcquisitionError",
    "AcquisitionManifest",
    "acquire_youtube_audio",
    "build_ytdlp_command",
    "load_acquisition_manifest",
]
