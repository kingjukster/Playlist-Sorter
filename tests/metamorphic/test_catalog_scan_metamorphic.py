"""Metamorphic boundaries for the offline, exact-byte catalog scanner."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from playlist_sorter.catalog import scan as scan_module
from playlist_sorter.catalog.scan import scan_library


def _write_generated_wav(path: Path, *, sample: float, frames: int = 800) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(struct.pack("<h", round(sample * 32767)) * frames)
    return path


def _scan_ids(root: Path) -> list[str]:
    songs, failures = scan_library(root)
    assert not failures
    return [song.song_id for song in songs]


def test_renaming_preserves_exact_byte_identity_but_updates_source_path(tmp_path):
    original = _write_generated_wav(tmp_path / "before.wav", sample=0.125)
    first, failures = scan_library(tmp_path)
    assert not failures
    original.rename(tmp_path / "after.wav")

    second, failures = scan_library(tmp_path)

    assert not failures
    assert first[0].song_id == second[0].song_id
    assert Path(second[0].source_path).name == "after.wav"


def test_exact_duplicate_bytes_remain_observable_and_reordered_input_is_deterministic(tmp_path):
    first = _write_generated_wav(tmp_path / "library" / "z.wav", sample=0.25)
    duplicate = tmp_path / "library" / "a-copy.wav"
    duplicate.write_bytes(first.read_bytes())
    _write_generated_wav(tmp_path / "library" / "m.wav", sample=-0.25)

    songs, failures = scan_library(tmp_path / "library")

    assert not failures
    assert [Path(song.source_path).name for song in songs] == ["a-copy.wav", "m.wav", "z.wav"]
    assert songs[0].song_id == songs[2].song_id
    reordered = tmp_path / "reordered"
    # Create the equivalent bytes in a different insertion order.  The scanner
    # owns ordering by normalized path, not directory enumeration order.
    _write_generated_wav(reordered / "m.wav", sample=-0.25)
    copied = _write_generated_wav(reordered / "z.wav", sample=0.25)
    (reordered / "a-copy.wav").write_bytes(copied.read_bytes())
    assert _scan_ids(tmp_path / "library") == _scan_ids(reordered)


def test_duplicate_encodes_expose_shared_fingerprint_without_false_merge(tmp_path, monkeypatch):
    first = _write_generated_wav(tmp_path / "first-encode.wav", sample=0.125)
    second = _write_generated_wav(tmp_path / "second-encode.wav", sample=0.125)
    second.write_bytes(second.read_bytes() + b"synthetic-container-padding")
    monkeypatch.setattr(scan_module, "_chromaprint", lambda _path: ("shared-fixture", "available"))

    songs, failures = scan_library(tmp_path)

    assert not failures
    assert first.read_bytes() != second.read_bytes()
    assert {song.chromaprint for song in songs} == {"shared-fixture"}
    assert len({song.song_id for song in songs}) == 2


def test_missing_lyrics_are_not_inferred_and_short_tracks_still_have_wav_evidence(tmp_path):
    _write_generated_wav(tmp_path / "short.wav", sample=0.01, frames=1)

    songs, failures = scan_library(tmp_path)

    assert not failures
    assert songs[0].duration_seconds == 1 / 8_000
    assert songs[0].metadata == {"sample_rate": "8000", "channels": "1"}
    assert "lyrics" not in songs[0].metadata


def test_corrupt_audio_is_isolated_and_small_byte_perturbations_change_identity(tmp_path):
    _write_generated_wav(tmp_path / "reference.wav", sample=0.125)
    _write_generated_wav(tmp_path / "perturbed.wav", sample=0.1251)
    (tmp_path / "corrupt.wav").write_bytes(b"not a wav fixture")

    songs, failures = scan_library(tmp_path)

    assert {Path(song.source_path).name for song in songs} == {"perturbed.wav", "reference.wav"}
    assert len({song.song_id for song in songs}) == 2
    assert [(Path(failure.path).name, failure.code) for failure in failures] == [
        ("corrupt.wav", "corrupt_audio")
    ]
