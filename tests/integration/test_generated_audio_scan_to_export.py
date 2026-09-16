"""Small, redistributable scan-to-export coverage using generated WAV bytes."""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

from playlist_sorter.catalog.scan import scan_library
from playlist_sorter.export import create_preview, export_run


def _write_generated_wav(path: Path, *, samples: tuple[float, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(struct.pack("<h", round(sample * 32767)) for sample in samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(pcm)
    return path


def _write_canonical_run(run_dir: Path, songs) -> None:
    run_dir.mkdir()
    song_items = [
        {
            "song_id": song.song_id,
            "source_path": song.source_path,
            "title": f"Generated {index}",
            "artist": "Playlist Sorter fixture",
            "duration_seconds": song.duration_seconds,
            "codec": song.codec,
            "integrity_fingerprint": song.sha256,
        }
        for index, song in enumerate(songs, start=1)
    ]
    candidate_id = "generated-tones"
    payloads = {
        "songs.json": {"schema_version": "1.0", "items": song_items},
        "playlists.json": {
            "schema_version": "1.0",
            "items": [
                {
                    "candidate_id": candidate_id,
                    "name": "Generated tones",
                    "granularity": "fixture",
                    "member_song_ids": [song["song_id"] for song in song_items],
                    "stability": 1.0,
                    "cohesion": 1.0,
                    "separation": 1.0,
                    "novelty": 0.0,
                    "coverage": 1.0,
                }
            ],
        },
        "memberships.json": {"schema_version": "1.0", "items": []},
        "manifest.json": {
            "schema_version": "1.0",
            "run_id": "generated-audio-fixture",
            "created_at": "2026-01-01T00:00:00Z",
            "discovery_run": "generated-audio-fixture",
            "artifacts": [],
            "candidates": [candidate_id],
            "generator_version": "1.0",
        },
    }
    for filename, payload in payloads.items():
        (run_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_generated_audio_scan_feeds_canonical_export_without_touching_media(tmp_path):
    library = tmp_path / "library"
    first = _write_generated_wav(library / "zeta.wav", samples=(0.1,) * 800)
    second = _write_generated_wav(library / "alpha.wav", samples=(-0.1,) * 400)
    before = {path: path.read_bytes() for path in (first, second)}

    songs, failures = scan_library(library)

    assert not failures
    assert [Path(song.source_path).name for song in songs] == ["alpha.wav", "zeta.wav"]
    assert all(song.duration_seconds is not None for song in songs)
    assert {path: path.read_bytes() for path in (first, second)} == before

    run_dir = tmp_path / "canonical-run"
    _write_canonical_run(run_dir, songs)
    preview = create_preview(run_dir, "csv")
    destination = tmp_path / "exports" / "generated.csv"

    assert export_run(run_dir, "csv", destination) == preview
    assert destination.read_bytes() == preview.content
    assert preview.row_count == 2
    assert "Generated 1" in preview.text
    assert {path: path.read_bytes() for path in (first, second)} == before
