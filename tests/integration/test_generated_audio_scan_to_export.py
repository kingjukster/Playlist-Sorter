"""Small, redistributable scan-to-export coverage using generated WAV bytes."""

from __future__ import annotations

import struct
import wave
import json
from pathlib import Path

from playlist_sorter.export import create_preview, export_run
from playlist_sorter.pipeline import build_features, discover_run, scan_to_run


def _write_generated_wav(path: Path, *, samples: tuple[float, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(struct.pack("<h", round(sample * 32767)) for sample in samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(pcm)
    return path


def test_generated_audio_scan_feeds_canonical_export_without_touching_media(tmp_path):
    library = tmp_path / "library"
    first = _write_generated_wav(library / "zeta.wav", samples=(0.1,) * 800)
    second = _write_generated_wav(library / "alpha.wav", samples=(-0.1,) * 400)
    before = {path: path.read_bytes() for path in (first, second)}

    run_dir = tmp_path / "canonical-run"
    scan = scan_to_run(library, run_dir)
    features = build_features(run_dir)
    song_ids = [
        item["song_id"] for item in json.loads((run_dir / "songs.json").read_text())["items"]
    ]
    guidance = tmp_path / "guidance.yaml"
    guidance.write_text(
        json.dumps(
            {
                "guidance_id": "generated-guidance",
                "name": "generated concept",
                "positive_song_ids": song_ids,
            }
        ),
        encoding="utf-8",
    )
    discovery = discover_run(run_dir, guidance)
    preview = create_preview(run_dir, "csv")
    destination = tmp_path / "exports" / "generated.csv"

    assert export_run(run_dir, "csv", destination) == preview
    assert destination.read_bytes() == preview.content
    assert scan["songs"] == 2
    assert features["features"] == 2
    assert discovery["abstained"] is True
    assert discovery["guidance"] == "generated-guidance"
    assert preview.row_count == 0
    assert "playlist_name" in preview.text
    assert {path: path.read_bytes() for path in (first, second)} == before


def test_scan_retains_duplicate_files_but_emits_unique_canonical_songs(tmp_path):
    library = tmp_path / "library"
    first = _write_generated_wav(library / "first.wav", samples=(0.1,) * 800)
    duplicate = library / "duplicate.wav"
    duplicate.write_bytes(first.read_bytes())

    run_dir = tmp_path / "canonical-run"
    scan = scan_to_run(library, run_dir)
    catalog = json.loads((run_dir / "catalog.json").read_text(encoding="utf-8"))["items"]
    songs = json.loads((run_dir / "songs.json").read_text(encoding="utf-8"))["items"]

    assert scan["source_files"] == 2
    assert scan["songs"] == 1
    assert len(catalog) == 2
    assert len(songs) == 1
