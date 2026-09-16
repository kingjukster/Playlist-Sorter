from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from playlist_sorter.audio.segments import decode_wav_mono_24khz, select_segments
from playlist_sorter.catalog.scan import scan_library
from playlist_sorter.embeddings.cache import EmbeddingCache
from playlist_sorter.embeddings.registry import MODEL_REVISIONS, adapter_for
from playlist_sorter.features.descriptors import describe_samples
from playlist_sorter.features.lyrics import chunk_lyrics, pooled_spherical_mean, remove_duplicate_lines

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "audio" / "make_tiny_wav.py"
_SPEC = importlib.util.spec_from_file_location("tiny_wav_fixture", _FIXTURE)
assert _SPEC and _SPEC.loader
_FIXTURE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE_MODULE)
write_tiny_wav = _FIXTURE_MODULE.write_tiny_wav


def test_catalog_identity_renames_and_exact_duplicates_without_mutating_bytes(tmp_path):
    first = write_tiny_wav(tmp_path / "first.wav")
    before = first.read_bytes()
    second = tmp_path / "renamed.wav"
    shutil.copyfile(first, second)
    songs, failures = scan_library(tmp_path)
    assert not failures and len(songs) == 2
    assert {song.sha256 for song in songs} == {songs[0].song_id}
    assert first.read_bytes() == before


def test_corrupt_and_short_audio_are_isolated(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not wav")
    short = write_tiny_wav(tmp_path / "short.wav", seconds=.1)
    songs, failures = scan_library(tmp_path)
    assert len(songs) == 1 and failures[0].code == "corrupt_audio"
    samples, rate = decode_wav_mono_24khz(short)
    segments = select_segments(samples, rate)
    assert segments == [segments[0]] and segments[0].end_seconds == 10.0


def test_segmentation_dedupes_overlap_and_descriptors_are_deterministic():
    samples = [0.0] * 24_000 + [0.5] * (24_000 * 60) + [0.0] * 24_000
    segments = select_segments(samples, 24_000)
    assert len({segment.start_seconds for segment in segments}) == len(segments)
    assert all(segment.end_seconds - segment.start_seconds == pytest.approx(10) for segment in segments)
    assert describe_samples(samples, 24_000) == describe_samples(samples, 24_000)


def test_cache_hit_invalidation_and_model_revisions(tmp_path):
    cache = EmbeddingCache(tmp_path)
    provenance = {"source_sha256": "a", "model_revision": MODEL_REVISIONS["qwen"][1], "segment": 0}
    assert cache.get(provenance) is None
    cache.put(provenance, [1.0, 0.0])
    assert cache.get(provenance) == [1.0, 0.0]
    assert cache.get({**provenance, "model_revision": "new"}) is None
    with pytest.raises(RuntimeError, match="intentionally lazy"):
        adapter_for("qwen").load()


def test_lyrics_missing_duplicate_removal_chunking_and_weighted_pooling():
    assert remove_duplicate_lines("a\na\n\nb") == "a\nb"
    assert chunk_lyrics([], 384, 64) == []
    assert [len(chunk) for chunk in chunk_lyrics(list(range(800)))] == [384, 384, 160]
    assert pooled_spherical_mean([[1, 0], [0, 1]], [3, 1]) == pytest.approx([0.948683, 0.316228])
