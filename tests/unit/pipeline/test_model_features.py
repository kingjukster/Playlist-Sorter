from __future__ import annotations

import json

import pytest

from playlist_sorter.audio import Segment
from playlist_sorter.embeddings import (
    EmbeddingCache,
    MODEL_REVISIONS,
    MuQAdapter,
    QwenLyricsAdapter,
)
from playlist_sorter.pipeline import service


class FakeModel:
    def __init__(self) -> None:
        self.audio_calls = 0
        self.text_calls = 0

    def embed_audio(self, batch):
        self.audio_calls += 1
        return [[1.0, 0.0] for _ in batch]

    def tokenize(self, text):
        return list(range(len(text.split())))

    def embed_text(self, batch):
        self.text_calls += 1
        return [[0.0, 1.0] for _ in batch]


def _songs(run_dir, lyrics="one\none\ntwo"):
    payload = {
        "schema_version": "1.0",
        "items": [
            {
                "song_id": "song-1",
                "source_path": str(run_dir / "song.wav"),
                "lyrics": lyrics,
                "lyrics_provenance": "fixture",
                "integrity_fingerprint": "source-v1",
            }
        ],
    }
    (run_dir / "songs.json").write_text(json.dumps(payload), encoding="utf-8")


def _adapters(fake):
    result = {
        "muq": MuQAdapter("muq", *MODEL_REVISIONS["muq"]),
        "mulan": MuQAdapter("mulan", *MODEL_REVISIONS["mulan"]),
        "qwen": QwenLyricsAdapter("qwen", *MODEL_REVISIONS["qwen"]),
    }
    for adapter in result.values():
        adapter._model = fake
    return result


def test_research_views_abstain_without_lyrics_and_warm_cache_skips_inference(
    tmp_path, monkeypatch
):
    _songs(tmp_path)
    monkeypatch.setattr(service, "decode_audio_mono_24khz", lambda path: ([0.1] * 48_000, 24_000))
    monkeypatch.setattr(
        service,
        "select_segments",
        lambda samples, rate: [Segment(0.0, 1.0, "fixture")],
    )
    fake = FakeModel()
    adapters = _adapters(fake)
    cache = EmbeddingCache(tmp_path / "cache")

    first = service.build_features(tmp_path, "research", adapters=adapters, cache=cache)
    calls = (fake.audio_calls, fake.text_calls)
    second = service.build_features(tmp_path, "research", adapters=adapters, cache=cache)
    rows = service._read_collection(tmp_path, "features.json")

    assert {row["feature_view"] for row in rows} == {"acoustic", "semantic_audio", "lyrics"}
    assert calls == (fake.audio_calls, fake.text_calls)
    assert first["model_inference"] == second["model_inference"] == "model-backed research views"

    _songs(tmp_path, lyrics=None)
    service.build_features(tmp_path, "research", adapters=adapters, cache=cache)
    rows = service._read_collection(tmp_path, "features.json")
    assert "lyrics" not in {row["feature_view"] for row in rows}


def test_descriptor_profile_never_touches_adapters_and_unknown_profile_fails(tmp_path, monkeypatch):
    _songs(tmp_path)
    monkeypatch.setattr(service, "decode_audio_mono_24khz", lambda path: ([0.1] * 48_000, 24_000))
    monkeypatch.setattr(
        service,
        "select_segments",
        lambda samples, rate: [Segment(0.0, 1.0, "fixture")],
    )

    result = service.build_features(tmp_path, "descriptors", adapters={"muq": object()})
    rows = service._read_collection(tmp_path, "features.json")
    assert result["model_inference"] == "not_run; descriptor lens only"
    assert [row["feature_view"] for row in rows] == ["descriptors"]
    with pytest.raises(ValueError, match="unsupported feature profile"):
        service.build_features(tmp_path, "Research")
