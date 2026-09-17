from __future__ import annotations

import json

import pytest

from playlist_sorter.discovery.backends import SeededLeidenBackend
from playlist_sorter.graph import FusedGraph, ViewGraph
from playlist_sorter.pipeline import service


def _write_run(run_dir, views):
    song_ids = [f"song-{index:02d}" for index in range(20)]
    songs = [
        {
            "song_id": song_id,
            "source_path": str(run_dir / f"{song_id}.wav"),
            "integrity_fingerprint": f"source-{index:02d}",
        }
        for index, song_id in enumerate(song_ids)
    ]
    features = [
        {"song_id": song_id, "feature_view": view, "vector": [1.0, index / 100.0]}
        for view in views
        for index, song_id in enumerate(song_ids)
    ]
    (run_dir / "songs.json").write_text(
        json.dumps({"schema_version": "1.0", "items": songs}), encoding="utf-8"
    )
    (run_dir / "features.json").write_text(
        json.dumps({"schema_version": "1.0", "items": features}), encoding="utf-8"
    )
    return song_ids


def _complete_edges(group):
    return {(left, right): 0.9 for left in group for right in group if left != right}


def _fake_graphs(song_ids, vectors_by_lens, *, k):
    groups = {
        "semantic_audio": range(0, 8),
        "acoustic": range(4, 12),
        "lyrics": range(8, 16),
        "descriptors": range(0, 8),
    }
    base = {
        lens: ViewGraph(lens, tuple(song_ids), _complete_edges(groups[lens]), k, "cpu")
        for lens in vectors_by_lens
    }
    first = next(iter(base.values()))
    fused = FusedGraph(tuple(song_ids), dict(first.edges), {}, {})
    return {**base, "fused": fused}


class _ComponentsOnly:
    def __init__(self, *, min_edge_weight):
        self.backend = SeededLeidenBackend(min_edge_weight=min_edge_weight)

    def cluster(self, song_ids, edges, *, resolution, seed):
        return self.backend._components(song_ids, edges, resolution=resolution, seed=seed)


def test_multilens_discovery_is_independent_overlap_safe_and_guidance_auditable(
    tmp_path, monkeypatch
):
    song_ids = _write_run(tmp_path, ("semantic_audio", "acoustic"))
    monkeypatch.setattr(service, "_discovery_graphs", _fake_graphs)
    monkeypatch.setattr(service, "SeededLeidenBackend", _ComponentsOnly)
    monkeypatch.setattr(service, "leiden_available", lambda: True)
    guidance = tmp_path / "guidance.yaml"
    guidance.write_text(
        json.dumps(
            {
                "guidance_id": "g",
                "name": "seeded",
                "positive_song_ids": song_ids[:2],
            }
        ),
        encoding="utf-8",
    )

    summary = service.discover_run(tmp_path, guidance)
    playlists = service._read_collection(tmp_path, "playlists.json")
    memberships = service._read_collection(tmp_path, "memberships.json")
    evidence = json.loads((tmp_path / "discovery_evidence.json").read_text(encoding="utf-8"))

    assert summary["lenses"] == ["acoustic", "fused", "semantic_audio"]
    assert summary["backends"] == ["deterministic-components"]
    assert summary["fallback_reasons"] == ["leiden-unavailable"]
    assert summary["guidance"] == "g"
    assert (tmp_path / "unguided_graph.json").is_file()
    selected = [item for item in evidence["candidates"] if item["selected"]]
    assert any(
        item["lens"] == "acoustic" and item["passed_gates"] for item in evidence["candidates"]
    )
    assert any(item["lens"] in {"semantic_audio", "fused"} for item in selected)
    member_sets = [set(item["member_song_ids"]) for item in playlists]
    assert any(
        left & right for index, left in enumerate(member_sets) for right in member_sets[index + 1 :]
    )
    assert sum(item["song_id"] == "song-04" for item in memberships) >= 2
    assert all("membership_score" in item and "score" not in item for item in memberships)


def test_descriptor_only_discovery_remains_supported(tmp_path, monkeypatch):
    _write_run(tmp_path, ("descriptors",))
    monkeypatch.setattr(service, "_discovery_graphs", _fake_graphs)
    monkeypatch.setattr(service, "SeededLeidenBackend", _ComponentsOnly)

    summary = service.discover_run(tmp_path)
    evidence = json.loads((tmp_path / "discovery_evidence.json").read_text(encoding="utf-8"))

    assert summary["lenses"] == ["descriptors", "fused"]
    assert any(item["passed_gates"] for item in evidence["candidates"])


def test_discovery_deduplicates_exact_source_song_ids(tmp_path, monkeypatch):
    song_ids = _write_run(tmp_path, ("semantic_audio", "acoustic"))
    songs_path = tmp_path / "songs.json"
    songs = json.loads(songs_path.read_text(encoding="utf-8"))
    duplicate = dict(songs["items"][0])
    duplicate["source_path"] = str(tmp_path / "duplicate-encode.wav")
    songs["items"].append(duplicate)
    songs_path.write_text(json.dumps(songs), encoding="utf-8")
    monkeypatch.setattr(service, "_discovery_graphs", _fake_graphs)
    monkeypatch.setattr(service, "SeededLeidenBackend", _ComponentsOnly)
    monkeypatch.setattr(service, "leiden_available", lambda: True)

    summary = service.discover_run(tmp_path)

    assert summary["usable_songs"] == len(song_ids)


def test_research_discovery_fails_closed_without_leiden(tmp_path, monkeypatch):
    _write_run(tmp_path, ("semantic_audio", "acoustic"))
    monkeypatch.setattr(service, "leiden_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires igraph and leidenalg"):
        service.discover_run(tmp_path)
