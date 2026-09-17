from __future__ import annotations

import json

import pytest

from playlist_sorter.export import ExportError, create_preview, export_run, load_export_data
from playlist_sorter.export import service


def _song(song_id: str, source_path: str) -> dict[str, object]:
    return {
        "song_id": song_id,
        "source_path": source_path,
        "integrity_fingerprint": f"hash-{song_id}",
        "title": f"Title {song_id}",
        "artist": "Artist",
        "duration_seconds": 12.9,
    }


def _playlist(candidate_id: str, songs: list[str]) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "name": f"Playlist {candidate_id}",
        "granularity": "medium",
        "member_song_ids": songs,
        "stability": 0.8,
        "cohesion": 0.7,
        "separation": 0.6,
        "novelty": 0.5,
        "coverage": 0.9,
    }


def _run(tmp_path):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "songs.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "items": [_song("b", r"C:\Music\B.mp3"), _song("a", r"D:\Music\A.mp3")],
            }
        ),
        encoding="utf-8",
    )
    (run / "playlists.json").write_text(
        json.dumps({"schema_version": "1.0", "items": [_playlist("z", ["b", "a"])]}),
        encoding="utf-8",
    )
    (run / "memberships.json").write_text(
        json.dumps({"schema_version": "1.0", "items": []}), encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "r",
                "created_at": "2026-01-01T00:00:00Z",
                "discovery_run": "d",
                "artifacts": [],
                "candidates": ["z"],
                "generator_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    return run


@pytest.mark.parametrize("format", ["json", "csv", "m3u8", "html"])
def test_preview_round_trip_is_deterministic_and_preserves_sources(tmp_path, format):
    run = _run(tmp_path)
    before = (run / "songs.json").read_bytes()
    first = create_preview(run, format)
    second = create_preview(run, format)
    assert first == second
    target = tmp_path / f"export.{format}"
    assert export_run(run, format, target) == first
    assert target.read_bytes() == first.content
    assert (run / "songs.json").read_bytes() == before
    if format == "json":
        assert [row["song_id"] for row in json.loads(first.text)["playlists"]] == ["a", "b"]
    if format == "m3u8":
        assert "D:/Music/A.mp3" in first.text
    if format == "html":
        assert "Your discovered playlists" in first.text
        assert "Playlist z" in first.text
        assert "2 songs" in first.text
        assert 'Artist</strong><span class="separator"> — </span>Title a' in first.text
        assert "D:/Music/A.mp3" not in first.text


def test_html_report_escapes_names_and_explains_quality_metrics(tmp_path):
    run = _run(tmp_path)
    songs = json.loads((run / "songs.json").read_text(encoding="utf-8"))
    songs["items"][0]["artist"] = '<script>alert("artist")</script>'
    songs["items"][0]["title"] = "Rock & Roll"
    (run / "songs.json").write_text(json.dumps(songs), encoding="utf-8")
    playlists = json.loads((run / "playlists.json").read_text(encoding="utf-8"))
    playlists["items"][0]["name"] = "Friends < Favorites"
    playlists["items"][0]["naming_evidence"] = ["energy & tone"]
    (run / "playlists.json").write_text(json.dumps(playlists), encoding="utf-8")

    report = create_preview(run, "html").text

    assert "Friends &lt; Favorites" in report
    assert "&lt;script&gt;alert(&quot;artist&quot;)&lt;/script&gt;" in report
    assert "Rock &amp; Roll" in report
    assert "Why it grouped: energy &amp; tone" in report
    assert "How consistently the same songs grouped together" in report
    assert '<script>alert("artist")</script>' not in report


def test_missing_or_unversioned_canonical_artifacts_fail_closed(tmp_path):
    run = _run(tmp_path)
    (run / "manifest.json").unlink()
    with pytest.raises(ExportError, match="canonical"):
        load_export_data(run)
    run = _run(tmp_path / "other")
    (run / "songs.json").write_text(json.dumps([_song("a", r"C:\Music\A.mp3")]), encoding="utf-8")
    with pytest.raises(ExportError, match="canonical"):
        load_export_data(run)


def test_malformed_artifacts_and_unsafe_paths_fail_before_export(tmp_path):
    run = _run(tmp_path)
    (run / "playlists.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ExportError):
        create_preview(run, "json")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "songs.json").write_text(
        json.dumps({"schema_version": "1.0", "items": [_song("a", r"C:\Music\..\secret.mp3")]}),
        encoding="utf-8",
    )
    (bad / "playlists.json").write_text(
        json.dumps({"schema_version": "1.0", "items": [_playlist("p", ["a"])]}), encoding="utf-8"
    )
    (bad / "memberships.json").write_text(
        json.dumps({"schema_version": "1.0", "items": []}), encoding="utf-8"
    )
    (bad / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "r",
                "created_at": "2026-01-01T00:00:00Z",
                "discovery_run": "d",
                "artifacts": [],
                "candidates": ["p"],
                "generator_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExportError, match="parent traversal"):
        create_preview(bad, "m3u8")


def test_failed_replace_leaves_prior_output_intact(tmp_path, monkeypatch):
    run = _run(tmp_path)
    target = tmp_path / "existing.json"
    target.write_bytes(b"old bytes")
    preview = create_preview(run, "json")

    def fail_replace(source, destination):
        raise OSError("simulated")

    monkeypatch.setattr(service.os, "replace", fail_replace)
    with pytest.raises(ExportError, match="atomically"):
        service.write_preview(preview, target)
    assert target.read_bytes() == b"old bytes"
    assert not list(tmp_path.glob(".existing.json.*.tmp"))


def test_manifest_and_unknown_playlist_members_fail_closed(tmp_path):
    run = _run(tmp_path)
    (run / "manifest.json").write_text(json.dumps({"schema_version": "9.0"}), encoding="utf-8")
    with pytest.raises(ExportError, match="canonical"):
        load_export_data(run)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "r",
                "created_at": "2026-01-01T00:00:00Z",
                "discovery_run": "d",
                "artifacts": [],
                "candidates": ["p"],
                "generator_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    (run / "playlists.json").write_text(
        json.dumps({"schema_version": "1.0", "items": [_playlist("p", ["missing"])]}),
        encoding="utf-8",
    )
    with pytest.raises(ExportError, match="canonical"):
        load_export_data(run)
