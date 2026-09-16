from __future__ import annotations

import plistlib
from pathlib import Path

from music_profile.builder import build_profile
from music_profile.importers import import_csv, import_itunes_xml
from music_profile.musickit import fetch_playlist, playlist_reference
from music_profile.util import apple_id_from_url, duration_to_ms, write_json
from music_profile.validation import validate_profile


def test_url_ids_and_duration() -> None:
    assert apple_id_from_url("https://music.apple.com/us/song/name/123", "song") == "123"
    assert apple_id_from_url("https://music.apple.com/us/album/name/456?i=123", "album") == "456"
    assert duration_to_ms("1:02:03") == 3_723_000
    assert duration_to_ms("3:05") == 185_000


def test_playlist_reference_distinguishes_library_and_replay() -> None:
    assert playlist_reference("https://music.apple.com/us/library/playlist/p.private") == (
        "p.private",
        "library",
    )
    assert playlist_reference(
        "https://music.apple.com/us/playlist/replay-2026/pl.rp-example?l=en-US"
    ) == ("pl.rp-example", "catalog")


def test_catalog_playlist_fetch_paginates(monkeypatch) -> None:
    responses = {
        "/v1/catalog/us/playlists/pl.rp-example": {
            "data": [{"attributes": {"name": "Replay", "description": "Favorites"}}]
        },
        "/v1/catalog/us/playlists/pl.rp-example/tracks?limit=100": {
            "data": [
                {
                    "id": "1",
                    "type": "songs",
                    "attributes": {"name": "One", "artistName": "Artist"},
                }
            ],
            "next": "/next-page",
        },
        "/next-page": {
            "data": [
                {
                    "id": "2",
                    "type": "songs",
                    "attributes": {"name": "Two", "artistName": "Artist"},
                }
            ]
        },
    }

    monkeypatch.setattr("music_profile.musickit._request", lambda path, *_args: responses[path])
    payload = fetch_playlist("pl.rp-example", "developer-token")

    assert payload["playlist"]["source_kind"] == "catalog"
    assert payload["source_track_count"] == 2
    assert [track["apple_music_id"] for track in payload["tracks"]] == ["1", "2"]


def test_build_deduplicates_and_computes_membership(tmp_path: Path) -> None:
    track = {
        "title": "Song",
        "artist": "Artist",
        "album": "Album",
        "song_url": "https://music.apple.com/us/song/song/123",
        "album_url": "https://music.apple.com/us/album/album/456?i=123",
    }
    paths = []
    for playlist_id in ("p.one", "p.two"):
        path = tmp_path / f"{playlist_id}.json"
        write_json(
            path, {"playlist": {"playlist_id": playlist_id, "name": playlist_id}, "tracks": [track]}
        )
        paths.append(path)

    profile, warnings, unresolved = build_profile(paths)

    assert not warnings
    assert not unresolved
    assert len(profile["tracks"]) == 1
    assert profile["tracks"][0]["playlist_membership"] == ["p.one", "p.two"]
    assert len(profile["summary_statistics"]["duplicate_tracks_across_playlists"]) == 1
    assert validate_profile(profile)["valid"] is True


def test_incomplete_browser_capture_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    write_json(
        path,
        {
            "playlist": {
                "playlist_id": "p.large",
                "name": "Large",
                "source_track_count": 785,
                "is_complete": False,
            },
            "tracks": [{"title": "Visible", "artist": "Artist"}],
        },
    )
    profile, warnings, unresolved = build_profile([path])
    report = validate_profile(profile, warnings=warnings, unresolved_items=unresolved)
    assert report["valid"] is False
    assert any("incomplete source capture" in error for error in report["errors"])
    assert any("of 785 source tracks" in error for error in report["errors"])


def test_csv_groups_playlists(tmp_path: Path) -> None:
    path = tmp_path / "tracks.csv"
    path.write_text(
        "playlist_name,title,artist,album,duration\nA,One,Artist,Album,3:00\nB,Two,Other,Else,2:00\n",
        encoding="utf-8",
    )
    playlists = import_csv(path)
    assert [playlist["name"] for playlist in playlists] == ["A", "B"]
    assert playlists[0]["tracks"][0]["duration_ms"] == 180_000


def test_itunes_xml_resolves_playlist_items(tmp_path: Path) -> None:
    path = tmp_path / "Library.xml"
    payload = {
        "Tracks": {"1": {"Track ID": 1, "Name": "Song", "Artist": "Artist", "Album": "Album"}},
        "Playlists": [{"Name": "Playlist", "Playlist Items": [{"Track ID": 1}]}],
    }
    with path.open("wb") as handle:
        plistlib.dump(payload, handle)
    playlists = import_itunes_xml(path)
    assert playlists[0]["track_count"] == 1
    assert playlists[0]["tracks"][0]["title"] == "Song"
