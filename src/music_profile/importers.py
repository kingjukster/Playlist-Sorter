from __future__ import annotations

import csv
import plistlib
from pathlib import Path
from typing import Any

from music_profile.util import apple_id_from_url, duration_to_ms, read_json, stable_id


def _playlist_id(name: str, url: str = "", supplied: str = "") -> str:
    if supplied:
        return supplied
    if "/playlist/" in url:
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        if tail:
            return tail
    return stable_id("playlist", name, url)


def _track_from_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    title = str(raw.get("title") or raw.get("name") or raw.get("Name") or "").strip()
    artist = str(raw.get("artist") or raw.get("Artist") or "").strip()
    album = str(raw.get("album") or raw.get("Album") or "").strip()
    source_url = str(raw.get("source_url") or raw.get("song_url") or raw.get("url") or "")
    album_url = str(raw.get("album_url") or "")
    song_id = str(raw.get("apple_music_id") or apple_id_from_url(source_url, "song"))
    album_apple_id = str(
        raw.get("album_apple_music_id")
        or raw.get("album_apple_id")
        or apple_id_from_url(album_url, "album")
    )
    release_date = str(raw.get("release_date") or "")
    raw_year = raw.get("release_year") or raw.get("Year")
    release_year = int(raw_year) if str(raw_year).isdigit() else None
    if release_year is None and len(release_date) >= 4 and release_date[:4].isdigit():
        release_year = int(release_date[:4])
    raw_genres = raw.get("genres") or raw.get("Genre") or []
    if isinstance(raw_genres, str):
        genres = [item.strip() for item in raw_genres.replace(";", ",").split(",") if item.strip()]
    else:
        genres = list(raw_genres)
    return {
        "track_id": str(
            raw.get("track_id")
            or (f"am_song_{song_id}" if song_id else stable_id("track", title, artist, album))
        ),
        "apple_music_id": song_id,
        "title": title,
        "artist": artist,
        "album_artist": str(raw.get("album_artist") or raw.get("Album Artist") or ""),
        "album": album,
        "album_id": f"am_album_{album_apple_id}" if album_apple_id else "",
        "album_apple_music_id": album_apple_id,
        "release_date": release_date,
        "release_year": release_year,
        "track_number": raw.get("track_number") or raw.get("Track Number"),
        "disc_number": raw.get("disc_number") or raw.get("Disc Number"),
        "duration_ms": duration_to_ms(
            raw.get("duration_ms") or raw.get("duration") or raw.get("Total Time")
        ),
        "genres": genres,
        "is_explicit": raw.get("is_explicit"),
        "isrc": str(raw.get("isrc") or ""),
        "composer": str(raw.get("composer") or raw.get("Composer") or ""),
        "record_label": str(raw.get("record_label") or ""),
        "playlist_membership": [],
        "source_url": source_url,
        "album_source_url": album_url,
    }


def import_browser_json(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    entries = data if isinstance(data, list) else [data]
    playlists: list[dict[str, Any]] = []
    for entry in entries:
        meta = entry.get("playlist", entry)
        name = str(meta.get("name") or path.stem)
        url = str(meta.get("url") or "")
        playlist_id = _playlist_id(name, url, str(meta.get("playlist_id") or meta.get("id") or ""))
        tracks = [_track_from_mapping(track) for track in entry.get("tracks", [])]
        playlists.append(
            {
                "playlist_id": playlist_id,
                "name": name,
                "description": str(meta.get("description") or ""),
                "url": url,
                "track_count": len(tracks),
                "source_track_count": meta.get("source_track_count")
                or entry.get("source_track_count"),
                "is_complete": entry.get("is_complete", meta.get("is_complete", True)),
                "tracks": tracks,
                "unresolved_items": list(entry.get("unresolved_items") or []),
            }
        )
    return playlists


def import_csv(path: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("playlist_name") or path.stem).strip()
            groups.setdefault(name, []).append(_track_from_mapping(row))
    return [
        {
            "playlist_id": _playlist_id(name),
            "name": name,
            "description": "",
            "url": "",
            "track_count": len(tracks),
            "source_track_count": len(tracks),
            "is_complete": True,
            "tracks": tracks,
            "unresolved_items": [],
        }
        for name, tracks in groups.items()
    ]


def import_itunes_xml(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        library = plistlib.load(handle)
    track_map = library.get("Tracks", {})
    playlists: list[dict[str, Any]] = []
    for raw_playlist in library.get("Playlists", []):
        name = str(raw_playlist.get("Name") or "Unnamed Playlist")
        tracks: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        for item in raw_playlist.get("Playlist Items", []):
            source_id = str(item.get("Track ID", ""))
            raw = track_map.get(source_id) or track_map.get(
                int(source_id) if source_id.isdigit() else source_id
            )
            if not raw:
                unresolved.append({"type": "track_reference", "source_id": source_id})
                continue
            tracks.append(_track_from_mapping(raw))
        playlists.append(
            {
                "playlist_id": str(
                    raw_playlist.get("Playlist Persistent ID") or _playlist_id(name)
                ),
                "name": name,
                "description": str(raw_playlist.get("Description") or ""),
                "url": "",
                "track_count": len(tracks),
                "source_track_count": len(tracks),
                "is_complete": True,
                "tracks": tracks,
                "unresolved_items": unresolved,
            }
        )
    return playlists


def import_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        return import_browser_json(path)
    if suffix == ".csv":
        return import_csv(path)
    if suffix in {".xml", ".plist"}:
        return import_itunes_xml(path)
    raise ValueError(f"Unsupported import format: {path.suffix}")
