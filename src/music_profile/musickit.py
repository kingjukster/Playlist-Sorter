from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from music_profile.util import duration_to_ms

API_ROOT = "https://api.music.apple.com"


def playlist_reference(reference: str) -> tuple[str, str]:
    """Return an Apple playlist ID and its API kind (library or catalog)."""
    value = urlsplit(reference).path.rstrip("/").rsplit("/", 1)[-1]
    if value.startswith("p."):
        return value, "library"
    if value.startswith("pl."):
        return value, "catalog"
    raise ValueError("Expected an Apple Music playlist ID beginning with 'p.' or 'pl.'")


def playlist_id_from_reference(reference: str) -> str:
    return playlist_reference(reference)[0]


def _request(path: str, developer_token: str, user_token: str = "") -> dict[str, Any]:
    url = path if path.startswith("https://") else urljoin(API_ROOT, path)
    headers = {
        "Authorization": f"Bearer {developer_token}",
        "Accept": "application/json",
        "User-Agent": "music-profile/0.1 (+local personal export)",
    }
    if user_token:
        headers["Music-User-Token"] = user_token
    request = Request(
        url,
        headers=headers,
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - Apple HTTPS API only
        return json.load(response)


def _track(resource: dict[str, Any]) -> dict[str, Any]:
    attributes = resource.get("attributes", {})
    play_params = attributes.get("playParams", {})
    resource_id = resource.get("id") if resource.get("type") == "songs" else ""
    catalog_id = str(play_params.get("catalogId") or resource_id or "")
    album_data = ((resource.get("relationships") or {}).get("albums") or {}).get("data") or []
    album_id = str(album_data[0].get("id") or "") if album_data else ""
    source_url = str(attributes.get("url") or "")
    return {
        "apple_music_id": catalog_id,
        "track_id": f"am_song_{catalog_id}" if catalog_id else "",
        "title": attributes.get("name") or "",
        "artist": attributes.get("artistName") or "",
        "album_artist": attributes.get("albumArtistName") or "",
        "album": attributes.get("albumName") or "",
        "album_apple_music_id": album_id,
        "release_date": attributes.get("releaseDate") or "",
        "track_number": attributes.get("trackNumber"),
        "disc_number": attributes.get("discNumber"),
        "duration_ms": duration_to_ms(attributes.get("durationInMillis")),
        "genres": list(attributes.get("genreNames") or []),
        "is_explicit": attributes.get("contentRating") == "explicit"
        if attributes.get("contentRating")
        else None,
        "isrc": attributes.get("isrc") or "",
        "composer": attributes.get("composerName") or "",
        "record_label": "",
        "source_url": source_url,
    }


def fetch_playlist(
    reference: str,
    developer_token: str,
    user_token: str = "",
    country: str = "us",
) -> dict[str, Any]:
    """Fetch every page of a library or catalog playlist without exposing tokens."""
    playlist_id, kind = playlist_reference(reference)
    if kind == "library" and not user_token:
        raise ValueError("A Music User Token is required for a private library playlist")
    if kind == "library":
        resource_path = f"/v1/me/library/playlists/{playlist_id}"
    else:
        resource_path = f"/v1/catalog/{country.lower()}/playlists/{playlist_id}"
    metadata = _request(resource_path, developer_token, user_token)
    playlist_resource = (metadata.get("data") or [{}])[0]
    attributes = playlist_resource.get("attributes", {})
    path: str | None = f"{resource_path}/tracks?limit=100"
    tracks: list[dict[str, Any]] = []
    seen_pages: set[str] = set()
    while path:
        if path in seen_pages:
            raise ValueError(f"Apple Music pagination loop detected at {path}")
        seen_pages.add(path)
        page = _request(path, developer_token, user_token)
        tracks.extend(_track(item) for item in page.get("data", []))
        path = page.get("next")
    return {
        "playlist": {
            "playlist_id": playlist_id,
            "name": attributes.get("name") or playlist_id,
            "description": _description(attributes.get("description")),
            "url": (
                reference if reference.startswith("https://") else str(attributes.get("url") or "")
            ),
            "source_kind": kind,
            "source_track_count": len(tracks),
            "is_complete": True,
        },
        "is_complete": True,
        "source_track_count": len(tracks),
        "tracks": tracks,
    }


def _description(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("standard") or value.get("short") or "")
    return str(value or "")


def fetch_library_playlist(reference: str, developer_token: str, user_token: str) -> dict[str, Any]:
    """Backward-compatible wrapper for callers that fetch private library playlists."""
    playlist_id, kind = playlist_reference(reference)
    if kind != "library":
        raise ValueError(f"{playlist_id} is a catalog playlist; use fetch_playlist")
    return fetch_playlist(reference, developer_token, user_token)
