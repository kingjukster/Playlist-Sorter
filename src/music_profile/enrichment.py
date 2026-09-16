from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOOKUP_URL = "https://itunes.apple.com/lookup"


def _chunks(values: list[str], size: int = 50) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _lookup(
    ids: list[str], *, entity: str | None = None, country: str = "us"
) -> list[dict[str, Any]]:
    if not ids:
        return []
    query: dict[str, str] = {"id": ",".join(ids), "country": country}
    if entity:
        query["entity"] = entity
    request = Request(
        f"{LOOKUP_URL}?{urlencode(query)}",
        headers={"User-Agent": "music-profile/0.1 (+local personal export)"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS Apple endpoint
        return json.load(response).get("results", [])


def enrich_tracks(tracks: list[dict[str, Any]], *, country: str = "us") -> list[str]:
    warnings: list[str] = []
    by_id = {track["apple_music_id"]: track for track in tracks if track["apple_music_id"]}
    for batch in _chunks(sorted(by_id)):
        try:
            results = _lookup(batch, country=country)
        except (OSError, TimeoutError, ValueError) as error:
            warnings.append(f"Apple lookup failed for {len(batch)} track IDs: {error}")
            continue
        returned: set[str] = set()
        for item in results:
            item_id = str(item.get("trackId") or "")
            if not item_id or item_id not in by_id:
                continue
            returned.add(item_id)
            track = by_id[item_id]
            track["title"] = item.get("trackName") or track["title"]
            track["artist"] = item.get("artistName") or track["artist"]
            track["album_artist"] = item.get("collectionArtistName") or track["album_artist"]
            track["album"] = item.get("collectionName") or track["album"]
            collection_id = str(item.get("collectionId") or "")
            if collection_id:
                track["album_apple_music_id"] = collection_id
                track["album_id"] = f"am_album_{collection_id}"
            released = str(item.get("releaseDate") or "")[:10]
            track["release_date"] = released or track["release_date"]
            track["release_year"] = (
                int(released[:4])
                if len(released) >= 4 and released[:4].isdigit()
                else track["release_year"]
            )
            track["track_number"] = item.get("trackNumber") or track["track_number"]
            track["disc_number"] = item.get("discNumber") or track["disc_number"]
            track["duration_ms"] = item.get("trackTimeMillis") or track["duration_ms"]
            genre = item.get("primaryGenreName")
            if genre:
                track["genres"] = sorted(set([*track["genres"], genre]))
            advisory = item.get("trackExplicitness")
            if advisory:
                track["is_explicit"] = advisory == "explicit"
            track["isrc"] = item.get("isrc") or track["isrc"]
        missing = sorted(set(batch) - returned)
        if missing:
            warnings.append(f"Apple lookup returned no metadata for {len(missing)} track IDs")
        time.sleep(0.05)
    return warnings


def fetch_album_track_listings(
    albums: list[dict[str, Any]], *, minimum_dataset_tracks: int = 3, country: str = "us"
) -> list[str]:
    warnings: list[str] = []
    eligible = {
        album["apple_music_id"]: album
        for album in albums
        if album["apple_music_id"] and album["dataset_track_count"] >= minimum_dataset_tracks
    }
    for album_id, album in sorted(eligible.items()):
        try:
            results = _lookup([album_id], entity="song", country=country)
        except (OSError, TimeoutError, ValueError) as error:
            warnings.append(f"Album lookup failed for {album_id}: {error}")
            continue
        songs = [item for item in results if item.get("wrapperType") == "track"]
        songs.sort(key=lambda item: (item.get("discNumber") or 0, item.get("trackNumber") or 0))
        album["full_track_listing"] = [
            {"track_number": item.get("trackNumber"), "title": item.get("trackName") or ""}
            for item in songs
        ]
        if songs:
            album["track_count"] = max(
                [int(item.get("trackCount") or 0) for item in songs] + [len(songs)]
            )
            album["dataset_coverage_ratio"] = round(
                album["dataset_track_count"] / album["track_count"], 6
            )
        else:
            warnings.append(f"Apple lookup returned no track listing for album {album_id}")
        time.sleep(0.05)
    return warnings
