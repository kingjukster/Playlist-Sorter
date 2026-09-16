from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from music_profile.enrichment import enrich_tracks, fetch_album_track_listings
from music_profile.importers import import_path
from music_profile.util import normalized, read_json, stable_id, write_json


def _merge_track(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key == "playlist_membership" or key == "genres":
            existing[key] = sorted(set([*existing[key], *value]))
        elif existing.get(key) in (None, "", []) and value not in (None, "", []):
            existing[key] = value


def _build_albums(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        if not track["album"]:
            continue
        album_id = track["album_id"] or stable_id(
            "album", track["album"], track["album_artist"] or track["artist"]
        )
        track["album_id"] = album_id
        groups[album_id].append(track)
    albums: list[dict[str, Any]] = []
    for album_id, members in groups.items():
        first = members[0]
        years = [track["release_year"] for track in members if track["release_year"]]
        dates = [track["release_date"] for track in members if track["release_date"]]
        genres = sorted({genre for track in members for genre in track["genres"]})
        albums.append(
            {
                "album_id": album_id,
                "apple_music_id": first.get("album_apple_music_id", ""),
                "title": first["album"],
                "artist": first["album_artist"] or first["artist"],
                "release_date": min(dates) if dates else "",
                "release_year": min(years) if years else None,
                "genres": genres,
                "track_count": None,
                "record_label": next(
                    (track["record_label"] for track in members if track["record_label"]), ""
                ),
                "tracks_in_user_dataset": sorted(track["track_id"] for track in members),
                "dataset_track_count": len(members),
                "dataset_coverage_ratio": None,
                "source_url": next(
                    (
                        track["album_source_url"]
                        for track in members
                        if track.get("album_source_url")
                    ),
                    "",
                ),
                "full_track_listing": [],
            }
        )
    return sorted(albums, key=lambda item: (normalized(item["artist"]), normalized(item["title"])))


def _build_artists(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        if track["artist"]:
            groups[normalized(track["artist"])].append(track)
    artists: list[dict[str, Any]] = []
    for _, members in sorted(groups.items()):
        first = members[0]
        artists.append(
            {
                "artist_id": stable_id("artist", first["artist"]),
                "apple_music_id": "",
                "name": first["artist"],
                "genres": sorted({genre for track in members for genre in track["genres"]}),
                "track_count_in_dataset": len(members),
                "album_count_in_dataset": len(
                    {track["album_id"] for track in members if track["album_id"]}
                ),
                "playlist_count": len(
                    {playlist for track in members for playlist in track["playlist_membership"]}
                ),
            }
        )
    return artists


def _statistics(
    playlists: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    albums: list[dict[str, Any]],
    artists: list[dict[str, Any]],
) -> dict[str, Any]:
    years = [track["release_year"] for track in tracks if track["release_year"]]
    by_year = Counter(str(year) for year in years)
    by_decade = Counter(f"{year // 10 * 10}s" for year in years)
    by_genre = Counter(genre for track in tracks for genre in track["genres"])
    duplicates = [
        {"track_id": track["track_id"], "playlists": track["playlist_membership"]}
        for track in tracks
        if len(track["playlist_membership"]) > 1
    ]
    return {
        "total_playlists": len(playlists),
        "total_unique_tracks": len(tracks),
        "total_unique_albums": len(albums),
        "total_unique_artists": len(artists),
        "tracks_by_release_year": dict(sorted(by_year.items())),
        "tracks_by_decade": dict(sorted(by_decade.items())),
        "tracks_by_genre": dict(by_genre.most_common()),
        "top_artists_by_track_count": [
            {"artist": artist["name"], "track_count": artist["track_count_in_dataset"]}
            for artist in sorted(
                artists,
                key=lambda item: (-item["track_count_in_dataset"], normalized(item["name"])),
            )[:25]
        ],
        "top_albums_by_track_count": [
            {
                "album": album["title"],
                "artist": album["artist"],
                "track_count": album["dataset_track_count"],
            }
            for album in sorted(
                albums,
                key=lambda item: (
                    -item["dataset_track_count"],
                    normalized(item["artist"]),
                    normalized(item["title"]),
                ),
            )[:25]
        ],
        "albums_with_highest_dataset_coverage_ratio": [
            {
                "album_id": album["album_id"],
                "album": album["title"],
                "coverage_ratio": album["dataset_coverage_ratio"],
            }
            for album in sorted(
                (item for item in albums if item["dataset_coverage_ratio"] is not None),
                key=lambda item: (-item["dataset_coverage_ratio"], normalized(item["title"])),
            )[:25]
        ],
        "artists_appearing_across_multiple_playlists": [
            artist["artist_id"] for artist in artists if artist["playlist_count"] > 1
        ],
        "albums_appearing_across_multiple_playlists": [
            album["album_id"]
            for album in albums
            if len(
                {
                    playlist
                    for track in tracks
                    if track["album_id"] == album["album_id"]
                    for playlist in track["playlist_membership"]
                }
            )
            > 1
        ],
        "duplicate_tracks_across_playlists": duplicates,
        "average_release_year": round(statistics.mean(years), 2) if years else None,
        "median_release_year": statistics.median(years) if years else None,
    }


def build_profile(
    input_paths: list[Path],
    *,
    manual_preferences: Path | None = None,
    enrich: bool = False,
    country: str = "us",
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    imported = [playlist for path in input_paths for playlist in import_path(path)]
    canonical: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    playlist_records: list[dict[str, Any]] = []
    for playlist in imported:
        refs: list[str] = []
        unresolved.extend(playlist.pop("unresolved_items", []))
        for track in playlist.pop("tracks"):
            track["playlist_membership"] = [playlist["playlist_id"]]
            key = track["track_id"]
            if key in canonical:
                _merge_track(canonical[key], track)
            else:
                canonical[key] = track
            refs.append(key)
        playlist["tracks"] = refs
        playlist["track_count"] = len(refs)
        playlist_records.append(playlist)
    tracks = sorted(canonical.values(), key=lambda item: item["track_id"])
    warnings: list[str] = []
    if enrich:
        warnings.extend(enrich_tracks(tracks, country=country))
    albums = _build_albums(tracks)
    for track in tracks:
        track.pop("album_source_url", None)
    if enrich:
        warnings.extend(fetch_album_track_listings(albums, country=country))
    artists = _build_artists(tracks)
    prefs = {"favorite_albums": [], "disliked_albums": [], "album_ratings": []}
    if manual_preferences and manual_preferences.exists():
        raw_prefs = read_json(manual_preferences)
        prefs["favorite_albums"] = raw_prefs.get("favorite_albums", [])
        prefs["disliked_albums"] = raw_prefs.get("disliked_albums", [])
        prefs["album_ratings"] = raw_prefs.get("album_ratings", [])
    limitations = [
        (
            "Private-library extraction uses a user-created browser capture or exported "
            "library file; credentials and session tokens are never collected."
        ),
        (
            "Apple catalog lookup may omit private uploads, removed releases, credits, "
            "editorial notes, or region-specific metadata."
        ),
    ]
    if not enrich:
        limitations.append("Public Apple catalog enrichment was not requested for this build.")
    profile = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "platform": "Apple Music",
            "collection_method": "local browser capture and/or exported file",
            "limitations": limitations,
        },
        "user_inputs": prefs,
        "playlists": sorted(playlist_records, key=lambda item: item["playlist_id"]),
        "tracks": tracks,
        "albums": albums,
        "artists": artists,
        "summary_statistics": _statistics(playlist_records, tracks, albums, artists),
    }
    return profile, warnings, unresolved


def write_summary(profile: dict[str, Any], path: Path) -> None:
    stats = profile["summary_statistics"]
    lines = [
        "# Music profile source summary",
        "",
        f"Generated: {profile['generated_at']}",
        "",
        (
            "This is an objective inventory of the supplied Apple Music data. "
            "It does not make taste or recommendation claims."
        ),
        "",
        "## Dataset",
        "",
        f"- Playlists: {stats['total_playlists']}",
        f"- Unique tracks: {stats['total_unique_tracks']}",
        f"- Unique albums: {stats['total_unique_albums']}",
        f"- Unique artists: {stats['total_unique_artists']}",
        f"- Average release year: {stats['average_release_year']}",
        f"- Median release year: {stats['median_release_year']}",
        "",
        "## Top artists by included tracks",
        "",
    ]
    lines.extend(
        f"- {item['artist']}: {item['track_count']}"
        for item in stats["top_artists_by_track_count"][:15]
    )
    lines.extend(["", "## Top albums by included tracks", ""])
    lines.extend(
        f"- {item['artist']} — {item['album']}: {item['track_count']}"
        for item in stats["top_albums_by_track_count"][:15]
    )
    lines.extend(["", "## Recorded limitations", ""])
    lines.extend(f"- {item}" for item in profile["source"]["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(profile: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "music_profile_source.json", profile)
    write_summary(profile, output_dir / "music_profile_summary.md")
