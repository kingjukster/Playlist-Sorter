from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from music_profile.util import read_json, write_json


def validate_profile(
    profile: dict[str, Any],
    *,
    warnings: list[str] | None = None,
    unresolved_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    report_warnings = list(warnings or [])
    unresolved = list(unresolved_items or [])
    required = {
        "schema_version",
        "generated_at",
        "source",
        "user_inputs",
        "playlists",
        "tracks",
        "albums",
        "artists",
        "summary_statistics",
    }
    missing = sorted(required - set(profile))
    if missing:
        errors.append(f"Missing top-level fields: {', '.join(missing)}")
        return {
            "valid": False,
            "errors": errors,
            "warnings": report_warnings,
            "unresolved_items": unresolved,
        }
    track_ids = [item.get("track_id") for item in profile["tracks"]]
    album_ids = [item.get("album_id") for item in profile["albums"]]
    artist_ids = [item.get("artist_id") for item in profile["artists"]]
    for label, values in (("track", track_ids), ("album", album_ids), ("artist", artist_ids)):
        duplicates = sorted(
            value for value, count in Counter(values).items() if value and count > 1
        )
        if duplicates:
            errors.append(f"Duplicate {label} IDs: {', '.join(duplicates)}")
    track_set = set(track_ids)
    album_set = set(album_ids)
    for playlist in profile["playlists"]:
        missing_refs = sorted(set(playlist.get("tracks", [])) - track_set)
        if missing_refs:
            errors.append(
                f"Playlist {playlist.get('playlist_id')} has missing track refs: {missing_refs}"
            )
        if playlist.get("track_count") != len(playlist.get("tracks", [])):
            errors.append(
                f"Playlist {playlist.get('playlist_id')} track_count does not match its references"
            )
        source_count = playlist.get("source_track_count")
        if playlist.get("is_complete") is False:
            errors.append(
                f"Playlist {playlist.get('playlist_id')} is marked as an incomplete source capture"
            )
        if isinstance(source_count, int) and source_count != playlist.get("track_count", 0):
            errors.append(
                f"Playlist {playlist.get('playlist_id')} contains {playlist.get('track_count', 0)} "
                f"of {source_count} source tracks"
            )
    for track in profile["tracks"]:
        if track.get("album") and track.get("album_id") not in album_set:
            errors.append(f"Track {track.get('track_id')} has an unresolved album reference")
        if not track.get("title") or not track.get("artist"):
            unresolved.append(
                {
                    "type": "track_metadata",
                    "track_id": track.get("track_id"),
                    "title": track.get("title", ""),
                    "artist": track.get("artist", ""),
                }
            )
    expected = {
        "total_playlists": len(profile["playlists"]),
        "total_unique_tracks": len(profile["tracks"]),
        "total_unique_albums": len(profile["albums"]),
        "total_unique_artists": len(profile["artists"]),
    }
    for key, value in expected.items():
        if profile["summary_statistics"].get(key) != value:
            errors.append(f"summary_statistics.{key} is inconsistent: expected {value}")
    if unresolved:
        report_warnings.append(f"{len(unresolved)} item(s) could not be fully resolved")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": report_warnings,
        "unresolved_items": unresolved,
    }


def validate_file(path: Path, report_path: Path | None = None) -> dict[str, Any]:
    report = validate_profile(read_json(path))
    if report_path:
        write_json(report_path, report)
    return report
