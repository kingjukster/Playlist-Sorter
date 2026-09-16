"""Pure export rendering plus an atomic, opt-in write boundary.

The preview contains every byte that will be written.  This keeps the UI and
CLI honest: rendering is side-effect free, and accepting a preview is the only
operation that touches a derived output file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from playlist_sorter.core.artifacts import canonical_json
from playlist_sorter.core.contracts import PlaylistCandidate, SongRecord
from playlist_sorter.review import load_canonical_run

ExportFormat = Literal["json", "csv", "m3u8"]


class ExportError(ValueError):
    """A run artifact or requested export is invalid and must not be written."""


@dataclass(frozen=True)
class ExportData:
    """Validated canonical inputs; source media is never opened or changed."""

    run_dir: Path
    songs: tuple[SongRecord, ...]
    playlists: tuple[PlaylistCandidate, ...]


@dataclass(frozen=True)
class ExportPreview:
    """An immutable rendering which callers may display before writing."""

    format: ExportFormat
    content: bytes
    sha256: str
    row_count: int

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


def load_export_data(run_dir: str | Path) -> ExportData:
    """Read the same strict canonical artifact set used by the review surface."""

    root = Path(run_dir).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ExportError("run directory does not exist")
    try:
        artifacts = load_canonical_run(root)
    except ValueError as exc:
        raise ExportError("invalid canonical run artifacts") from exc
    return ExportData(
        root,
        tuple(sorted(artifacts.songs, key=lambda song: song.song_id)),
        tuple(sorted(artifacts.candidates, key=lambda item: item.candidate_id)),
    )


def normalize_media_path(source_path: str) -> str:
    """Produce a stable M3U path without resolving, touching, or changing media."""

    if not isinstance(source_path, str) or not source_path.strip() or "\x00" in source_path:
        raise ExportError("source path must be a non-empty path")
    # PureWindowsPath handles paths produced on the supported Windows host while
    # the slash form makes the derived text portable and deterministic.
    path = PureWindowsPath(source_path)
    if any(part == ".." for part in path.parts):
        raise ExportError("source path must not contain parent traversal")
    normalized = path.as_posix()
    if not normalized or normalized == ".":
        raise ExportError("source path is not usable")
    return normalized


def _rows(data: ExportData) -> list[dict[str, Any]]:
    songs = {song.song_id: song for song in data.songs}
    rows: list[dict[str, Any]] = []
    for playlist in data.playlists:
        for song_id in sorted(playlist.member_song_ids):
            song = songs[song_id]
            rows.append(
                {
                    "candidate_id": playlist.candidate_id,
                    "playlist_name": playlist.name,
                    "song_id": song.song_id,
                    "source_path": normalize_media_path(song.source_path),
                    "title": song.title,
                    "artist": song.artist,
                    "duration_seconds": song.duration_seconds,
                }
            )
    return rows


def _json_bytes(rows: list[dict[str, Any]]) -> bytes:
    payload = {"schema_version": "1.0", "playlists": rows}
    return (canonical_json(payload) + "\n").encode("utf-8")


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "candidate_id",
        "playlist_name",
        "song_id",
        "source_path",
        "title",
        "artist",
        "duration_seconds",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _m3u8_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = ["#EXTM3U"]
    current_candidate: str | None = None
    for row in rows:
        if row["candidate_id"] != current_candidate:
            current_candidate = str(row["candidate_id"])
            lines.append(f"#PLAYLIST:{row['playlist_name']}")
        duration = -1 if row["duration_seconds"] is None else int(float(row["duration_seconds"]))
        label = " - ".join(part for part in (row["artist"], row["title"]) if part) or row["song_id"]
        lines.extend((f"#EXTINF:{duration},{label}", str(row["source_path"])))
    return ("\n".join(lines) + "\n").encode("utf-8")


def create_preview(run_dir: str | Path, format: ExportFormat) -> ExportPreview:
    """Render a deterministic export.  This function performs no writes."""

    if format not in {"json", "csv", "m3u8"}:
        raise ExportError("format must be one of: json, csv, m3u8")
    rows = _rows(load_export_data(run_dir))
    renderers = {"json": _json_bytes, "csv": _csv_bytes, "m3u8": _m3u8_bytes}
    content = renderers[format](rows)
    return ExportPreview(format, content, hashlib.sha256(content).hexdigest(), len(rows))


def write_preview(preview: ExportPreview, destination: str | Path) -> Path:
    """Atomically replace one derived file after validating an unchanged preview."""

    if hashlib.sha256(preview.content).hexdigest() != preview.sha256:
        raise ExportError("preview checksum does not match its content")
    output = Path(destination).expanduser().resolve(strict=False)
    if output.name in {"", "."} or output.exists() and output.is_dir():
        raise ExportError("destination must be a file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(preview.content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise ExportError(f"could not atomically write export: {output}") from exc
    return output


def export_run(run_dir: str | Path, format: ExportFormat, destination: str | Path) -> ExportPreview:
    """Create a preview and accept it in one explicit, deterministic call."""

    preview = create_preview(run_dir, format)
    write_preview(preview, destination)
    return preview
