"""Pure export rendering plus an atomic, opt-in write boundary.

The preview contains every byte that will be written.  This keeps the UI and
CLI honest: rendering is side-effect free, and accepting a preview is the only
operation that touches a derived output file.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from playlist_sorter.core.artifacts import canonical_json
from playlist_sorter.core.contracts import PlaylistCandidate, SongRecord
from playlist_sorter.review import load_canonical_run

ExportFormat = Literal["json", "csv", "m3u8", "html"]


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


def _display_title(song: SongRecord) -> str:
    return song.title.strip() or PureWindowsPath(song.source_path).stem or song.song_id


def _display_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = max(0, round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _html_bytes(data: ExportData) -> bytes:
    """Render a private, self-contained report without exposing source paths."""

    songs = {song.song_id: song for song in data.songs}
    playlist_sections: list[str] = []
    total_memberships = 0
    for index, playlist in enumerate(data.playlists, start=1):
        members = [songs[song_id] for song_id in playlist.member_song_ids]
        members.sort(key=lambda song: (song.artist.casefold(), _display_title(song).casefold()))
        total_memberships += len(members)
        song_rows: list[str] = []
        for position, song in enumerate(members, start=1):
            artist = song.artist.strip() or "Unknown artist"
            duration = _display_duration(song.duration_seconds)
            duration_markup = (
                f'<span class="duration">{html.escape(duration)}</span>' if duration else ""
            )
            song_rows.append(
                '<li><span class="position">'
                f"{position}</span><span><strong>{html.escape(artist)}</strong>"
                f'<span class="separator"> — </span>{html.escape(_display_title(song))}'
                f"</span>{duration_markup}</li>"
            )
        evidence = " · ".join(playlist.naming_evidence)
        evidence_markup = (
            f'<p class="evidence">Why it grouped: {html.escape(evidence)}</p>' if evidence else ""
        )
        metrics = (
            ("Stability", playlist.stability, "How consistently the same songs grouped together"),
            ("Cohesion", playlist.cohesion, "How similar the songs are within this playlist"),
            ("Separation", playlist.separation, "How distinct this playlist is from nearby songs"),
            ("Novelty", playlist.novelty, "How different this playlist is from other results"),
            ("Library coverage", playlist.coverage, "Share of the analyzed library represented"),
        )
        metric_markup = "".join(
            '<div class="metric" title="'
            f'{html.escape(description)}"><span>{html.escape(label)}</span>'
            f"<strong>{value:.0%}</strong></div>"
            for label, value, description in metrics
        )
        playlist_sections.append(
            f'<section class="playlist" id="playlist-{index}">'
            '<div class="playlist-heading"><div>'
            f'<p class="eyebrow">{html.escape(playlist.granularity.title())} playlist</p>'
            f"<h2>{html.escape(playlist.name)}</h2>"
            f'<p class="subtitle">{len(members)} songs</p></div>'
            f'<span class="number">{index:02d}</span></div>'
            f'<div class="metrics">{metric_markup}</div>{evidence_markup}'
            f'<ol class="songs">{"".join(song_rows)}</ol></section>'
        )

    empty_state = (
        '<section class="empty"><h2>No playlists passed the quality gates</h2>'
        "<p>The sorter abstained instead of presenting a low-quality grouping.</p></section>"
        if not data.playlists
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Playlist Sorter Report</title>
  <style>
    :root {{ color-scheme: dark; --bg:#10110f; --panel:#191b17; --ink:#f6f3e8;
      --muted:#aaa99f; --line:#34372f; --accent:#d7ff64; --accent2:#ff8f70; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 Inter,Segoe UI,sans-serif; }}
    main {{ width:min(1040px,calc(100% - 32px)); margin:0 auto; padding:64px 0 96px; }}
    header {{ border-bottom:1px solid var(--line); padding-bottom:32px; margin-bottom:32px; }}
    h1 {{ font-size:clamp(2.5rem,8vw,5.5rem); line-height:.92; letter-spacing:-.06em; margin:8px 0 18px; }}
    h2 {{ font-size:clamp(1.65rem,4vw,2.4rem); line-height:1.05; letter-spacing:-.035em; margin:3px 0; }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.12em; font-size:.75rem; font-weight:800; margin:0; }}
    .intro,.subtitle,.evidence {{ color:var(--muted); margin:0; }}
    .summary {{ display:flex; gap:28px; flex-wrap:wrap; margin-top:25px; }}
    .summary strong {{ display:block; font-size:1.55rem; color:var(--accent); }}
    .summary span {{ color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }}
    .playlist,.empty {{ background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:clamp(20px,4vw,38px); margin-top:22px; }}
    .playlist-heading {{ display:flex; justify-content:space-between; align-items:flex-start; gap:24px; }}
    .number {{ color:var(--accent2); font-size:2.2rem; font-weight:800; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(125px,1fr)); gap:8px; margin:25px 0 14px; }}
    .metric {{ border:1px solid var(--line); border-radius:12px; padding:11px 13px; }}
    .metric span {{ display:block; color:var(--muted); font-size:.75rem; }}
    .metric strong {{ font-size:1.15rem; }}
    .evidence {{ font-size:.85rem; margin:12px 0 20px; }}
    .songs {{ list-style:none; padding:0; margin:24px 0 0; border-top:1px solid var(--line); }}
    .songs li {{ display:grid; grid-template-columns:34px 1fr auto; gap:10px; padding:13px 2px; border-bottom:1px solid var(--line); align-items:baseline; }}
    .position,.duration {{ color:var(--muted); font-variant-numeric:tabular-nums; font-size:.85rem; }}
    .separator {{ color:var(--muted); }}
    @media (max-width:560px) {{ .songs li {{ grid-template-columns:28px 1fr; }} .duration {{ display:none; }} }}
    @media print {{ :root {{ color-scheme:light; --bg:#fff; --panel:#fff; --ink:#111; --muted:#555; --line:#ddd; --accent:#315b00; --accent2:#9b2d12; }} main {{ padding:20px 0; }} .playlist {{ break-inside:avoid; }} }}
  </style>
</head>
<body><main>
  <header><p class="eyebrow">Private local report</p><h1>Your discovered playlists</h1>
    <p class="intro">The strongest groupings found in your music, shown in an easy-to-scan format.</p>
    <div class="summary"><div><strong>{len(data.playlists)}</strong><span>playlists</span></div>
      <div><strong>{total_memberships}</strong><span>song placements</span></div>
      <div><strong>{len(data.songs)}</strong><span>songs analyzed</span></div></div>
  </header>
  {empty_state}{"".join(playlist_sections)}
</main></body>
</html>
"""
    return document.encode("utf-8")


def create_preview(run_dir: str | Path, format: ExportFormat) -> ExportPreview:
    """Render a deterministic export.  This function performs no writes."""

    if format not in {"json", "csv", "m3u8", "html"}:
        raise ExportError("format must be one of: json, csv, m3u8, html")
    data = load_export_data(run_dir)
    rows = _rows(data)
    if format == "html":
        content = _html_bytes(data)
        return ExportPreview(format, content, hashlib.sha256(content).hexdigest(), len(rows))
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
