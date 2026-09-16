"""Fail-closed YouTube audio acquisition for explicitly authorized videos.

This module never searches for media, consumes playlists, imports browser
cookies, logs in, bypasses geographic restrictions, or handles DRM. Every URL
must be supplied by the user with an affirmative rights declaration.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Literal
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SUPPORTED_AUDIO_FORMATS = ("flac", "wav")
YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"})
RightsBasis = Literal["owned", "permission", "creative_commons", "public_domain"]
Runner = Callable[..., subprocess.CompletedProcess[str]]


class AcquisitionError(ValueError):
    """The manifest, runtime, or acquisition result is unsafe or invalid."""


class AcquisitionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    rights_basis: RightsBasis
    rights_note: str = Field(min_length=3, max_length=500)
    authorization_confirmed: Literal[True]
    title: str | None = None
    artist: str | None = None

    @field_validator("url")
    @classmethod
    def individual_youtube_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or host not in YOUTUBE_HOSTS
            or parsed.username
            or parsed.password
        ):
            raise ValueError("URL must be an HTTPS youtube.com or youtu.be video URL")
        query = parse_qs(parsed.query)
        if "list" in query:
            raise ValueError("playlist and mix URLs are not accepted; provide individual videos")
        if host == "youtu.be":
            video_id = parsed.path.strip("/")
        elif parsed.path == "/watch":
            video_id = query.get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/live/")):
            video_id = parsed.path.split("/", 2)[2]
        else:
            video_id = ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
            raise ValueError("URL must identify one concrete YouTube video")
        return value.strip()


class AcquisitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    items: list[AcquisitionEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_urls(self) -> "AcquisitionManifest":
        urls = [item.url for item in self.items]
        duplicates = sorted({url for url in urls if urls.count(url) > 1})
        if duplicates:
            raise ValueError(f"duplicate URLs are not allowed: {duplicates}")
        return self


def _truthy(value: str) -> bool:
    return value.strip().casefold() in {"true", "yes", "1"}


def load_acquisition_manifest(path: str | Path) -> AcquisitionManifest:
    """Load strict JSON, YAML, or CSV without accepting unknown fields."""
    source = Path(path)
    try:
        if source.suffix.casefold() == ".csv":
            with source.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            payload: Any = {
                "schema_version": "1.0",
                "items": [
                    {
                        **row,
                        "authorization_confirmed": _truthy(row.get("authorization_confirmed", "")),
                        "title": row.get("title") or None,
                        "artist": row.get("artist") or None,
                    }
                    for row in rows
                ],
            }
        elif source.suffix.casefold() in {".yaml", ".yml"}:
            import yaml  # type: ignore[import-untyped]

            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        elif source.suffix.casefold() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise AcquisitionError("manifest must use .csv, .json, .yaml, or .yml")
        return AcquisitionManifest.model_validate(payload)
    except AcquisitionError:
        raise
    except (OSError, csv.Error, json.JSONDecodeError, ValidationError) as exc:
        raise AcquisitionError(f"invalid acquisition manifest: {source}") from exc


def _download_id(url: str) -> str:
    return f"youtube-{hashlib.sha256(url.encode()).hexdigest()[:16]}"


def build_ytdlp_command(
    entry: AcquisitionEntry,
    *,
    output_dir: str | Path,
    audio_format: str,
) -> tuple[str, ...]:
    """Build an inspectable command with authentication and config disabled."""
    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        raise AcquisitionError(f"audio format must be one of: {', '.join(SUPPORTED_AUDIO_FORMATS)}")
    root = Path(output_dir).resolve(strict=False)
    media = root / "media"
    output_template = media / f"{_download_id(entry.url)}.%(ext)s"
    return (
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-config",
        "--no-playlist",
        "--no-remote-components",
        "--no-overwrites",
        "--continue",
        "--extract-audio",
        "--audio-format",
        audio_format,
        "--write-info-json",
        "--download-archive",
        str(root / "download-archive.txt"),
        "--output",
        str(output_template),
        "--print",
        "after_move:%(filepath)s",
        entry.url,
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verified_receipts(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("status") == "completed":
                latest[str(payload["url"])] = payload
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AcquisitionError("receipts.jsonl is malformed; refusing an ambiguous rerun") from exc
    return latest


def _runtime_check() -> str:
    try:
        from yt_dlp.version import __version__  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AcquisitionError("yt-dlp is not installed in this runtime") from exc
    if shutil.which("ffmpeg") is None:
        raise AcquisitionError("FFmpeg is required and must be available on PATH")
    return str(__version__)


def acquire_youtube_audio(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    audio_format: str = "flac",
    runner: Runner = subprocess.run,
    check_runtime: bool = True,
) -> dict[str, Any]:
    """Download authorized entries and append durable per-entry receipts.

    Failures are isolated per entry. A rerun skips a completed item only when
    its recorded file still exists and its SHA-256 still matches.
    """
    manifest = load_acquisition_manifest(manifest_path)
    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        raise AcquisitionError(f"audio format must be one of: {', '.join(SUPPORTED_AUDIO_FORMATS)}")
    yt_dlp_version = _runtime_check() if check_runtime else "unchecked-test-runtime"
    root = Path(output_dir).expanduser().resolve(strict=False)
    media = root / "media"
    media.mkdir(parents=True, exist_ok=True)
    receipts_path = root / "receipts.jsonl"
    prior = _verified_receipts(receipts_path)
    _atomic_text(
        root / "resolved-manifest.json",
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    started_at = datetime.now(UTC)
    for entry in manifest.items:
        download_id = _download_id(entry.url)
        target = media / f"{download_id}.{audio_format}"
        previous = prior.get(entry.url)
        if (
            previous
            and target.is_file()
            and previous.get("sha256") == _sha256(target)
            and previous.get("audio_format") == audio_format
        ):
            counts["skipped"] += 1
            continue
        command = build_ytdlp_command(entry, output_dir=root, audio_format=audio_format)
        _append_jsonl(
            root / "commands.jsonl",
            {
                "at": datetime.now(UTC),
                "download_id": download_id,
                "command": list(command),
                "url": entry.url,
            },
        )
        completed = runner(command, check=False, capture_output=True, text=True)
        if completed.returncode or not target.is_file():
            counts["failed"] += 1
            _append_jsonl(
                receipts_path,
                {
                    "at": datetime.now(UTC),
                    "download_id": download_id,
                    "status": "failed",
                    "url": entry.url,
                    "returncode": completed.returncode,
                    "error": completed.stderr[-4000:],
                },
            )
            continue
        receipt = {
            "at": datetime.now(UTC),
            "audio_format": audio_format,
            "authorization_confirmed": True,
            "download_id": download_id,
            "file": str(target),
            "rights_basis": entry.rights_basis,
            "rights_note": entry.rights_note,
            "sha256": _sha256(target),
            "source_info": str(media / f"{download_id}.info.json"),
            "status": "completed",
            "url": entry.url,
        }
        _append_jsonl(receipts_path, receipt)
        counts["completed"] += 1
    summary = {
        **counts,
        "audio_format": audio_format,
        "elapsed_seconds": (datetime.now(UTC) - started_at).total_seconds(),
        "manifest": str(Path(manifest_path).resolve(strict=False)),
        "media_directory": str(media),
        "output_directory": str(root),
        "python": sys.version.split()[0],
        "requested": len(manifest.items),
        "yt_dlp_version": yt_dlp_version,
    }
    _atomic_text(root / "run_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_text(
        root / "run_summary.md",
        "# YouTube acquisition summary\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in summary.items())
        + "\n",
    )
    return summary
