"""Deterministic catalog scan with exact source-byte identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

SUPPORTED_EXTENSIONS = frozenset({".mp3", ".ogg", ".flac", ".wav", ".m4a"})


@dataclass(frozen=True)
class CatalogFailure:
    path: str
    code: str
    detail: str


@dataclass(frozen=True)
class CatalogSong:
    song_id: str
    source_path: str
    sha256: str
    size_bytes: int
    duration_seconds: float | None
    codec: str
    metadata: dict[str, str]
    chromaprint: str | None = None
    chromaprint_status: str = "not_attempted"
    variant_group_id: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_evidence(path: Path) -> tuple[float, dict[str, str]]:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate(), {
            "sample_rate": str(handle.getframerate()),
            "channels": str(handle.getnchannels()),
        }


def _chromaprint(path: Path) -> tuple[str | None, str]:
    """Invoke fpcalc only when it is installed; scanning remains offline/read-only."""
    executable = shutil.which("fpcalc")
    if executable is None:
        return None, "unavailable: fpcalc is not on PATH"
    try:
        completed = subprocess.run(
            [executable, "-json", str(path)], check=False, capture_output=True, text=True
        )
    except OSError as exc:
        return None, f"unavailable: {exc}"
    if completed.returncode:
        return None, f"failed: {completed.stderr.strip() or 'fpcalc returned non-zero'}"
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("fingerprint"):
        return str(payload["fingerprint"]), "available"
    for line in completed.stdout.splitlines():
        if line.startswith("FINGERPRINT="):
            return line.partition("=")[2], "available"
    return None, "failed: fpcalc returned no fingerprint"


def scan_library(root: str | Path) -> tuple[list[CatalogSong], list[CatalogFailure]]:
    """Scan sorted files without writing, decoding, or modifying source bytes."""
    library = Path(root)
    songs: list[CatalogSong] = []
    failures: list[CatalogFailure] = []
    if not library.is_dir():
        return [], [CatalogFailure(str(library), "invalid_library", "path is not a directory")]
    for path in sorted(
        (item for item in library.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix().casefold(),
    ):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            duration, metadata = (
                _wav_evidence(path) if path.suffix.lower() == ".wav" else (None, {})
            )
            digest = _sha256(path)
            fingerprint, fingerprint_status = _chromaprint(path)
            song = CatalogSong(
                digest,
                str(path.resolve()),
                digest,
                path.stat().st_size,
                duration,
                path.suffix.lower().lstrip("."),
                metadata,
                fingerprint,
                fingerprint_status,
            )
            songs.append(song)
        except (OSError, wave.Error, EOFError) as exc:
            failures.append(CatalogFailure(str(path.resolve()), "corrupt_audio", str(exc)))
    return group_musical_variants(songs), failures


def group_musical_variants(
    songs: list[CatalogSong], *, duration_tolerance_seconds: float = 2.0
) -> list[CatalogSong]:
    """Assign stable variant groups using exact bytes or Chromaprint plus duration.

    SHA-256 remains the song/file identity. Distinct encodes share a group only
    when their fingerprints match and their measured durations are sufficiently
    close; missing fingerprints fail closed to exact-byte groups.
    """
    if duration_tolerance_seconds < 0:
        raise ValueError("duration tolerance must be non-negative")
    groups: list[tuple[str, float | None, str]] = []
    result: list[CatalogSong] = []
    for song in songs:
        group_id = f"exact-{song.sha256}"
        if song.chromaprint:
            for fingerprint, duration, candidate_group in groups:
                duration_matches = (
                    duration is None
                    or song.duration_seconds is None
                    or abs(duration - song.duration_seconds)
                    <= max(duration_tolerance_seconds, 0.02 * max(duration, song.duration_seconds))
                )
                if fingerprint == song.chromaprint and duration_matches:
                    group_id = candidate_group
                    break
            else:
                identity = f"{song.chromaprint}|{song.duration_seconds or 'unknown'}"
                group_id = f"variant-{hashlib.sha256(identity.encode()).hexdigest()}"
                groups.append((song.chromaprint, song.duration_seconds, group_id))
        result.append(
            CatalogSong(
                **{
                    **asdict(song),
                    "variant_group_id": group_id,
                }
            )
        )
    return result


def write_catalog_parquet(rows: list[CatalogSong], output: str | Path) -> Path:
    """Write a deterministic Parquet catalog when pyarrow is deliberately installed."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires optional dependency 'pyarrow'; install it in the runtime profile"
        ) from exc
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(row) for row in sorted(rows, key=lambda row: row.song_id)])
    pq.write_table(table, target, compression="zstd", use_dictionary=False, write_statistics=False)
    return target
