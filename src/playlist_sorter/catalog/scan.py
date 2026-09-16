"""Deterministic catalog scan with exact source-byte identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_evidence(path: Path) -> tuple[float, dict[str, str]]:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate(), {
            "sample_rate": str(handle.getframerate()), "channels": str(handle.getnchannels())
        }


def scan_library(root: str | Path) -> tuple[list[CatalogSong], list[CatalogFailure]]:
    """Scan sorted files without writing, decoding, or modifying source bytes."""
    library = Path(root)
    songs: list[CatalogSong] = []
    failures: list[CatalogFailure] = []
    if not library.is_dir():
        return [], [CatalogFailure(str(library), "invalid_library", "path is not a directory")]
    for path in sorted((item for item in library.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            duration, metadata = _wav_evidence(path) if path.suffix.lower() == ".wav" else (None, {})
            digest = _sha256(path)
            songs.append(CatalogSong(digest, str(path.resolve()), digest, path.stat().st_size, duration,
                                     path.suffix.lower().lstrip("."), metadata))
        except (OSError, wave.Error, EOFError) as exc:
            failures.append(CatalogFailure(str(path.resolve()), "corrupt_audio", str(exc)))
    return songs, failures


def write_catalog_parquet(rows: list[CatalogSong], output: str | Path) -> Path:
    """Write a deterministic Parquet catalog when pyarrow is deliberately installed."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output requires optional dependency 'pyarrow'; install it in the runtime profile") from exc
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(row) for row in sorted(rows, key=lambda row: row.song_id)])
    pq.write_table(table, target, compression="zstd", use_dictionary=False, write_statistics=False)
    return target
