"""Read the small, versioned canonical review artifact set fail-closed."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from playlist_sorter.core.contracts import Membership, PlaylistCandidate, RunManifest, SongRecord


class CanonicalArtifactError(ValueError):
    """An artifact is absent, malformed, incompatible, or inconsistent."""


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class CanonicalRunArtifacts:
    manifest: RunManifest
    candidates: tuple[PlaylistCandidate, ...]
    memberships: tuple[Membership, ...]
    songs: tuple[SongRecord, ...]

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(candidate.candidate_id for candidate in self.candidates)

    @property
    def song_ids(self) -> frozenset[str]:
        return frozenset(song.song_id for song in self.songs)


class CanonicalRunReader:
    """Read canonical JSON artifacts, rejecting every unsupported shape/version."""

    def __init__(self, run_directory: str | Path):
        self.run_directory = Path(run_directory)

    def load(self) -> CanonicalRunArtifacts:
        manifest = self._model("manifest.json", RunManifest)
        candidates = self._collection("playlists.json", PlaylistCandidate)
        memberships = self._collection("memberships.json", Membership)
        songs = self._collection("songs.json", SongRecord)
        artifacts = CanonicalRunArtifacts(manifest, candidates, memberships, songs)
        self._validate_relationships(artifacts)
        return artifacts

    def _read_json(self, filename: str) -> Any:
        path = self.run_directory / filename
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise CanonicalArtifactError(f"cannot read canonical artifact {filename}") from error

    def _model(self, filename: str, model: type[T]) -> T:
        payload = self._read_json(filename)
        if not isinstance(payload, dict):
            raise CanonicalArtifactError(f"{filename} must be a versioned object")
        try:
            return model.model_validate(payload)
        except ValidationError as error:
            raise CanonicalArtifactError(f"invalid canonical artifact {filename}") from error

    def _collection(self, filename: str, model: type[T]) -> tuple[T, ...]:
        payload = self._read_json(filename)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "items"}
            or not isinstance(payload["items"], list)
        ):
            raise CanonicalArtifactError(f"{filename} must contain only schema_version and items")
        if payload["schema_version"] != "1.0":
            raise CanonicalArtifactError(f"unsupported schema version in {filename}")
        try:
            return tuple(model.model_validate(item) for item in payload["items"])
        except (ValidationError, TypeError) as error:
            raise CanonicalArtifactError(f"invalid canonical artifact {filename}") from error

    @staticmethod
    def _validate_relationships(artifacts: CanonicalRunArtifacts) -> None:
        candidate_ids = artifacts.candidate_ids
        song_ids = artifacts.song_ids
        if len(candidate_ids) != len(artifacts.candidates) or len(song_ids) != len(artifacts.songs):
            raise CanonicalArtifactError("canonical candidates and songs must have unique IDs")
        if set(artifacts.manifest.candidates) != candidate_ids:
            raise CanonicalArtifactError("manifest candidates do not match playlists")
        for candidate in artifacts.candidates:
            evidence_ids = (
                set(candidate.member_song_ids)
                | set(candidate.representative_song_ids)
                | set(candidate.core_song_ids)
                | set(candidate.boundary_song_ids)
                | set(candidate.excluded_song_ids)
            )
            if not evidence_ids <= song_ids:
                raise CanonicalArtifactError("candidate references an unknown song")
        for membership in artifacts.memberships:
            if membership.candidate_id not in candidate_ids or membership.song_id not in song_ids:
                raise CanonicalArtifactError("membership references an unknown candidate or song")


def load_canonical_run(run_directory: str | Path) -> CanonicalRunArtifacts:
    """Convenience entry point for shared services and thin presentation layers."""
    return CanonicalRunReader(run_directory).load()
