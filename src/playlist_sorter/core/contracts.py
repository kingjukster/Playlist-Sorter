"""Stable, JSON-friendly data contracts for Playlist Sorter."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class FeatureProfile(str, Enum):
    """Explicit feature execution profiles; unknown values are never coerced."""

    DESCRIPTORS = "descriptors"
    RESEARCH = "research"

    @property
    def requires_model_backed_features(self) -> bool:
        return self is FeatureProfile.RESEARCH

    @property
    def descriptors(self) -> bool:
        return self is FeatureProfile.DESCRIPTORS


def parse_feature_profile(value: str) -> FeatureProfile:
    """Parse one exact supported profile name without normalization."""
    if not isinstance(value, str):
        raise ValueError("feature profile must be an exact supported string")
    try:
        return FeatureProfile(value)
    except ValueError as exc:
        raise ValueError(f"unsupported feature profile: {value!r}") from exc


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
    schema_version: Literal["1.0"] = "1.0"


class AcquisitionProvenance(Contract):
    """Verified metadata carried from an explicitly authorized acquisition."""

    acquisition_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str | None = None
    artist: str | None = None
    source: str = Field(min_length=1)

    @field_validator("acquisition_id", "url", "source", "title", "artist")
    @classmethod
    def nonblank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value


class SongRecord(Contract):
    song_id: str
    source_path: str
    title: str = ""
    artist: str = ""
    album: str = ""
    track_number: int | None = Field(default=None, ge=1)
    date: str | None = None
    genre: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    codec: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    lyrics: str | None = None
    lyrics_provenance: str | None = None
    language: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    file_size: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None
    integrity_fingerprint: str
    variant_group_id: str | None = None
    acquisition: AcquisitionProvenance | None = None
    preprocessing_state: str = "pending"
    error: str | None = None


class AudioSegment(Contract):
    segment_id: str
    song_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    index: int = Field(ge=0)

    @field_validator("end_seconds")
    @classmethod
    def after_start(cls, v, info):
        if "start_seconds" in info.data and v <= info.data["start_seconds"]:
            raise ValueError("end_seconds must exceed start_seconds")
        return v


class FeatureArtifact(Contract):
    artifact_id: str
    song_id: str
    segment_id: str | None = None
    modality: str
    feature_view: str
    model_repository: str
    model_revision: str
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    code_version: str
    vector_dtype: str
    dimensions: int = Field(gt=0)
    normalized: bool = False
    source_fingerprint: str
    created_at: datetime
    elapsed_seconds: float = Field(ge=0)
    cache_valid: bool = True
    vector: list[float] | None = None


class GuidanceSet(Contract):
    guidance_id: str
    name: str | None = None
    positive_song_ids: list[str] = Field(default_factory=list, max_length=19)
    negative_song_ids: list[str] = Field(default_factory=list, max_length=19)
    must_link: list[list[str]] = Field(default_factory=list, max_length=19)
    cannot_link: list[list[str]] = Field(default_factory=list, max_length=19)

    @model_validator(mode="after")
    def validate_constraints(self) -> "GuidanceSet":
        groups = self.must_link + self.cannot_link
        if any(len(group) != 2 or not all(group) for group in groups):
            raise ValueError("must_link and cannot_link entries must be non-empty song-id pairs")
        mentioned = set(self.positive_song_ids) | set(self.negative_song_ids)
        mentioned.update(song_id for group in groups for song_id in group)
        if len(mentioned) > 19:
            raise ValueError("guidance references at most 19 unique song IDs in total")
        overlap = set(self.positive_song_ids) & set(self.negative_song_ids)
        if overlap:
            raise ValueError(f"song IDs cannot be both positive and negative: {sorted(overlap)}")
        must_pairs = {frozenset(pair) for pair in self.must_link}
        cannot_pairs = {frozenset(pair) for pair in self.cannot_link}
        if must_pairs & cannot_pairs:
            raise ValueError("a pair cannot be both must_link and cannot_link")
        return self

    def validate_known_song_ids(self, known_song_ids: set[str]) -> None:
        mentioned = set(self.positive_song_ids) | set(self.negative_song_ids)
        mentioned.update(
            song_id for group in self.must_link + self.cannot_link for song_id in group
        )
        unknown = sorted(mentioned - known_song_ids)
        if unknown:
            raise ValueError(f"guidance references unknown song IDs: {unknown}")

    def ready_for(self, known_song_ids: set[str]) -> bool:
        try:
            self.validate_known_song_ids(known_song_ids)
        except ValueError:
            return False
        return True


class Membership(Contract):
    song_id: str
    candidate_id: str
    membership_score: float = Field(
        ge=0, le=1, validation_alias=AliasChoices("membership_score", "score")
    )
    threshold: float = Field(ge=0, le=1)
    status: Literal["member", "abstained", "excluded"] = "member"
    reason: str | None = None

    @property
    def score(self) -> float:
        """Compatibility accessor; serialized artifacts use membership_score."""
        return self.membership_score


class PlaylistCandidate(Contract):
    candidate_id: str
    name: str
    granularity: str
    resolution: float | None = Field(default=None, ge=0)
    member_song_ids: list[str] = Field(default_factory=list)
    representative_song_ids: list[str] = Field(default_factory=list)
    core_song_ids: list[str] = Field(default_factory=list)
    boundary_song_ids: list[str] = Field(default_factory=list)
    excluded_song_ids: list[str] = Field(default_factory=list)
    stability: float = Field(ge=0, le=1)
    cohesion: float = Field(ge=0, le=1)
    separation: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    naming_evidence: list[str] = Field(default_factory=list)
    lineage: dict[str, list[str]] = Field(default_factory=dict)
    guidance_ids: list[str] = Field(default_factory=list)


class ReviewAction(Contract):
    action_id: str
    candidate_id: str
    action: Literal["approve", "reject", "rename", "split", "merge", "intrusion", "omission"]
    actor: str
    created_at: datetime
    song_id: str | None = None
    value: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class DiscoveryRun(Contract):
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["running", "completed", "failed"] = "running"
    song_count: int = Field(ge=0)
    candidate_ids: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class RunManifest(Contract):
    run_id: str
    created_at: datetime
    discovery_run: str
    artifacts: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    generator_version: str
    git_revision: str | None = None
