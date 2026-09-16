"""Fail-closed validation and deterministic, post-discovery guidance.

This module deliberately adjusts an existing graph instead of replacing its
unguided result.  It is safe to use on private song identifiers only; no title,
artist, lyric, or embedding data is persisted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Hashable, Iterable, Mapping, Sequence, TypeVar

from playlist_sorter.core.contracts import GuidanceSet


class GuidanceError(ValueError):
    """A guidance set is invalid or unsafe to apply."""


@dataclass(frozen=True)
class GuidancePolicy:
    """Fixed MVP policy; limits are intentionally not user-tunable."""

    max_unique_songs: int = 19
    named_concept_minimum_positive_seeds: int = 2
    guided_weight: float = 0.25
    unguided_weight: float = 0.75
    must_link_floor: float = 0.95

    def __post_init__(self) -> None:
        if self.max_unique_songs != 19:
            raise GuidanceError("the MVP guidance limit is exactly 19 unique songs")
        if self.named_concept_minimum_positive_seeds < 2:
            raise GuidanceError("named concepts require at least two positive seeds")
        if abs(self.guided_weight + self.unguided_weight - 1.0) > 1e-12:
            raise GuidanceError("guided and unguided weights must sum to one")
        if not 0 <= self.must_link_floor <= 1:
            raise GuidanceError("must-link floor must be in [0, 1]")


@dataclass(frozen=True)
class Prototype:
    """Centroid and uncertainty for one seed polarity."""

    song_ids: tuple[str, ...]
    centroid: tuple[float, ...]
    uncertainty: float


@dataclass(frozen=True)
class UnguidedComparison:
    """Preserves original results so reports can explain every adjustment."""

    unguided_edges: dict[tuple[str, str], float]
    guided_edges: dict[tuple[str, str], float]
    removed_cannot_links: tuple[tuple[str, str], ...]
    enforced_must_links: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GuidanceResult:
    positive: Prototype | None
    negative: Prototype | None
    comparison: UnguidedComparison


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if not left or not right or left == right:
        raise GuidanceError("constraints must reference two distinct, non-empty song IDs")
    return (left, right) if left < right else (right, left)


def _duplicates(values: Iterable[T]) -> set[T]:
    seen: set[T] = set()
    repeated: set[T] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def validate_guidance(
    guidance: GuidanceSet,
    known_song_ids: set[str] | None = None,
    *,
    policy: GuidancePolicy = GuidancePolicy(),
) -> None:
    """Raise ``GuidanceError`` unless a complete guidance set is safe to use."""
    positive = tuple(guidance.positive_song_ids)
    negative = tuple(guidance.negative_song_ids)
    if duplicates := (_duplicates(positive) | _duplicates(negative)):
        raise GuidanceError(f"duplicate guidance examples are not allowed: {sorted(duplicates)}")
    try:
        must_pairs = tuple(_canonical_pair(*pair) for pair in guidance.must_link)
        cannot_pairs = tuple(_canonical_pair(*pair) for pair in guidance.cannot_link)
    except (TypeError, ValueError) as exc:
        raise GuidanceError("constraints must be pairs of distinct song IDs") from exc
    if _duplicates(must_pairs) or _duplicates(cannot_pairs):
        raise GuidanceError("duplicate guidance constraints are not allowed")
    mentioned = set(positive) | set(negative)
    mentioned.update(song for pair in must_pairs + cannot_pairs for song in pair)
    if len(mentioned) > policy.max_unique_songs:
        raise GuidanceError("guidance references more than 19 unique song IDs")
    if overlap := set(positive) & set(negative):
        raise GuidanceError(f"positive and negative examples conflict: {sorted(overlap)}")
    if conflict := set(must_pairs) & set(cannot_pairs):
        raise GuidanceError(f"must-link and cannot-link constraints conflict: {sorted(conflict)}")
    if guidance.name and len(positive) < policy.named_concept_minimum_positive_seeds:
        raise GuidanceError("a named concept requires at least two positive seeds")
    if known_song_ids is not None:
        unknown = sorted(mentioned - known_song_ids)
        if unknown:
            raise GuidanceError(f"guidance references unknown song IDs: {unknown}")


def _mean(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise GuidanceError("a prototype needs at least one seed vector")
    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise GuidanceError("prototype vectors must have one non-zero shared dimension")
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _prototype(song_ids: Sequence[str], vectors: Mapping[str, Sequence[float]]) -> Prototype | None:
    if not song_ids:
        return None
    missing = sorted(set(song_ids) - set(vectors))
    if missing:
        raise GuidanceError(f"seed vectors missing for song IDs: {missing}")
    seed_vectors = tuple(vectors[song_id] for song_id in song_ids)
    centroid = _mean(seed_vectors)
    uncertainty = sum(_distance(vector, centroid) for vector in seed_vectors) / len(seed_vectors)
    return Prototype(tuple(song_ids), centroid, uncertainty)


def build_prototypes(
    guidance: GuidanceSet,
    vectors: Mapping[str, Sequence[float]],
    *,
    known_song_ids: set[str] | None = None,
    policy: GuidancePolicy = GuidancePolicy(),
) -> tuple[Prototype | None, Prototype | None]:
    """Build positive and negative prototypes after validation, without learning."""
    validate_guidance(guidance, known_song_ids, policy=policy)
    return _prototype(guidance.positive_song_ids, vectors), _prototype(
        guidance.negative_song_ids, vectors
    )


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def apply_guidance(
    guidance: GuidanceSet,
    unguided_edges: Mapping[tuple[str, str], float],
    *,
    known_song_ids: set[str] | None = None,
    policy: GuidancePolicy = GuidancePolicy(),
    vectors: Mapping[str, Sequence[float]] | None = None,
    guidance_edges: Mapping[tuple[str, str], float] | None = None,
) -> GuidanceResult:
    """Mix graph evidence 75/25, force must-links, and remove cannot-links.

    The returned comparison retains an independent copy of the unguided graph.
    Prototype construction is optional because graph-only guidance is valid.
    """
    validate_guidance(guidance, known_song_ids, policy=policy)
    original: dict[tuple[str, str], float] = {}
    for raw_pair, score in unguided_edges.items():
        try:
            pair = _canonical_pair(*raw_pair)
        except (TypeError, ValueError) as exc:
            raise GuidanceError("unguided graph contains an invalid edge") from exc
        if pair in original:
            raise GuidanceError(f"unguided graph has duplicate undirected edge: {pair}")
        original[pair] = _clamp(score)
    guided = dict(original)
    if guidance_edges is not None:
        lens: dict[tuple[str, str], float] = {}
        for raw_pair, score in guidance_edges.items():
            pair = _canonical_pair(*raw_pair)
            if pair in lens:
                raise GuidanceError(f"guidance lens has duplicate undirected edge: {pair}")
            lens[pair] = _clamp(score)
        guided = {
            pair: _clamp(
                policy.unguided_weight * original.get(pair, 0.0)
                + policy.guided_weight * lens.get(pair, 0.0)
            )
            for pair in sorted(set(original) | set(lens))
        }
    must_pairs = tuple(_canonical_pair(*pair) for pair in guidance.must_link)
    cannot_pairs = tuple(_canonical_pair(*pair) for pair in guidance.cannot_link)
    for pair in cannot_pairs:
        guided.pop(pair, None)
    for pair in must_pairs:
        unguided = original.get(pair, 0.0)
        guided[pair] = max(
            policy.must_link_floor, _clamp(policy.unguided_weight * unguided + policy.guided_weight)
        )
    positive, negative = (None, None)
    if vectors is not None:
        positive, negative = build_prototypes(
            guidance, vectors, known_song_ids=known_song_ids, policy=policy
        )
    comparison = UnguidedComparison(
        unguided_edges=original,
        guided_edges=guided,
        removed_cannot_links=cannot_pairs,
        enforced_must_links=must_pairs,
    )
    return GuidanceResult(positive=positive, negative=negative, comparison=comparison)


T = TypeVar("T", bound=Hashable)
