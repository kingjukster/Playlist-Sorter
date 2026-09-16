"""Stability consensus, overlap, abstention, hierarchy, and candidate policy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, exp, floor
from typing import Iterable, Mapping, Sequence

PERTURBATION_COUNT = 12
LOG_RESOLUTIONS = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Community:
    candidate_id: str
    member_song_ids: frozenset[str]
    lens: str = "fused"
    resolution: float = 1.0


@dataclass(frozen=True)
class Perturbation:
    trial: int
    segment_seed: int
    edge_multiplier: float
    cluster_seed: int


@dataclass(frozen=True)
class CandidatePolicy:
    min_stability: float = 0.75
    min_core_recurrence: float = 0.80
    min_margin: float = 0.05
    min_songs: int = 8
    duplicate_jaccard: float = 0.80
    duplicate_centroid_cosine: float = 0.95
    meaningful_parent_child_ratio: float = 1.5
    max_candidates: int = 40
    quotas: Mapping[str, int] = field(
        default_factory=lambda: {"broad": 8, "medium": 16, "narrow": 16}
    )
    rank_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "stability": 0.40,
            "cohesion": 0.25,
            "distinctiveness": 0.20,
            "novelty": 0.15,
        }
    )


CANDIDATE_POLICY = CandidatePolicy()


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    member_song_ids: frozenset[str]
    granularity: str
    centroid: tuple[float, ...]
    stability: float
    core_recurrence: float
    cohesion: float
    distinctiveness: float
    novelty: float
    margin: float
    lens: str = "fused"
    resolution: float = 1.0
    lineage: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def rank(self) -> float:
        weights = CANDIDATE_POLICY.rank_weights
        return (
            weights["stability"] * self.stability
            + weights["cohesion"] * self.cohesion
            + weights["distinctiveness"] * self.distinctiveness
            + weights["novelty"] * self.novelty
        )


def default_perturbations(seed: int = 0) -> tuple[Perturbation, ...]:
    """Return the fixed 12 bounded segment/edge/seed perturbation trials."""
    trials = tuple(
        Perturbation(
            trial=trial,
            segment_seed=seed + segment_variant,
            edge_multiplier=edge_multiplier,
            cluster_seed=seed + trial,
        )
        for trial, (segment_variant, edge_multiplier) in enumerate(
            (segment_variant, edge_multiplier)
            for segment_variant in range(4)
            for edge_multiplier in (0.95, 1.0, 1.05)
        )
    )
    assert len(trials) == PERTURBATION_COUNT
    return trials


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def align_communities(
    reference: Sequence[Community], trials: Sequence[Community]
) -> dict[str, tuple[str, float]]:
    """One-to-one deterministic maximum-Jaccard matching across communities."""
    choices = sorted(
        (
            (
                -jaccard(left.member_song_ids, right.member_song_ids),
                left.candidate_id,
                right.candidate_id,
            )
            for left in reference
            for right in trials
        )
    )
    used_left: set[str] = set()
    used_right: set[str] = set()
    aligned: dict[str, tuple[str, float]] = {}
    for negative_score, left_id, right_id in choices:
        if left_id in used_left or right_id in used_right:
            continue
        used_left.add(left_id)
        used_right.add(right_id)
        aligned[left_id] = (right_id, -negative_score)
    return aligned


def community_consensus(
    reference: Community,
    trial_communities: Sequence[Sequence[Community]],
) -> tuple[float, dict[str, float]]:
    """Compute mean aligned Jaccard and member recurrence for one candidate."""
    recurrence = {song_id: 0 for song_id in reference.member_song_ids}
    aligned_scores: list[float] = []
    for communities in trial_communities:
        aligned = align_communities((reference,), communities)
        match = aligned.get(reference.candidate_id)
        if match is None:
            aligned_scores.append(0.0)
            continue
        matched_id, score = match
        aligned_scores.append(score)
        matched = next(item for item in communities if item.candidate_id == matched_id)
        for song_id in reference.member_song_ids & matched.member_song_ids:
            recurrence[song_id] += 1
    count = len(trial_communities)
    return (
        sum(aligned_scores) / count if count else 0.0,
        {song_id: seen / count if count else 0.0 for song_id, seen in recurrence.items()},
    )


def candidate_size_bounds(
    granularity: str,
    library_size: int,
    *,
    policy: CandidatePolicy = CANDIDATE_POLICY,
) -> tuple[int, int]:
    """Return executable broad/medium/narrow size policy bounds."""
    if library_size < 1:
        raise ValueError("library_size must be positive")
    if granularity == "broad":
        return max(policy.min_songs, ceil(library_size * 0.05)), max(
            policy.min_songs, floor(library_size * 0.25)
        )
    if granularity == "medium":
        return max(policy.min_songs, ceil(library_size * 0.01)), max(
            policy.min_songs, floor(library_size * 0.10)
        )
    if granularity == "narrow":
        return policy.min_songs, max(policy.min_songs, floor(library_size * 0.03))
    raise ValueError("granularity must be broad, medium, or narrow")


def passes_candidate_gates(
    candidate: CandidateEvidence,
    *,
    library_size: int,
    policy: CandidatePolicy = CANDIDATE_POLICY,
) -> bool:
    """Apply frozen size, stability, recurrence, and margin retention gates."""
    lower, upper = candidate_size_bounds(candidate.granularity, library_size, policy=policy)
    return (
        lower <= len(candidate.member_song_ids) <= upper
        and candidate.stability >= policy.min_stability
        and candidate.core_recurrence >= policy.min_core_recurrence
        and candidate.margin >= policy.min_margin
    )


def _centroid_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else -1.0


def _meaningful_parent_child(
    left: CandidateEvidence, right: CandidateEvidence, ratio: float
) -> bool:
    smaller, larger = sorted((left.member_song_ids, right.member_song_ids), key=len)
    return bool(smaller) and smaller <= larger and len(larger) / len(smaller) >= ratio


def _duplicate(left: CandidateEvidence, right: CandidateEvidence, policy: CandidatePolicy) -> bool:
    centroid_matches = (
        left.member_song_ids == right.member_song_ids
        or _centroid_cosine(left.centroid, right.centroid) >= policy.duplicate_centroid_cosine
    )
    return (
        jaccard(left.member_song_ids, right.member_song_ids) >= policy.duplicate_jaccard
        and centroid_matches
        and not _meaningful_parent_child(left, right, policy.meaningful_parent_child_ratio)
    )


def select_candidates(
    candidates: Iterable[CandidateEvidence],
    *,
    library_size: int,
    policy: CandidatePolicy = CANDIDATE_POLICY,
) -> tuple[CandidateEvidence, ...]:
    """Gate, rank, deduplicate, and quota candidate playlists deterministically."""
    eligible = [
        candidate
        for candidate in candidates
        if passes_candidate_gates(candidate, library_size=library_size, policy=policy)
    ]
    ordered = sorted(eligible, key=lambda item: (-item.rank, item.candidate_id))
    selected: list[CandidateEvidence] = []
    quotas = {kind: 0 for kind in policy.quotas}
    for candidate in ordered:
        if len(selected) >= policy.max_candidates:
            break
        if quotas[candidate.granularity] >= policy.quotas[candidate.granularity]:
            continue
        if any(_duplicate(candidate, existing, policy) for existing in selected):
            continue
        selected.append(candidate)
        quotas[candidate.granularity] += 1
    hierarchy = build_hierarchy(selected, ratio=policy.meaningful_parent_child_ratio)
    return tuple(
        replace(candidate, lineage=hierarchy.get(candidate.candidate_id, {}))
        for candidate in selected
    )


def build_hierarchy(
    candidates: Iterable[CandidateEvidence], *, ratio: float = 1.5
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Attach parent/child links for meaningful nested retained candidates."""
    ordered = sorted(candidates, key=lambda item: (len(item.member_song_ids), item.candidate_id))
    parents: dict[str, list[str]] = {item.candidate_id: [] for item in ordered}
    children: dict[str, list[str]] = {item.candidate_id: [] for item in ordered}
    for child in ordered:
        possible = [
            parent
            for parent in ordered
            if len(parent.member_song_ids) >= ratio * len(child.member_song_ids)
            and child.member_song_ids <= parent.member_song_ids
        ]
        if possible:
            parent = min(possible, key=lambda item: (len(item.member_song_ids), item.candidate_id))
            parents[child.candidate_id].append(parent.candidate_id)
            children[parent.candidate_id].append(child.candidate_id)
    return {
        candidate_id: {
            "parents": tuple(sorted(parents[candidate_id])),
            "children": tuple(sorted(children[candidate_id])),
        }
        for candidate_id in parents
    }


def calibrated_membership_score(
    *,
    within_similarity: float,
    outside_similarity: float,
    recurrence: float,
    view_coverage: float,
) -> float:
    """Calibrate an overlap-safe membership score; it is never a probability.

    The score combines an in-vs-out margin with independently measured trial
    recurrence and present-view coverage.  Callers may threshold each candidate
    independently, so a song can belong to multiple playlists.
    """
    values = (within_similarity, outside_similarity, recurrence, view_coverage)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("membership inputs must be bounded between zero and one")
    margin_component = 1.0 / (1.0 + exp(-12.0 * (within_similarity - outside_similarity)))
    return max(0.0, min(1.0, margin_component * recurrence * view_coverage))


def abstention_reason(
    *,
    membership_score: float,
    threshold: float,
    recurrence: float,
    view_coverage: float,
    min_recurrence: float = CANDIDATE_POLICY.min_core_recurrence,
    min_view_coverage: float = 0.50,
) -> str | None:
    """Explain why weak evidence remains deliberately unassigned."""
    if recurrence < min_recurrence:
        return "instability"
    if view_coverage < min_view_coverage:
        return "insufficient_view_coverage"
    if membership_score < threshold:
        return "weak_similarity"
    return None


def membership_decision(
    *,
    within_similarity: float,
    outside_similarity: float,
    recurrence: float,
    view_coverage: float,
    threshold: float = 0.60,
) -> dict[str, float | str | None]:
    """Return explicit overlap membership or an abstention with its reason."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be bounded between zero and one")
    score = calibrated_membership_score(
        within_similarity=within_similarity,
        outside_similarity=outside_similarity,
        recurrence=recurrence,
        view_coverage=view_coverage,
    )
    reason = abstention_reason(
        membership_score=score,
        threshold=threshold,
        recurrence=recurrence,
        view_coverage=view_coverage,
    )
    return {
        "membership_score": score,
        "status": "abstained" if reason else "member",
        "reason": reason,
    }
