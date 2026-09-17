"""Dependency-free, deterministic metrics and qualification gates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import comb, log2
from typing import AbstractSet, Hashable, Iterable, Mapping, Sequence


def _pairs(
    groups: Mapping[Hashable, Iterable[Hashable]] | Iterable[Iterable[Hashable]],
) -> set[frozenset[Hashable]]:
    collections = groups.values() if isinstance(groups, Mapping) else groups
    return {
        frozenset(pair)
        for group in collections
        for pair in combinations(sorted(set(group), key=repr), 2)
    }


def _f1(predicted: AbstractSet[Hashable], expected: AbstractSet[Hashable]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    return 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)


def pairwise_overlap_f1(
    predicted_groups: Mapping[Hashable, Iterable[Hashable]] | Iterable[Iterable[Hashable]],
    expected_groups: Mapping[Hashable, Iterable[Hashable]] | Iterable[Iterable[Hashable]],
) -> float:
    """F1 over same-playlist pairs; overlapping groups are represented exactly."""
    return _f1(_pairs(predicted_groups), _pairs(expected_groups))


def outlier_f1(
    predicted_outliers: Iterable[Hashable], expected_outliers: Iterable[Hashable]
) -> float:
    """F1 for abstained/outlier songs, with empty-vs-empty treated as perfect."""
    return _f1(set(predicted_outliers), set(expected_outliers))


def adjusted_rand_index(predicted: Sequence[Hashable], expected: Sequence[Hashable]) -> float:
    """Adjusted Rand index without a scikit-learn dependency."""
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected labels must have equal length")
    count = len(predicted)
    if count < 2:
        return 1.0
    contingency: dict[tuple[Hashable, Hashable], int] = defaultdict(int)
    predicted_sizes: dict[Hashable, int] = defaultdict(int)
    expected_sizes: dict[Hashable, int] = defaultdict(int)
    for actual, proposed in zip(expected, predicted, strict=True):
        contingency[(actual, proposed)] += 1
        expected_sizes[actual] += 1
        predicted_sizes[proposed] += 1
    index = sum(comb(value, 2) for value in contingency.values())
    expected_index = sum(comb(value, 2) for value in expected_sizes.values())
    predicted_index = sum(comb(value, 2) for value in predicted_sizes.values())
    total_pairs = comb(count, 2)
    chance = expected_index * predicted_index / total_pairs
    maximum = (expected_index + predicted_index) / 2
    return 1.0 if maximum == chance else (index - chance) / (maximum - chance)


def _relevance(
    ranked: Sequence[Hashable], relevant: Mapping[Hashable, float] | Iterable[Hashable]
) -> dict[Hashable, float]:
    if isinstance(relevant, Mapping):
        return {item: max(0.0, float(value)) for item, value in relevant.items()}
    return {item: 1.0 for item in relevant}


def precision_at_k(
    ranked: Sequence[Hashable], relevant: Mapping[Hashable, float] | Iterable[Hashable], k: int = 20
) -> float:
    """Precision@k; the denominator is min(k, ranked length), never padded."""
    if k <= 0:
        raise ValueError("k must be positive")
    selected = ranked[:k]
    if not selected:
        return 0.0
    labels = _relevance(ranked, relevant)
    return sum(labels.get(item, 0.0) > 0 for item in selected) / len(selected)


def ndcg_at_k(
    ranked: Sequence[Hashable], relevant: Mapping[Hashable, float] | Iterable[Hashable], k: int = 20
) -> float:
    """NDCG@k supporting binary or graded relevance."""
    if k <= 0:
        raise ValueError("k must be positive")
    labels = _relevance(ranked, relevant)
    observed = sum(
        (2 ** labels.get(item, 0.0) - 1) / log2(index + 2) for index, item in enumerate(ranked[:k])
    )
    ideal_values = sorted(labels.values(), reverse=True)[:k]
    ideal = sum((2**value - 1) / log2(index + 2) for index, value in enumerate(ideal_values))
    return 0.0 if ideal == 0 else observed / ideal


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Conservative binomial lower confidence bound for blinded preferences."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if trials == 0:
        return 0.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    spread = z * ((proportion * (1 - proportion) + z * z / (4 * trials)) / trials) ** 0.5
    return (centre - spread) / denominator


@dataclass(frozen=True)
class ArtistDisjointSplit:
    train_song_ids: tuple[str, ...]
    evaluation_song_ids: tuple[str, ...]
    train_artists: frozenset[str]
    evaluation_artists: frozenset[str]
    tuning_allowed: bool = False

    def __post_init__(self) -> None:
        if self.train_artists & self.evaluation_artists:
            raise ValueError("artist-disjoint evaluation cannot share artists")
        if self.tuning_allowed:
            raise ValueError("private evaluation labels must not tune the system")


def artist_disjoint_split(
    song_artists: Mapping[str, str], *, evaluation_fraction: float = 0.2
) -> ArtistDisjointSplit:
    """Deterministically hold out complete artists without exposing labels to tuning."""
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between zero and one")
    artists = sorted(set(song_artists.values()))
    evaluation_count = max(1, round(len(artists) * evaluation_fraction)) if artists else 0
    evaluation_artists = frozenset(artists[-evaluation_count:]) if evaluation_count else frozenset()
    evaluation = tuple(
        sorted(song for song, artist in song_artists.items() if artist in evaluation_artists)
    )
    train = tuple(
        sorted(song for song, artist in song_artists.items() if artist not in evaluation_artists)
    )
    return ArtistDisjointSplit(
        train, evaluation, frozenset(artists) - evaluation_artists, evaluation_artists
    )


@dataclass(frozen=True)
class ApproximateSearchGate:
    recall_at_30: float
    speedup_percent: float
    passed: bool


def approximate_search_gate(
    exact_neighbors: Mapping[Hashable, Sequence[Hashable]],
    approximate_neighbors: Mapping[Hashable, Sequence[Hashable]],
    *,
    exact_seconds: float,
    approximate_seconds: float,
) -> ApproximateSearchGate:
    """Require recall@30 >= .98 and at least 50% lower wall time."""
    if exact_seconds <= 0 or approximate_seconds <= 0:
        raise ValueError("search timings must be positive")
    queries = sorted(set(exact_neighbors) | set(approximate_neighbors), key=repr)
    recalls = []
    for query in queries:
        expected = set(exact_neighbors.get(query, ())[:30])
        observed = set(approximate_neighbors.get(query, ())[:30])
        recalls.append(1.0 if not expected else len(expected & observed) / len(expected))
    recall = sum(recalls) / len(recalls) if recalls else 1.0
    speedup = 100 * (exact_seconds - approximate_seconds) / exact_seconds
    return ApproximateSearchGate(recall, speedup, recall >= 0.98 and speedup >= 50.0)


@dataclass(frozen=True)
class CachePerformanceGate:
    cache_hit_rate: float
    cold_seconds: float
    warm_seconds: float
    passed: bool


def cache_performance_gate(
    *,
    cache_hits: int,
    cache_requests: int,
    cold_seconds: float,
    warm_seconds: float,
    min_cache_hit_rate: float = 0.99,
    min_warm_speedup_factor: float = 5.0,
) -> CachePerformanceGate:
    """Require 99% reuse and a warm rerun at least five times faster by default."""
    if cache_hits < 0 or cache_requests < 0 or cache_hits > cache_requests:
        raise ValueError("cache hits must be between zero and cache requests")
    if cold_seconds <= 0 or warm_seconds <= 0:
        raise ValueError("cache timings must be positive")
    rate = cache_hits / cache_requests if cache_requests else 0.0
    return CachePerformanceGate(
        rate,
        cold_seconds,
        warm_seconds,
        rate >= min_cache_hit_rate and cold_seconds / warm_seconds >= min_warm_speedup_factor,
    )


@dataclass(frozen=True)
class ResourceGate:
    peak_vram_percent: float
    peak_host_ram_percent: float
    passed: bool


def resource_performance_gate(
    *, peak_vram_percent: float, peak_host_ram_percent: float
) -> ResourceGate:
    """Require measured peak VRAM below 90% and host RAM below 85%."""
    if not 0 <= peak_vram_percent <= 100 or not 0 <= peak_host_ram_percent <= 100:
        raise ValueError("resource percentages must be in [0, 100]")
    return ResourceGate(
        peak_vram_percent,
        peak_host_ram_percent,
        peak_vram_percent < 90 and peak_host_ram_percent < 85,
    )


@dataclass(frozen=True)
class QualificationGate:
    full_system_ndcg_passed: bool
    precision_regression_passed: bool
    blind_preference_passed: bool
    guided_ndcg_passed: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.full_system_ndcg_passed,
                self.precision_regression_passed,
                self.blind_preference_passed,
                self.guided_ndcg_passed,
            )
        )


def qualification_gate(
    *,
    full_system_macro_ndcg_at_20: float,
    strongest_baseline_macro_ndcg_at_20: float,
    precision_at_20_regression_count: int,
    blind_preference_successes: int,
    blind_preference_trials: int,
    guided_ndcg_at_20: float,
    unguided_ndcg_at_20: float,
    guided_stability_drop: float,
) -> QualificationGate:
    """Encode frozen private/full-run gates without claiming they were executed."""
    values = (
        full_system_macro_ndcg_at_20,
        strongest_baseline_macro_ndcg_at_20,
        guided_ndcg_at_20,
        unguided_ndcg_at_20,
    )
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("NDCG values must be in [0, 1]")
    if precision_at_20_regression_count < 0 or guided_stability_drop < 0:
        raise ValueError("regressions and stability drop must be non-negative")
    full_target = strongest_baseline_macro_ndcg_at_20 * 1.05
    guided_target = unguided_ndcg_at_20 * 1.05
    blind_rate = (
        blind_preference_successes / blind_preference_trials if blind_preference_trials else 0.0
    )
    blind_passed = (
        blind_preference_trials >= 100
        and blind_rate >= 0.60
        and wilson_lower_bound(blind_preference_successes, blind_preference_trials) > 0.50
    )
    return QualificationGate(
        full_system_macro_ndcg_at_20 >= full_target,
        precision_at_20_regression_count <= 1,
        blind_passed,
        guided_ndcg_at_20 >= guided_target and guided_stability_drop <= 0.03,
    )
