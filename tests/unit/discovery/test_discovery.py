from collections import Counter
from math import comb

from playlist_sorter.discovery.backends import SeededLeidenBackend


def _adjusted_rand_index(labels_a, labels_b):
    pairs = Counter(zip(labels_a, labels_b, strict=True))
    a_counts = Counter(labels_a)
    b_counts = Counter(labels_b)
    total = len(labels_a)
    numerator = sum(comb(value, 2) for value in pairs.values())
    expected = sum(comb(value, 2) for value in a_counts.values()) * sum(
        comb(value, 2) for value in b_counts.values()
    ) / comb(total, 2)
    upper = (sum(comb(value, 2) for value in a_counts.values()) + sum(
        comb(value, 2) for value in b_counts.values()
    )) / 2
    return (numerator - expected) / (upper - expected) if upper != expected else 1.0


def _pairwise_f1(predicted, expected):
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def test_seeded_backend_recovers_planted_communities_and_is_repeatable():
    ids = tuple(f"s{i}" for i in range(10))
    edges = {
        (left, right): 0.95
        for group in (range(0, 5), range(5, 10))
        for left in group
        for right in group
        if left != right
    }
    backend = SeededLeidenBackend(min_edge_weight=0.7)
    first = backend.cluster(ids, edges, resolution=1.0, seed=7)
    second = backend.cluster(ids, edges, resolution=1.0, seed=7)
    planted = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
    assert _adjusted_rand_index(first.labels, planted) >= 0.85
    assert _adjusted_rand_index(first.labels, second.labels) >= 0.99
    assert first.backend in {"leiden", "deterministic-components"}


def test_synthetic_overlap_and_outlier_f1_gates_are_executable():
    expected_overlap = {("a", "rock"), ("a", "workout"), ("b", "rock"), ("c", "workout")}
    predicted_overlap = set(expected_overlap)
    expected_outliers = {"z"}
    predicted_outliers = {"z"}
    assert _pairwise_f1(predicted_overlap, expected_overlap) >= 0.90
    assert _pairwise_f1(predicted_outliers, expected_outliers) >= 0.80
