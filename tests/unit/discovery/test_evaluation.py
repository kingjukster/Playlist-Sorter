import pytest

from playlist_sorter.evaluation import (
    BASELINES,
    adjusted_rand_index,
    approximate_search_gate,
    artist_disjoint_split,
    ndcg_at_k,
    outlier_f1,
    pairwise_overlap_f1,
    precision_at_k,
    qualification_gate,
    resource_performance_gate,
    wilson_lower_bound,
)
from playlist_sorter.evaluation.metrics import cache_performance_gate


def test_deterministic_core_metrics():
    assert adjusted_rand_index([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0
    assert pairwise_overlap_f1([["a", "b"], ["b", "c"]], [["a", "b"], ["b", "c"]]) == 1.0
    assert outlier_f1([], []) == 1.0
    assert precision_at_k(["a", "x"], {"a"}, 20) == 0.5
    assert ndcg_at_k(["a", "x"], {"a"}, 20) == 1.0
    assert 0 < wilson_lower_bound(9, 10) < 0.9


def test_artist_disjoint_split_is_deterministic_and_disallows_tuning():
    split = artist_disjoint_split({"s1": "a", "s2": "b", "s3": "b"}, evaluation_fraction=0.5)
    assert not (split.train_artists & split.evaluation_artists)
    assert split.tuning_allowed is False
    with pytest.raises(ValueError, match="tune"):
        type(split)((), (), frozenset(), frozenset(), tuning_allowed=True)


def test_search_and_cache_qualification_gates_are_measured():
    exact = {"q": tuple(range(30))}
    qualified = approximate_search_gate(exact, exact, exact_seconds=10, approximate_seconds=5)
    assert qualified.passed and qualified.recall_at_30 == 1.0 and qualified.speedup_percent == 50.0
    unqualified = approximate_search_gate(
        exact, {"q": tuple(range(29))}, exact_seconds=10, approximate_seconds=6
    )
    assert not unqualified.passed
    assert cache_performance_gate(
        cache_hits=99, cache_requests=100, cold_seconds=10, warm_seconds=2
    ).passed
    assert not cache_performance_gate(
        cache_hits=98, cache_requests=100, cold_seconds=10, warm_seconds=2
    ).passed
    assert not cache_performance_gate(
        cache_hits=99, cache_requests=100, cold_seconds=10, warm_seconds=2.01
    ).passed


def test_baseline_interfaces_include_required_ablation_set_without_claiming_availability():
    identifiers = {baseline.identifier for baseline in BASELINES}
    assert identifiers == {
        "metadata_only",
        "descriptor_kmeans",
        "pooled_muq_mulan_hdbscan",
        "single_lens_leiden",
        "fused_leiden_no_consensus",
        "full_system",
    }
    assert all(isinstance(baseline.available, bool) for baseline in BASELINES)


def test_private_qualification_and_resource_gates_are_exact_and_not_claimed():
    assert resource_performance_gate(peak_vram_percent=89.9, peak_host_ram_percent=84.9).passed
    assert not resource_performance_gate(peak_vram_percent=90, peak_host_ram_percent=80).passed
    result = qualification_gate(
        full_system_macro_ndcg_at_20=0.525,
        strongest_baseline_macro_ndcg_at_20=0.5,
        precision_at_20_regression_count=1,
        blind_preference_successes=60,
        blind_preference_trials=100,
        guided_ndcg_at_20=0.525,
        unguided_ndcg_at_20=0.5,
        guided_stability_drop=0.03,
    )
    assert result.passed
