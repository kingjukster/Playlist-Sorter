from playlist_sorter.evaluation.metrics import approximate_search_gate, cache_performance_gate


def test_approximate_search_threshold_is_exactly_qualifying():
    expected = {"q": tuple(range(50))}
    result = approximate_search_gate(
        expected, {"q": tuple(range(30))}, exact_seconds=20, approximate_seconds=10
    )
    assert result.recall_at_30 == 1.0
    assert result.passed


def test_cache_gate_accepts_only_exact_reuse_and_five_x_boundaries():
    assert cache_performance_gate(
        cache_hits=99, cache_requests=100, cold_seconds=50, warm_seconds=10
    ).passed
    assert not cache_performance_gate(
        cache_hits=100, cache_requests=100, cold_seconds=50, warm_seconds=10.01
    ).passed
