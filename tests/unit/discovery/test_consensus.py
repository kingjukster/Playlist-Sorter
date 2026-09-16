from playlist_sorter.discovery.consensus import (
    CandidateEvidence,
    Community,
    align_communities,
    candidate_size_bounds,
    calibrated_membership_score,
    community_consensus,
    default_perturbations,
    passes_candidate_gates,
    select_candidates,
)


def _candidate(candidate_id, members, granularity="medium", rank=0.9):
    return CandidateEvidence(
        candidate_id=candidate_id,
        member_song_ids=frozenset(members),
        granularity=granularity,
        centroid=(1.0, 0.0),
        stability=rank,
        core_recurrence=0.9,
        cohesion=rank,
        distinctiveness=rank,
        novelty=rank,
        margin=0.1,
    )


def test_twelve_bounded_perturbations_cover_edge_and_segment_variants():
    perturbations = default_perturbations(100)
    assert len(perturbations) == 12
    assert {item.edge_multiplier for item in perturbations} == {0.95, 1.0, 1.05}
    assert len({item.segment_seed for item in perturbations}) == 4
    assert len({item.cluster_seed for item in perturbations}) == 12


def test_maximum_jaccard_alignment_and_recurrence_are_deterministic():
    reference = Community("r", frozenset({"a", "b", "c"}))
    trial = Community("t", frozenset({"a", "b", "c"}))
    alignment = align_communities((reference,), (trial,))
    assert alignment == {"r": ("t", 1.0)}
    stability, recurrence = community_consensus(reference, ((trial,), (trial,)))
    assert stability == 1.0
    assert recurrence == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_candidate_gates_size_classes_dedup_ranking_and_quota_policy():
    broad = _candidate("broad", range(10), "broad", 0.99)
    medium = _candidate("medium", range(20, 28), "medium", 0.90)
    duplicate = _candidate("duplicate", range(20, 28), "medium", 0.80)
    narrow = _candidate("narrow", range(40, 48), "narrow", 0.85)
    assert candidate_size_bounds("broad", 100) == (8, 25)
    assert candidate_size_bounds("medium", 100) == (8, 10)
    assert candidate_size_bounds("narrow", 100) == (8, 8)
    assert passes_candidate_gates(medium, library_size=100)
    selected = select_candidates((duplicate, narrow, medium, broad), library_size=100)
    assert [item.candidate_id for item in selected] == ["broad", "medium", "narrow"]


def test_calibrated_membership_is_overlap_safe_and_abstains_when_weak():
    strong = calibrated_membership_score(
        within_similarity=0.9, outside_similarity=0.3, recurrence=1.0, view_coverage=1.0
    )
    weak = calibrated_membership_score(
        within_similarity=0.4, outside_similarity=0.5, recurrence=0.5, view_coverage=0.4
    )
    assert 0.9 < strong <= 1.0
    assert 0.0 <= weak < 0.5


def test_exact_cross_lens_membership_duplicates_ignore_centroid_dimensions():
    left = _candidate("left", range(8), "broad", 0.99)
    right = CandidateEvidence(
        **{
            **left.__dict__,
            "candidate_id": "right",
            "centroid": (1.0, 0.0, 0.0),
            "lens": "lyrics",
        }
    )
    selected = select_candidates((right, left), library_size=20)
    assert len(selected) == 1
