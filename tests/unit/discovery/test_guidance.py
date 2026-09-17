import pytest

from playlist_sorter.core.contracts import GuidanceSet
from playlist_sorter.guidance import (
    GuidanceError,
    apply_guidance,
    build_prototypes,
    validate_guidance,
)


def guidance(**changes):
    return GuidanceSet(guidance_id="g", **changes)


def test_named_concept_requires_two_positive_seeds():
    with pytest.raises(GuidanceError, match="two positive"):
        validate_guidance(guidance(name="late night", positive_song_ids=["one"]))


def test_duplicate_unknown_contradictory_and_twentieth_examples_fail_closed():
    with pytest.raises(GuidanceError, match="duplicate"):
        validate_guidance(guidance(positive_song_ids=["one", "one"]))
    with pytest.raises(GuidanceError, match="unknown"):
        validate_guidance(guidance(positive_song_ids=["one"]), {"two"})
    with pytest.raises(ValueError, match="both positive and negative"):
        guidance(positive_song_ids=["one"], negative_song_ids=["one"])
    with pytest.raises(ValueError, match="at most 19"):
        guidance(positive_song_ids=[str(index) for index in range(20)])


def test_duplicate_and_contradictory_constraints_fail_closed():
    with pytest.raises(GuidanceError, match="duplicate"):
        validate_guidance(guidance(must_link=[["a", "b"], ["b", "a"]]))
    with pytest.raises(ValueError, match="both must_link and cannot_link"):
        guidance(must_link=[["a", "b"]], cannot_link=[["a", "b"]])


def test_prototypes_and_graph_adjustments_preserve_unguided_output():
    item = guidance(
        positive_song_ids=["a", "b"],
        negative_song_ids=["c"],
        must_link=[["a", "b"]],
        cannot_link=[["a", "c"]],
    )
    positive, negative = build_prototypes(item, {"a": [1.0, 0.0], "b": [0.8, 0.2], "c": [0.0, 1.0]})
    assert positive and negative and positive.uncertainty > 0
    result = apply_guidance(
        item,
        {("a", "b"): 0.5, ("a", "c"): 0.9, ("b", "c"): 0.4},
        guidance_edges={("b", "c"): 0.8},
    )
    assert result.comparison.unguided_edges[("a", "c")] == 0.9
    assert ("a", "c") not in result.comparison.guided_edges
    assert result.comparison.guided_edges[("a", "b")] >= 0.95
    assert result.comparison.guided_edges[("b", "c")] == pytest.approx(0.5)
