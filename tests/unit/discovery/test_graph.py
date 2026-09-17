import pytest

torch = pytest.importorskip("torch")

from playlist_sorter.graph import (  # noqa: E402
    ViewGraph,
    build_multi_lens_graphs,
    build_view_graph,
    exact_cosine_topk,
    fuse_view_graphs,
    metadata_leakage_diagnostics,
)


def test_exact_blockwise_top_k_is_deterministic_and_excludes_self():
    vectors = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    first_indices, first_scores = exact_cosine_topk(vectors, k=2, block_size=2, device="cpu")
    second_indices, second_scores = exact_cosine_topk(vectors, k=2, block_size=3, device="cpu")
    assert torch.equal(first_indices, second_indices)
    assert torch.equal(first_scores, second_scores)
    assert all(row not in neighbors.tolist() for row, neighbors in enumerate(first_indices))
    assert first_indices[0, 0].item() == 1
    assert first_indices[2, 0].item() == 3


def test_views_fuse_only_present_modalities_and_preserve_provenance():
    ids = ("a", "b", "c")
    semantic = ViewGraph("semantic_audio", ids, {(0, 1): 0.9, (1, 0): 0.9}, 30, "cpu")
    lyrics = ViewGraph("lyrics", ids, {(0, 1): 0.8, (1, 0): 0.8}, 30, "cpu")
    fused = fuse_view_graphs((semantic, lyrics))
    assert set(fused.contributions[(0, 1)]) == {"semantic_audio", "lyrics"}
    assert fused.available_weight[(0, 1)] == pytest.approx(0.55)
    assert fused.edges[(0, 1)] == pytest.approx(sum(fused.contributions[(0, 1)].values()))


def test_missing_vectors_are_not_silently_converted_to_evidence():
    lenses = build_multi_lens_graphs(
        ("a", "b"),
        {"semantic_audio": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "lyrics": None},
        device="cpu",
    )
    assert set(lenses) == {"semantic_audio", "fused"}
    with pytest.raises(ValueError, match="two-dimensional"):
        build_view_graph("lyrics", ["a"], torch.tensor([0.0, 1.0]))


def test_song_level_missing_views_are_renormalized_per_comparison():
    lenses = build_multi_lens_graphs(
        ("a", "b", "c"),
        {
            "semantic_audio": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            "lyrics": [[1.0, 0.0], None, [0.0, 1.0]],
        },
        k=2,
        device="cpu",
    )
    fused = lenses["fused"]
    assert (0, 1) in fused.edges
    assert fused.available_weight[(0, 1)] == pytest.approx(0.35)
    assert (0, 2) in fused.edges
    assert fused.available_weight[(0, 2)] == pytest.approx(0.55)


def test_metadata_is_diagnostic_only():
    diagnostics = metadata_leakage_diagnostics(
        {(0, 1): 0.8, (1, 2): 0.7},
        artists=("one", "one", "two"),
        albums=("album", "album", "other"),
    )
    assert diagnostics["same_artist_rate"] == pytest.approx(0.5)
    assert diagnostics["same_album_rate"] == pytest.approx(0.5)
