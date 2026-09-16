"""Exact, inspectable multi-view similarity graph construction."""

from .knn import (
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_K,
    K_VARIANTS,
    VIEW_NAMES,
    FusedGraph,
    ViewGraph,
    build_multi_lens_graphs,
    build_view_graph,
    calibrate_scores,
    exact_cosine_topk,
    fuse_view_graphs,
    metadata_leakage_diagnostics,
)

__all__ = [
    "DEFAULT_FUSION_WEIGHTS",
    "DEFAULT_K",
    "K_VARIANTS",
    "VIEW_NAMES",
    "FusedGraph",
    "ViewGraph",
    "build_multi_lens_graphs",
    "build_view_graph",
    "calibrate_scores",
    "exact_cosine_topk",
    "fuse_view_graphs",
    "metadata_leakage_diagnostics",
]
