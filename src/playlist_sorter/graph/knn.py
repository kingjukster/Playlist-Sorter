"""Deterministic exact cosine k-NN graphs with late-fusion provenance.

Artist and album metadata deliberately never enter these similarity functions.
They can only be supplied to :func:`metadata_leakage_diagnostics` after graph
construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
from typing import Iterable, Mapping, Sequence

VIEW_NAMES = ("semantic_audio", "acoustic", "lyrics", "descriptors", "fused")
DEFAULT_K = 30
K_VARIANTS = (15, 30, 60)
DEFAULT_FUSION_WEIGHTS = {
    "semantic_audio": 0.35,
    "acoustic": 0.30,
    "lyrics": 0.20,
    "descriptors": 0.15,
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError(
            "exact graph construction requires an installed PyTorch runtime"
        ) from exc
    return torch


def _device(device: str | None):
    torch = _torch()
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return selected


@dataclass(frozen=True)
class ViewGraph:
    """One lens's directed top-k edges, keyed by stable song-position pairs."""

    view: str
    song_ids: tuple[str, ...]
    edges: dict[tuple[int, int], float]
    k: int
    device: str


@dataclass(frozen=True)
class FusedGraph:
    """Late-fused graph where every retained edge retains view contributions."""

    song_ids: tuple[str, ...]
    edges: dict[tuple[int, int], float]
    contributions: dict[tuple[int, int], dict[str, float]]
    available_weight: dict[tuple[int, int], float]


def build_multi_lens_graphs(
    song_ids: Sequence[str],
    vectors_by_view: Mapping[str, object | None],
    *,
    k: int = DEFAULT_K,
    block_size: int = 1_024,
    device: str | None = None,
    weights: Mapping[str, float] = DEFAULT_FUSION_WEIGHTS,
) -> dict[str, ViewGraph | FusedGraph]:
    """Build semantic, acoustic, lyric, descriptor, and fused graph lenses.

    A ``None`` modality is omitted rather than represented by a zero vector;
    fusion subsequently renormalizes the remaining edge-local evidence.
    """
    unknown = set(vectors_by_view) - set(DEFAULT_FUSION_WEIGHTS)
    if unknown:
        raise ValueError(f"unsupported views: {sorted(unknown)}")
    views = {
        name: build_view_graph(
            name, song_ids, vectors, k=k, block_size=block_size, device=device
        )
        for name, vectors in vectors_by_view.items()
        if vectors is not None
    }
    if not views:
        raise ValueError("at least one non-missing modality is required")
    return {**views, "fused": fuse_view_graphs(views.values(), weights=weights)}


def exact_cosine_topk(
    vectors,
    *,
    k: int = DEFAULT_K,
    block_size: int = 1_024,
    device: str | None = None,
):
    """Return exact cosine neighbors using bounded Torch matrix blocks.

    Results are ordered by descending score and then original column index,
    making equal-score ties repeatable.  CUDA is selected when available;
    otherwise Torch executes the same exact calculation on CPU.
    """
    torch = _torch()
    if k < 1:
        raise ValueError("k must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    values = torch.as_tensor(vectors, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    count = int(values.shape[0])
    if count < 2:
        empty_i = torch.empty((count, 0), dtype=torch.long)
        empty_s = torch.empty((count, 0), dtype=torch.float32)
        return empty_i, empty_s
    effective_k = min(k, count - 1)
    target = _device(device)
    normalized = torch.nn.functional.normalize(values.to(target), p=2, dim=1, eps=1e-12)
    all_indices: list[object] = []
    all_scores: list[object] = []
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        scores = normalized[start:stop] @ normalized.T
        rows = torch.arange(start, stop, device=target)
        scores[torch.arange(stop - start, device=target), rows] = -torch.inf
        # Stable argsort preserves ascending column order for exact ties.
        ordered = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :effective_k]
        selected = scores.gather(1, ordered)
        all_indices.append(ordered.cpu())
        all_scores.append(selected.cpu())
    return torch.cat(all_indices), torch.cat(all_scores)


def build_view_graph(
    view: str,
    song_ids: Sequence[str],
    vectors,
    *,
    k: int = DEFAULT_K,
    block_size: int = 1_024,
    device: str | None = None,
) -> ViewGraph:
    """Build a lens graph, rejecting missing vectors instead of inventing zeros."""
    if view not in VIEW_NAMES[:-1]:
        raise ValueError(f"unsupported view: {view}")
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("song_ids must be unique and preserve caller ordering")
    indices, scores = exact_cosine_topk(vectors, k=k, block_size=block_size, device=device)
    if int(indices.shape[0]) != len(song_ids):
        raise ValueError("vectors and song_ids must have the same row count")
    edges = {
        (source, int(neighbor)): float(scores[source, offset])
        for source in range(len(song_ids))
        for offset, neighbor in enumerate(indices[source])
    }
    selected = _device(device)
    return ViewGraph(view, tuple(song_ids), edges, min(k, max(0, len(song_ids) - 1)), str(selected))


def calibrate_scores(scores: Mapping[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Calibrate one view independently using its median and robust IQR.

    The sigmoid output is a comparable *edge score*, not a probability.  Empty
    graphs stay empty so missing modalities have no fabricated evidence.
    """
    if not scores:
        return {}
    ordered = sorted(float(value) for value in scores.values())
    median = ordered[len(ordered) // 2]
    q1 = ordered[(len(ordered) - 1) // 4]
    q3 = ordered[(3 * (len(ordered) - 1)) // 4]
    scale = max(q3 - q1, 1e-6)
    return {edge: 1.0 / (1.0 + exp(-(value - median) / scale)) for edge, value in scores.items()}


def fuse_view_graphs(
    graphs: Iterable[ViewGraph],
    *,
    weights: Mapping[str, float] = DEFAULT_FUSION_WEIGHTS,
) -> FusedGraph:
    """Late-fuse independent graphs and renormalize only present view weights."""
    by_view = {graph.view: graph for graph in graphs}
    if not by_view:
        raise ValueError("at least one view graph is required")
    if set(by_view) - set(DEFAULT_FUSION_WEIGHTS):
        raise ValueError("only base modality graphs may be fused")
    first = next(iter(by_view.values()))
    if any(graph.song_ids != first.song_ids for graph in by_view.values()):
        raise ValueError("all graphs must use identical ordered song_ids")
    unknown_weights = set(weights) - set(DEFAULT_FUSION_WEIGHTS)
    if unknown_weights or any(weight < 0 for weight in weights.values()):
        raise ValueError("fusion weights must be non-negative known views")
    calibrated = {view: calibrate_scores(graph.edges) for view, graph in by_view.items()}
    edge_views: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for view, edges in calibrated.items():
        for edge, score in edges.items():
            edge_views[edge][view] = score
    fused: dict[tuple[int, int], float] = {}
    contributions: dict[tuple[int, int], dict[str, float]] = {}
    availability: dict[tuple[int, int], float] = {}
    for edge in sorted(edge_views):
        present = edge_views[edge]
        denominator = sum(float(weights.get(view, 0.0)) for view in present)
        if denominator <= 0:
            continue
        parts = {
            view: float(weights[view]) * score / denominator for view, score in present.items()
        }
        contributions[edge] = parts
        availability[edge] = denominator
        fused[edge] = sum(parts.values())
    return FusedGraph(first.song_ids, fused, contributions, availability)


def metadata_leakage_diagnostics(
    edges: Mapping[tuple[int, int], float],
    *,
    artists: Sequence[str | None],
    albums: Sequence[str | None],
) -> dict[str, float | int]:
    """Report artist/album concentration without using metadata as features."""
    if len(artists) != len(albums):
        raise ValueError("artists and albums must have matching lengths")
    total = len(edges)
    same_artist = sum(
        bool(artists[left]) and artists[left] == artists[right] for left, right in edges
    )
    same_album = sum(bool(albums[left]) and albums[left] == albums[right] for left, right in edges)
    return {
        "edge_count": total,
        "same_artist_edges": same_artist,
        "same_album_edges": same_album,
        "same_artist_rate": same_artist / total if total else 0.0,
        "same_album_rate": same_album / total if total else 0.0,
    }
