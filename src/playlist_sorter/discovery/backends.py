"""Community-backend boundary with an honest deterministic fallback."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Mapping, Sequence


def leiden_available() -> bool:
    """Whether both optional packages needed for the Leiden backend are importable."""
    return find_spec("igraph") is not None and find_spec("leidenalg") is not None


def hdbscan_available() -> bool:
    """Expose optional HDBSCAN availability without claiming a fallback is HDBSCAN."""
    return find_spec("hdbscan") is not None


@dataclass(frozen=True)
class ClusterResult:
    labels: tuple[int, ...]
    resolution: float
    seed: int
    backend: str
    fallback_reason: str | None = None


class SeededLeidenBackend:
    """Use seeded Leiden if installed, otherwise deterministic threshold components."""

    def __init__(self, *, min_edge_weight: float = 0.50):
        if not 0 <= min_edge_weight <= 1:
            raise ValueError("min_edge_weight must be between zero and one")
        self.min_edge_weight = min_edge_weight

    def cluster(
        self,
        song_ids: Sequence[str],
        edges: Mapping[tuple[int, int], float],
        *,
        resolution: float,
        seed: int,
    ) -> ClusterResult:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if any(
            left < 0 or right < 0 or left >= len(song_ids) or right >= len(song_ids)
            for left, right in edges
        ):
            raise ValueError("edge endpoints must be valid song positions")
        if leiden_available():
            return self._leiden(song_ids, edges, resolution=resolution, seed=seed)
        return self._components(song_ids, edges, resolution=resolution, seed=seed)

    def _leiden(
        self,
        song_ids: Sequence[str],
        edges: Mapping[tuple[int, int], float],
        *,
        resolution: float,
        seed: int,
    ) -> ClusterResult:
        # Imported only after availability has been established: no optional
        # dependency is represented as installed on minimal CPU environments.
        import igraph  # type: ignore[import-not-found]
        import leidenalg  # type: ignore[import-not-found]

        undirected: dict[tuple[int, int], float] = {}
        for (left, right), score in edges.items():
            if left == right:
                continue
            key = (left, right) if left < right else (right, left)
            undirected[key] = max(float(score), undirected.get(key, float("-inf")))
        graph = igraph.Graph(n=len(song_ids), edges=sorted(undirected), directed=False)
        graph.es["weight"] = [undirected[edge] for edge in sorted(undirected)]
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        return ClusterResult(
            tuple(int(label) for label in partition.membership), resolution, seed, "leiden"
        )

    def _components(
        self,
        song_ids: Sequence[str],
        edges: Mapping[tuple[int, int], float],
        *,
        resolution: float,
        seed: int,
    ) -> ClusterResult:
        # Higher resolutions require stronger retained edges.  The seed is still
        # recorded to keep the result contract congruent with seeded Leiden.
        threshold = min(0.99, self.min_edge_weight + 0.05 * max(0.0, resolution - 1.0))
        adjacent: dict[int, set[int]] = defaultdict(set)
        for (left, right), score in edges.items():
            if float(score) >= threshold:
                adjacent[left].add(right)
                adjacent[right].add(left)
        labels = [-1] * len(song_ids)
        label = 0
        for start in range(len(song_ids)):
            if labels[start] != -1:
                continue
            stack = [start]
            labels[start] = label
            while stack:
                current = stack.pop()
                for neighbor in sorted(adjacent[current], reverse=True):
                    if labels[neighbor] == -1:
                        labels[neighbor] = label
                        stack.append(neighbor)
            label += 1
        return ClusterResult(
            tuple(labels), resolution, seed, "deterministic-components", "leiden-unavailable"
        )
