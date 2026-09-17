"""Baseline declarations that never imply optional backends are installed."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal


@dataclass(frozen=True)
class BaselineSpec:
    identifier: str
    label: str
    kind: Literal["baseline", "system"]
    required_modules: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return all(find_spec(module) is not None for module in self.required_modules)

    @property
    def unavailable_reason(self) -> str | None:
        missing = [module for module in self.required_modules if find_spec(module) is None]
        return None if not missing else f"optional modules unavailable: {', '.join(missing)}"


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("metadata_only", "Metadata-only clustering", "baseline"),
    BaselineSpec("descriptor_kmeans", "Descriptor k-means", "baseline", ("sklearn",)),
    BaselineSpec(
        "pooled_muq_mulan_hdbscan", "Pooled MuQ-MuLan + HDBSCAN", "baseline", ("hdbscan",)
    ),
    BaselineSpec("single_lens_leiden", "Single-lens Leiden", "baseline", ("igraph", "leidenalg")),
    BaselineSpec(
        "fused_leiden_no_consensus",
        "Fused Leiden without consensus",
        "baseline",
        ("igraph", "leidenalg"),
    ),
    BaselineSpec("full_system", "Full system", "system"),
)


def available_baselines() -> tuple[BaselineSpec, ...]:
    """Return declarations with live import availability, without importing optional engines."""
    return BASELINES
