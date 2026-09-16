"""Model metadata only: importing this module never imports ML libraries."""

from __future__ import annotations
from dataclasses import dataclass

MODEL_REVISIONS = {
    "muq": ("OpenMuQ/MuQ-large-msd-iter", "0562a57814f6f8bbd9fdea0a25921a2fce1a841a", "fp32"),
    "mulan": ("OpenMuQ/MuQ-MuLan-large", "2e01c796b71dca71b45251384c04cd7b237c9020", "fp32"),
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "bf16"),
}


@dataclass(frozen=True)
class LazyAdapter:
    name: str
    repository: str
    revision: str
    dtype: str

    def load(self) -> object:
        raise RuntimeError(
            f"{self.name} adapter is intentionally lazy; install its runtime adapter and explicitly load {self.repository}@{self.revision}"
        )


def adapter_for(name: str) -> LazyAdapter:
    try:
        repository, revision, dtype = MODEL_REVISIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown embedding adapter: {name}") from exc
    return LazyAdapter(name, repository, revision, dtype)
