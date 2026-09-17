"""Create stable candidate labels without participating in discovery."""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from playlist_sorter.core.contracts import PlaylistCandidate


@dataclass(frozen=True)
class NamingResult:
    """A generated label and its inspectable, ordered evidence."""

    candidate_id: str
    name: str
    explanation: str
    dominant_evidence: tuple[str, ...]


def _normalized_evidence(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize, count, and sort evidence independently of input ordering."""
    cleaned = [" ".join(value.split()) for value in values if value and value.strip()]
    counts = Counter(cleaned)
    return tuple(sorted(counts, key=lambda value: (-counts[value], value.casefold(), value)))


def name_candidate(candidate: PlaylistCandidate) -> NamingResult:
    """Name from candidate evidence only; the candidate is never modified."""
    evidence = _normalized_evidence(candidate.naming_evidence)
    dominant = evidence[:3]
    if dominant:
        name = " / ".join(dominant[:2])
        explanation = "Dominant evidence: " + ", ".join(dominant) + "."
    else:
        name = f"Playlist {candidate.candidate_id}"
        explanation = "No naming evidence was supplied; using a stable fallback label."
    return NamingResult(
        candidate_id=candidate.candidate_id,
        name=name,
        explanation=explanation,
        dominant_evidence=dominant,
    )


def explain_candidate(candidate: PlaylistCandidate) -> str:
    """Return the deterministic explanation used for the generated name."""
    return name_candidate(candidate).explanation
