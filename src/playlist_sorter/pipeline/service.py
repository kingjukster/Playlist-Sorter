"""End-to-end producers for the small canonical run artifact set.

The service deliberately supports a descriptor-only offline path. Pinned model
adapters can add views later without changing the run contracts; absent weights
are reported rather than silently replaced with fabricated embeddings.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Iterable

from playlist_sorter.audio import AudioDecodeError, decode_audio_mono_24khz, select_segments
from playlist_sorter.catalog import scan_library
from playlist_sorter.core.contracts import (
    GuidanceSet,
    Membership,
    PlaylistCandidate,
    RunManifest,
    SongRecord,
    parse_feature_profile,
)
from playlist_sorter.discovery import (
    CandidateEvidence,
    Community,
    SeededLeidenBackend,
    community_consensus,
    default_perturbations,
    select_candidates,
)
from playlist_sorter.evaluation import ndcg_at_k, precision_at_k
from playlist_sorter.embeddings import EmbeddingCache, adapter_for
from playlist_sorter.features import describe_samples
from playlist_sorter.guidance import apply_guidance, validate_guidance

GENERATOR_VERSION = "0.1.0"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _collection(items: Iterable[Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "items": [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in items
        ],
    }


def _read_collection(run_dir: Path, filename: str) -> list[dict[str, Any]]:
    path = run_dir / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {filename}; run the preceding pipeline stage") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError(f"unsupported or malformed {filename}")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"malformed {filename}")
    return items


def _manifest(run_dir: Path, candidate_ids: list[str] | None = None) -> RunManifest:
    existing = run_dir / "manifest.json"
    created_at = datetime.now(UTC)
    run_id = run_dir.name
    if existing.is_file():
        prior = RunManifest.model_validate_json(existing.read_text(encoding="utf-8"))
        created_at, run_id = prior.created_at, prior.run_id
    artifacts = sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".parquet", ".safetensors"}
    )
    return RunManifest(
        run_id=run_id,
        created_at=created_at,
        discovery_run=run_id,
        artifacts=artifacts,
        candidates=candidate_ids or [],
        generator_version=GENERATOR_VERSION,
    )


def scan_to_run(library: str | Path, output: str | Path) -> dict[str, Any]:
    """Read a library and create the initial canonical run without touching media."""
    root, run_dir = Path(library).expanduser(), Path(output).expanduser()
    songs, failures = scan_library(root)
    if not root.is_dir():
        raise ValueError(f"library is not a directory: {root}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("output run directory must be empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        SongRecord(
            song_id=song.song_id,
            source_path=song.source_path,
            title=Path(song.source_path).stem,
            duration_seconds=song.duration_seconds,
            codec=song.codec,
            sample_rate=int(song.metadata["sample_rate"])
            if "sample_rate" in song.metadata
            else None,
            channels=int(song.metadata["channels"]) if "channels" in song.metadata else None,
            file_size=song.size_bytes,
            integrity_fingerprint=song.sha256,
            variant_group_id=song.variant_group_id,
        )
        for song in songs
    ]
    _atomic_json(run_dir / "catalog.json", _collection([song.__dict__ for song in songs]))
    _atomic_json(
        run_dir / "catalog_failures.json", _collection([item.__dict__ for item in failures])
    )
    _atomic_json(run_dir / "songs.json", _collection(records))
    _atomic_json(run_dir / "playlists.json", _collection([]))
    _atomic_json(run_dir / "memberships.json", _collection([]))
    manifest = _manifest(run_dir)
    _atomic_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    return {
        "run": str(run_dir.resolve()),
        "songs": len(songs),
        "failures": len(failures),
        "variant_groups": len({song.variant_group_id for song in songs}),
    }


def _result_vector(result: Any) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    """Normalize the stable InferenceResult shape without importing ML code."""
    vector = getattr(result, "vector", result.get("vector") if isinstance(result, dict) else None)
    if vector is None:
        raise TypeError("adapter result must expose vector")
    provenance = getattr(result, "provenance", None) or {}
    telemetry = getattr(result, "telemetry", None)
    telemetry_record = dict(vars(telemetry)) if telemetry is not None else {}
    return [float(item) for item in vector], dict(provenance), telemetry_record


def build_features(
    run: str | Path,
    profile: str = "descriptors",
    *,
    adapters: dict[str, Any] | None = None,
    cache: Any | None = None,
) -> dict[str, Any]:
    """Build descriptor features or injected model-backed research views.

    ``adapters`` and ``cache`` are deliberately injection points: production
    callers may resolve them through ``adapter_for(name)``, while tests can use
    deterministic fakes without importing ML runtimes.
    """
    selected_profile = parse_feature_profile(profile)
    run_dir = Path(run)
    songs = [SongRecord.model_validate(item) for item in _read_collection(run_dir, "songs.json")]
    previous = (
        {item["source_fingerprint"]: item for item in _read_collection(run_dir, "features.json")}
        if (run_dir / "features.json").is_file()
        else {}
    )
    features: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    cache_hits = 0
    started = perf_counter()
    updated_songs: list[SongRecord] = []
    for song in songs:
        cached = previous.get(song.integrity_fingerprint)
        if (
            selected_profile.descriptors
            and cached
            and cached.get("preprocessing_config")
            == {
                "sample_rate": 24000,
                "window_seconds": 10.0,
            }
        ):
            features.append(cached)
            cache_hits += 1
            updated_songs.append(
                song.model_copy(update={"preprocessing_state": "complete", "error": None})
            )
            continue
        item_started = perf_counter()
        try:
            samples, sample_rate = decode_audio_mono_24khz(song.source_path)
            descriptor = describe_samples(samples, sample_rate)
            chosen = select_segments(samples, sample_rate)
        except AudioDecodeError as exc:
            failures.append({"song_id": song.song_id, "code": exc.code, "detail": exc.detail})
            updated_songs.append(
                song.model_copy(update={"preprocessing_state": "failed", "error": exc.code})
            )
            continue
        vector = [
            descriptor["duration_seconds"],
            descriptor["rms"],
            descriptor["peak"],
            descriptor["zero_crossing_rate"],
        ]
        if selected_profile.descriptors:
            features.append(
                {
                    "song_id": song.song_id,
                    "feature_view": "descriptors",
                    "vector": vector,
                    "source_fingerprint": song.integrity_fingerprint,
                    "preprocessing_config": {
                        "sample_rate": sample_rate,
                        "window_seconds": 10.0,
                    },
                    "elapsed_seconds": perf_counter() - item_started,
                }
            )
        else:
            adapters = adapters or {name: adapter_for(name) for name in ("muq", "mulan", "qwen")}
            cache = cache or EmbeddingCache(run_dir / "embedding_cache")
            sample_segments = [
                samples[
                    round(segment.start_seconds * sample_rate) : round(
                        min(segment.end_seconds, len(samples) / sample_rate) * sample_rate
                    )
                ]
                for segment in chosen
            ]
            for view, adapter_name in (("acoustic", "muq"), ("semantic_audio", "mulan")):
                result = adapters[adapter_name].embed_audio(
                    samples,
                    sample_segments,
                    sample_rate=sample_rate,
                    source_sha256=song.integrity_fingerprint,
                    cache=cache,
                )
                embedding, provenance, telemetry = _result_vector(result)
                cache_hits += int(bool(telemetry.get("cache_hit")))
                features.append(
                    {
                        "song_id": song.song_id,
                        "feature_view": view,
                        "vector": embedding,
                        "source_fingerprint": song.integrity_fingerprint,
                        "preprocessing_config": {"sample_rate": 24000, "mono": True},
                        "model_repository": getattr(
                            adapters[adapter_name], "repository", adapter_name
                        ),
                        "model_revision": getattr(adapters[adapter_name], "revision", "injected"),
                        "provenance": provenance,
                        "telemetry": telemetry,
                    }
                )
            if song.lyrics and song.lyrics.strip():
                result = adapters["qwen"].embed_lyrics(
                    song.lyrics,
                    source_sha256=song.integrity_fingerprint,
                    cache=cache,
                )
                embedding, provenance, telemetry = _result_vector(result)
                cache_hits += int(bool(telemetry.get("cache_hit")))
                features.append(
                    {
                        "song_id": song.song_id,
                        "feature_view": "lyrics",
                        "vector": embedding,
                        "source_fingerprint": song.integrity_fingerprint,
                        "preprocessing_config": {"lyrics": True},
                        "model_repository": getattr(adapters["qwen"], "repository", "qwen"),
                        "model_revision": getattr(adapters["qwen"], "revision", "injected"),
                        "provenance": provenance,
                        "telemetry": telemetry,
                    }
                )
        segments.extend(
            {
                "segment_id": f"{song.song_id}:{index}",
                "song_id": song.song_id,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "index": index,
                "reason": segment.reason,
            }
            for index, segment in enumerate(chosen)
        )
        updated_songs.append(
            song.model_copy(update={"preprocessing_state": "complete", "error": None})
        )
    _atomic_json(run_dir / "features.json", _collection(features))
    _atomic_json(run_dir / "segments.json", _collection(segments))
    _atomic_json(run_dir / "feature_failures.json", _collection(failures))
    _atomic_json(run_dir / "songs.json", _collection(updated_songs))
    _atomic_json(run_dir / "playlists.json", _collection([]))
    _atomic_json(run_dir / "memberships.json", _collection([]))
    _atomic_json(run_dir / "manifest.json", _manifest(run_dir).model_dump(mode="json"))
    return {
        "run": str(run_dir.resolve()),
        "features": len(features),
        "failures": len(failures),
        "cache_hits": cache_hits,
        "elapsed_seconds": perf_counter() - started,
        "model_inference": (
            "not_run; descriptor lens only"
            if selected_profile.descriptors
            else "model-backed research views"
        ),
    }


def _standardize(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return []
    columns = list(zip(*rows, strict=True))
    means = [sum(column) / len(column) for column in columns]
    scales = [
        max((sum((value - mean) ** 2 for value in column) / len(column)) ** 0.5, 1e-9)
        for column, mean in zip(columns, means, strict=True)
    ]
    return [
        [(value - means[index]) / scales[index] for index, value in enumerate(row)] for row in rows
    ]


def _cosine(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norms if norms else 0.0


def _edges(vectors: list[list[float]], k: int = 30) -> dict[tuple[int, int], float]:
    edges: dict[tuple[int, int], float] = {}
    for left, vector in enumerate(vectors):
        ranked = sorted(
            (
                (right, (_cosine(vector, other) + 1) / 2)
                for right, other in enumerate(vectors)
                if right != left
            ),
            key=lambda item: (-item[1], item[0]),
        )[:k]
        edges.update({(left, right): score for right, score in ranked})
    return edges


def _communities(song_ids: list[str], labels: tuple[int, ...], prefix: str) -> list[Community]:
    grouped: dict[int, set[str]] = defaultdict(set)
    for song_id, label in zip(song_ids, labels, strict=True):
        grouped[label].add(song_id)
    return [
        Community(f"{prefix}-{label}", frozenset(members), "descriptors")
        for label, members in sorted(grouped.items())
    ]


def _granularity(size: int, total: int) -> str:
    fraction = size / total
    if fraction <= 0.03:
        return "narrow"
    if fraction <= 0.10:
        return "medium"
    return "broad"


def _load_guidance(path: str | Path) -> GuidanceSet:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("YAML guidance requires PyYAML") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("guidance must be a YAML object")
    return GuidanceSet.model_validate(payload)


def discover_run(run: str | Path, guidance: str | Path | None = None) -> dict[str, Any]:
    """Discover stable descriptor communities and write canonical candidates."""
    run_dir = Path(run)
    songs = [SongRecord.model_validate(item) for item in _read_collection(run_dir, "songs.json")]
    feature_rows = _read_collection(run_dir, "features.json")
    by_song = {
        item["song_id"]: item["vector"]
        for item in feature_rows
        if item.get("feature_view") == "descriptors"
    }
    usable = [song.song_id for song in songs if song.song_id in by_song]
    vectors = _standardize([by_song[song_id] for song_id in usable])
    edges = _edges(vectors)
    guidance_id: str | None = None
    if guidance is not None:
        guidance_set = _load_guidance(guidance)
        validate_guidance(guidance_set, set(usable))
        named_edges: dict[tuple[str, str], float] = {}
        for (left, right), score in edges.items():
            left_id, right_id = usable[left], usable[right]
            pair = (left_id, right_id) if left_id < right_id else (right_id, left_id)
            named_edges[pair] = max(score, named_edges.get(pair, 0.0))
        vector_map = dict(zip(usable, vectors, strict=True))
        guidance_edges: dict[tuple[str, str], float] | None = None
        if guidance_set.positive_song_ids:
            positive = [vector_map[song_id] for song_id in guidance_set.positive_song_ids]
            positive_centroid = [
                sum(vector[dimension] for vector in positive) / len(positive)
                for dimension in range(len(positive[0]))
            ]
            negative_centroid = None
            if guidance_set.negative_song_ids:
                negative = [vector_map[song_id] for song_id in guidance_set.negative_song_ids]
                negative_centroid = [
                    sum(vector[dimension] for vector in negative) / len(negative)
                    for dimension in range(len(negative[0]))
                ]
            affinities = {
                song_id: max(
                    0.0,
                    min(
                        1.0,
                        (_cosine(vector, positive_centroid) + 1.0) / 2.0
                        - (
                            max(0.0, _cosine(vector, negative_centroid)) / 2.0
                            if negative_centroid
                            else 0.0
                        ),
                    ),
                )
                for song_id, vector in vector_map.items()
            }
            guidance_edges = {
                pair: (affinities[pair[0]] + affinities[pair[1]]) / 2.0 for pair in named_edges
            }
        guided = apply_guidance(
            guidance_set,
            named_edges,
            known_song_ids=set(usable),
            vectors=vector_map,
            guidance_edges=guidance_edges,
        )
        positions = {song_id: index for index, song_id in enumerate(usable)}
        edges = {
            (positions[left], positions[right]): score
            for (left, right), score in guided.comparison.guided_edges.items()
        }
        guidance_id = guidance_set.guidance_id
        _atomic_json(
            run_dir / "unguided_graph.json",
            {
                f"{left}|{right}": score
                for (left, right), score in guided.comparison.unguided_edges.items()
            },
        )
    backend = SeededLeidenBackend(min_edge_weight=0.50)
    evidence: list[CandidateEvidence] = []
    for resolution in (0.25, 0.5, 1.0, 2.0, 4.0):
        reference_result = backend.cluster(usable, edges, resolution=resolution, seed=0)
        reference = _communities(usable, reference_result.labels, f"r{resolution:g}")
        trials: list[list[Community]] = []
        for perturbation in default_perturbations():
            perturbed = {
                edge: max(0.0, min(1.0, score * perturbation.edge_multiplier))
                for edge, score in edges.items()
            }
            result = backend.cluster(
                usable, perturbed, resolution=resolution, seed=perturbation.cluster_seed
            )
            trials.append(_communities(usable, result.labels, f"t{perturbation.trial}"))
        for community in reference:
            if len(community.member_song_ids) < 8:
                continue
            stability, recurrence = community_consensus(community, trials)
            member_positions = [usable.index(song_id) for song_id in community.member_song_ids]
            inside = [
                score
                for (left, right), score in edges.items()
                if left in member_positions and right in member_positions
            ]
            outside = [
                score
                for (left, right), score in edges.items()
                if left in member_positions and right not in member_positions
            ]
            cohesion = sum(inside) / len(inside) if inside else 0.0
            nearest_outside = (
                sum(sorted(outside, reverse=True)[: max(1, len(member_positions))])
                / max(1, min(len(outside), len(member_positions)))
                if outside
                else 0.0
            )
            centroid = tuple(
                sum(vectors[index][dimension] for index in member_positions) / len(member_positions)
                for dimension in range(len(vectors[0]))
            )
            candidate_id = hashlib.sha256(
                "|".join(sorted(community.member_song_ids)).encode()
            ).hexdigest()[:16]
            evidence.append(
                CandidateEvidence(
                    candidate_id,
                    community.member_song_ids,
                    _granularity(len(member_positions), len(usable)),
                    centroid,
                    stability,
                    sum(value >= 0.8 for value in recurrence.values()) / len(recurrence),
                    cohesion,
                    max(0.0, cohesion - nearest_outside),
                    1.0 - len(member_positions) / len(usable),
                    cohesion - nearest_outside,
                    "descriptors",
                    resolution,
                )
            )
    selected = select_candidates(evidence, library_size=max(1, len(usable))) if usable else ()
    candidates: list[PlaylistCandidate] = []
    memberships: list[Membership] = []
    for item in selected:
        ordered = sorted(item.member_song_ids)
        candidate = PlaylistCandidate(
            candidate_id=item.candidate_id,
            name=f"Discovered {item.granularity} {item.candidate_id[:6]}",
            granularity=item.granularity,
            resolution=item.resolution,
            member_song_ids=ordered,
            representative_song_ids=ordered[: min(5, len(ordered))],
            core_song_ids=ordered,
            boundary_song_ids=ordered[-min(3, len(ordered)) :],
            excluded_song_ids=[
                song_id for song_id in usable if song_id not in item.member_song_ids
            ][:5],
            stability=item.stability,
            cohesion=max(0.0, min(1.0, item.cohesion)),
            separation=max(0.0, min(1.0, item.distinctiveness)),
            novelty=max(0.0, min(1.0, item.novelty)),
            coverage=len(ordered) / len(usable),
            naming_evidence=["descriptor similarity", item.granularity],
            lineage={key: list(value) for key, value in item.lineage.items()},
            guidance_ids=[guidance_id] if guidance_id else [],
        )
        candidates.append(candidate)
        memberships.extend(
            Membership(
                song_id=song_id,
                candidate_id=item.candidate_id,
                membership_score=1.0,
                threshold=0.6,
            )
            for song_id in ordered
        )
    _atomic_json(run_dir / "playlists.json", _collection(candidates))
    _atomic_json(run_dir / "memberships.json", _collection(memberships))
    _atomic_json(
        run_dir / "manifest.json",
        _manifest(run_dir, [item.candidate_id for item in candidates]).model_dump(mode="json"),
    )
    summary = {
        "run": str(run_dir.resolve()),
        "usable_songs": len(usable),
        "candidates": len(candidates),
        "lens": "descriptors",
        "guidance": guidance_id,
        "abstained": not candidates,
    }
    _atomic_json(run_dir / "discovery_summary.json", summary)
    return summary


def evaluate_run(run: str | Path, benchmark: str | Path) -> dict[str, Any]:
    """Evaluate discovered member rankings against private labels without tuning."""
    run_dir = Path(run)
    candidates = [
        PlaylistCandidate.model_validate(item)
        for item in _read_collection(run_dir, "playlists.json")
    ]
    benchmark_path = Path(benchmark)
    if benchmark_path.suffix.lower() == ".csv":
        labels: dict[str, list[str]] = defaultdict(list)
        with benchmark_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[row["playlist"]].append(row["song_id"])
    else:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        labels = payload.get("labels", payload)
    if not isinstance(labels, dict) or not all(
        isinstance(value, list) for value in labels.values()
    ):
        raise ValueError("benchmark must map playlist names to song ID lists")
    per_playlist: dict[str, dict[str, float]] = {}
    for name, relevant in labels.items():
        ranked = max(
            (candidate.member_song_ids for candidate in candidates),
            key=lambda ids: ndcg_at_k(ids, relevant),
            default=[],
        )
        per_playlist[str(name)] = {
            "ndcg_at_20": ndcg_at_k(ranked, relevant),
            "precision_at_20": precision_at_k(ranked, relevant),
        }
    macro = (
        sum(item["ndcg_at_20"] for item in per_playlist.values()) / len(per_playlist)
        if per_playlist
        else 0.0
    )
    result = {"macro_ndcg_at_20": macro, "playlists": per_playlist, "selection_only": True}
    _atomic_json(run_dir / "evaluation.json", result)
    return result
