from __future__ import annotations

from copy import deepcopy
import json

import pytest

from playlist_sorter.embeddings.cache import (
    EmbeddingCache,
    SafetensorsParquetCache,
    embedding_provenance,
    validate_embedding_provenance,
)


def _provenance(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "source_sha256": "source-v1",
        "model_repository": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": "revision-v1",
        "preprocessing": {"chunk_tokens": 384, "overlap_tokens": 64},
        "pooling": {"method": "token_weighted_spherical_mean"},
        "feature_view": "lyrics",
        "segment_id": "lyrics-pooled",
    }
    values.update(changes)
    return embedding_provenance(**values)  # type: ignore[arg-type]


def test_canonical_provenance_is_copied_and_complete():
    preprocessing = {"chunk_tokens": 384}
    provenance = embedding_provenance(
        source_fingerprint="source-v1",
        model_repository="model/repository",
        model_revision="revision-v1",
        preprocessing=preprocessing,
        pooling={"method": "mean"},
        feature_view="lyrics",
    )

    preprocessing["chunk_tokens"] = 512
    assert provenance == {
        "source_fingerprint": "source-v1",
        "model_repository": "model/repository",
        "model_revision": "revision-v1",
        "preprocessing": {"chunk_tokens": 384},
        "pooling": {"method": "mean"},
        "feature_view": "lyrics",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"source_sha256": None},
        {"source_fingerprint": "also-source"},
        {"model_repository": ""},
        {"preprocessing": ["not-a-mapping"]},
        {"pooling": "mean"},
        {"feature_view": 3},
        {"segment_id": 0},
        {"unexpected": "value"},
    ],
)
def test_canonical_provenance_rejects_incomplete_or_ambiguous_values(change):
    candidate = _provenance()
    candidate.update(change)
    with pytest.raises((TypeError, ValueError)):
        validate_embedding_provenance(candidate)


def test_cache_reuses_exact_canonical_provenance_and_invalidates_every_input(tmp_path):
    cache = EmbeddingCache(tmp_path)
    provenance = _provenance()
    cache.put(provenance, [1.0, 0.0])

    assert cache.get(deepcopy(provenance)) == [1.0, 0.0]
    variants = [
        _provenance(source_sha256="source-v2"),
        _provenance(model_repository="another/model"),
        _provenance(model_revision="revision-v2"),
        _provenance(preprocessing={"chunk_tokens": 512, "overlap_tokens": 64}),
        _provenance(pooling={"method": "unweighted_mean"}),
        _provenance(feature_view="acoustic"),
        _provenance(segment_id="lyrics-segment-0"),
    ]
    assert all(cache.get(variant) is None for variant in variants)


def test_safetensors_parquet_paths_change_for_every_canonical_provenance_input(tmp_path):
    cache = SafetensorsParquetCache(tmp_path)
    provenance = _provenance()
    vector_path, index_path = cache.paths(provenance)
    variants = [
        _provenance(source_sha256="source-v2"),
        _provenance(model_repository="another/model"),
        _provenance(model_revision="revision-v2"),
        _provenance(preprocessing={"chunk_tokens": 512, "overlap_tokens": 64}),
        _provenance(pooling={"method": "unweighted_mean"}),
        _provenance(feature_view="acoustic"),
        _provenance(segment_id="lyrics-segment-0"),
    ]
    assert all(cache.paths(variant) != (vector_path, index_path) for variant in variants)


def test_safetensors_parquet_cache_rejects_index_with_noncanonical_provenance(tmp_path):
    _ = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("safetensors.numpy")

    cache = SafetensorsParquetCache(tmp_path)
    provenance = _provenance()
    _, index_path = cache.put(provenance, [1.0, 0.0])
    assert cache.get(provenance) == [1.0, 0.0]

    pq.write_table(
        pa.Table.from_pylist(
            [{"cache_key": index_path.stem, "provenance": json.dumps(provenance, indent=2)}]
        ),
        index_path,
    )
    assert cache.get(provenance) is None
