from __future__ import annotations

from pathlib import Path

import pytest

from playlist_sorter.embeddings.cache import EmbeddingCache
from playlist_sorter.embeddings.registry import (
    MODEL_REVISIONS,
    MuQAdapter,
    QwenLyricsAdapter,
    adapter_for,
)


class FakeModel:
    def __init__(self, tokens: int = 3) -> None:
        self.audio_calls = 0
        self.text_calls = 0
        self.tokens = tokens

    def embed_audio(self, batch):
        self.audio_calls += 1
        return [[float(index + 1), 0.0] for index, _ in enumerate(batch)]

    def tokenize(self, text):
        return list(range(self.tokens))

    def embed_text(self, batch):
        self.text_calls += 1
        return [[1.0, float(len(tokens))] for tokens in batch]


def _loaded(adapter, fake):
    adapter._model = fake
    return adapter


def test_import_keeps_heavyweight_modules_unloaded_and_loader_is_offline(monkeypatch):
    adapter = adapter_for("muq")
    assert not adapter.loaded
    seen = {}
    monkeypatch.setattr(
        "playlist_sorter.embeddings.registry._snapshot",
        lambda repo, revision: seen.update(repo=repo, revision=revision) or Path("fake"),
    )
    assert adapter.load(lambda snapshot, device, dtype: FakeModel())
    assert seen == {"repo": MODEL_REVISIONS["muq"][0], "revision": MODEL_REVISIONS["muq"][1]}


def test_muq_segment_and_pooled_cache_reuse_without_model_calls(tmp_path):
    fake = FakeModel()
    adapter = _loaded(MuQAdapter("muq", *MODEL_REVISIONS["muq"]), fake)
    cache = EmbeddingCache(tmp_path)
    cold = adapter.embed_audio([0.0, 0.1], [[0.0], [0.1]], source_sha256="audio-v1", cache=cache)
    warm = adapter.embed_audio([0.0, 0.1], [[0.0], [0.1]], source_sha256="audio-v1", cache=cache)
    assert cold.vector == pytest.approx([1.0, 0.0])
    assert warm.vector == cold.vector and warm.telemetry.cache_hit and fake.audio_calls == 1
    with pytest.raises(ValueError, match="flat mono"):
        adapter.embed_audio([0.0], [[[0.0]]])
    with pytest.raises(ValueError, match="24 kHz"):
        adapter.embed_audio([0.0], sample_rate=44_100)


def test_qwen_owns_tokenization_and_uses_windows_weighting_and_cache(tmp_path):
    fake = FakeModel(tokens=800)
    adapter = _loaded(QwenLyricsAdapter("qwen", *MODEL_REVISIONS["qwen"]), fake)
    cache = EmbeddingCache(tmp_path)
    assert adapter.embed_lyrics(None).vector is None
    cold = adapter.embed_lyrics("same\nsame\n same", source_sha256="lyrics-v1", cache=cache)
    warm = adapter.embed_lyrics("same\nsame\n same", source_sha256="lyrics-v1", cache=cache)
    assert cold.vector is not None and warm.telemetry.cache_hit
    assert fake.text_calls == 1  # 3 overlapping windows fit in the default bounded batch.
    assert cold.telemetry.items == 3


def test_qwen_cuda_precision_and_peak_telemetry_are_runtime_derived(monkeypatch):
    class Cuda:
        def is_bf16_supported(self):
            return True

        def is_available(self):
            return True

        def max_memory_allocated(self, device):
            return 123

    class Torch:
        cuda = Cuda()

        def inference_mode(self):
            from contextlib import nullcontext

            return nullcontext()

    monkeypatch.setattr("playlist_sorter.embeddings.registry._torch", lambda: Torch())
    adapter = QwenLyricsAdapter("qwen", *MODEL_REVISIONS["qwen"])
    adapter._device = "cuda"
    telemetry = adapter._telemetry(items=1, batches=1, started=0.0)
    assert telemetry.dtype == "bf16" and telemetry.peak_vram_bytes == 123
