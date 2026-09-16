"""Pinned, offline-only production adapters for optional multimodal embeddings."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol, Sequence

from playlist_sorter.audio.segments import TARGET_RATE
from playlist_sorter.embeddings.cache import EmbeddingCache, embedding_provenance
from playlist_sorter.features.lyrics import chunk_lyrics, pooled_spherical_mean

MODEL_REVISIONS = {
    "muq": ("OpenMuQ/MuQ-large-msd-iter", "0562a57814f6f8bbd9fdea0a25921a2fce1a841a", "fp32"),
    "mulan": ("OpenMuQ/MuQ-MuLan-large", "2e01c796b71dca71b45251384c04cd7b237c9020", "fp32"),
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "bf16"),
}

XLM_ROBERTA_CONFIG = (
    "FacebookAI/xlm-roberta-base",
    "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
)


@dataclass(frozen=True)
class InferenceTelemetry:
    adapter: str
    device: str
    dtype: str
    items: int
    batches: int
    elapsed_seconds: float
    cache_hit: bool = False
    peak_vram_bytes: int | None = None


@dataclass(frozen=True)
class InferenceResult:
    vector: list[float] | None
    telemetry: InferenceTelemetry
    provenance: dict[str, Any] | None = None


class RuntimeModel(Protocol):
    def embed_audio(self, batch: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]: ...
    def embed_text(self, batch: Sequence[Sequence[int]]) -> Sequence[Sequence[float]]: ...
    def tokenize(self, text: str) -> Sequence[int]: ...


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    return [float(value) / norm for value in vector] if norm else [float(value) for value in vector]


def _exact_unique_lines(text: str) -> str:
    """Keep first occurrences of exact non-empty lines; whitespace is significant."""
    seen: set[str] = set()
    result: list[str] = []
    for line in text.splitlines():
        if line and line not in seen:
            seen.add(line)
            result.append(line)
    return "\n".join(result)


def _snapshot(repository: str, revision: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "adapter is intentionally lazy; install huggingface_hub to load it"
        ) from exc
    try:
        return Path(snapshot_download(repo_id=repository, revision=revision, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"adapter is intentionally lazy; exact local snapshot unavailable for {repository}@{revision}"
        ) from exc


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _default_device() -> str:
    torch = _torch()
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _pool_model_output(output: Any) -> list[float]:
    tensor = getattr(output, "pooler_output", None)
    if tensor is None:
        tensor = getattr(output, "last_hidden_state", output)
        if getattr(tensor, "ndim", 0) == 3:
            tensor = tensor.mean(dim=1)
    return tensor.detach().float().cpu().reshape(-1).tolist()


class _TransformersRuntime:
    """Minimal local Transformers bridge used only after an explicit load."""

    def __init__(self, snapshot: Path, device: str, dtype: str, *, audio: bool):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "adapter is intentionally lazy; install torch and transformers to load it"
            ) from exc
        kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": True}
        if dtype == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "fp32":
            kwargs["torch_dtype"] = torch.float32
        self.torch, self.device, self.audio = torch, device, audio
        self.model = AutoModel.from_pretrained(str(snapshot), **kwargs).to(device).eval()
        self.processor = (
            AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
            if audio
            else AutoTokenizer.from_pretrained(
                str(snapshot), local_files_only=True, trust_remote_code=True
            )
        )

    def embed_audio(self, batch: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        result: list[list[float]] = []
        for samples in batch:
            encoded = self.processor(samples, sampling_rate=TARGET_RATE, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            result.append(_pool_model_output(self.model(**encoded)))
        return result

    def tokenize(self, text: str) -> Sequence[int]:
        return self.processor(text, add_special_tokens=False)["input_ids"]

    def embed_text(self, batch: Sequence[Sequence[int]]) -> Sequence[Sequence[float]]:
        result: list[list[float]] = []
        for tokens in batch:
            encoded = self.processor.prepare_for_model(
                list(tokens), return_tensors="pt", padding=True
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            result.append(_pool_model_output(self.model(**encoded)))
        return result


class _MuQRuntime:
    """Official MuQ runtime backed only by exact local Hugging Face snapshots."""

    def __init__(self, snapshot: Path, device: str, variant: str):
        try:
            import torch
            from muq import MuQ, MuQConfig, MuQMuLan
            from transformers import XLMRobertaConfig, XLMRobertaModel
        except ImportError as exc:
            raise RuntimeError(
                "MuQ adapters require the optional research dependencies, including muq"
            ) from exc
        self.torch, self.device, self.variant = torch, device, variant
        if variant == "muq":
            model = MuQ.from_pretrained(str(snapshot), local_files_only=True)
        elif variant == "mulan":
            muq_repository, muq_revision, _ = MODEL_REVISIONS["muq"]
            muq_snapshot = _snapshot(muq_repository, muq_revision)
            xlm_snapshot = _snapshot(*XLM_ROBERTA_CONFIG)
            with (muq_snapshot / "config.json").open(encoding="utf-8") as handle:
                muq_config = MuQConfig(**json.load(handle))
            with (snapshot / "config.json").open(encoding="utf-8") as handle:
                mulan_config = json.load(handle)
            xlm_config = XLMRobertaConfig.from_json_file(str(xlm_snapshot / "config.json"))

            # MuQ-MuLan's constructor normally downloads two base models even though
            # its own checkpoint contains their trained weights. Bootstrap those
            # architectures from pinned local configs, then load the complete outer
            # checkpoint with torch's weights-only unpickler.
            muq_override = MuQ.__dict__.get("from_pretrained")
            xlm_override = XLMRobertaModel.__dict__.get("from_pretrained")
            setattr(
                MuQ,
                "from_pretrained",
                classmethod(lambda cls, *args, **kwargs: cls(muq_config)),
            )
            setattr(
                XLMRobertaModel,
                "from_pretrained",
                classmethod(lambda cls, *args, **kwargs: cls(xlm_config)),
            )
            try:
                model = MuQMuLan(mulan_config)
            finally:
                if muq_override is None:
                    delattr(MuQ, "from_pretrained")
                else:
                    setattr(MuQ, "from_pretrained", muq_override)
                if xlm_override is None:
                    delattr(XLMRobertaModel, "from_pretrained")
                else:
                    setattr(XLMRobertaModel, "from_pretrained", xlm_override)
            state = torch.load(
                snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True
            )
            model.load_state_dict(state, strict=True)
            del state
        else:
            raise ValueError(f"unsupported MuQ runtime variant: {variant}")
        self.model = model.to(device=device, dtype=torch.float32).eval()

    def embed_audio(self, batch: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        result: list[list[float]] = []
        for samples in batch:
            waveform = self.torch.tensor(
                samples, dtype=self.torch.float32, device=self.device
            ).unsqueeze(0)
            if self.variant == "muq":
                output = self.model(waveform, output_hidden_states=True)
                vector = output.last_hidden_state.mean(dim=1)
            else:
                vector = self.model(wavs=waveform, parallel_processing=False)
            result.append(vector.detach().float().cpu().reshape(-1).tolist())
        return result

    def tokenize(self, text: str) -> Sequence[int]:
        raise NotImplementedError("MuQ audio runtimes do not tokenize text")

    def embed_text(self, batch: Sequence[Sequence[int]]) -> Sequence[Sequence[float]]:
        raise NotImplementedError("MuQ audio runtimes do not embed token sequences")


@dataclass
class ProductionAdapter:
    name: str
    repository: str
    revision: str
    dtype: str
    batch_size: int = 8
    _model: RuntimeModel | None = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    @property
    def model_id(self) -> str:
        return f"{self.repository}@{self.revision}"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(
        self,
        loader: Callable[[Path, str, str], RuntimeModel] | None = None,
        *,
        device: str | None = None,
    ) -> RuntimeModel:
        if self._model is not None:
            return self._model
        chosen_device = device or _default_device()
        snapshot = _snapshot(self.repository, self.revision)
        factory = loader or self._runtime_loader
        self._model = factory(snapshot, chosen_device, self._effective_dtype(chosen_device))
        self._device = chosen_device
        return self._model

    def _runtime_loader(self, snapshot: Path, device: str, dtype: str) -> RuntimeModel:
        return _TransformersRuntime(snapshot, device, dtype, audio=False)

    def _effective_dtype(self, device: str) -> str:
        return self.dtype

    def _require_model(self) -> RuntimeModel:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    def _peak_vram(self) -> int | None:
        torch = _torch()
        if torch is None or not self.device.startswith("cuda") or not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(self.device))

    def _begin_vram_measurement(self) -> None:
        torch = _torch()
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _telemetry(
        self, *, items: int, batches: int, started: float, cache_hit: bool = False
    ) -> InferenceTelemetry:
        return InferenceTelemetry(
            self.name,
            self.device,
            self._effective_dtype(self.device),
            items,
            batches,
            perf_counter() - started,
            cache_hit,
            self._peak_vram(),
        )


class MuQAdapter(ProductionAdapter):
    """FP32 mono-24kHz MuQ/MuLan adapter with segment and track cache records."""

    def _runtime_loader(self, snapshot: Path, device: str, dtype: str) -> RuntimeModel:
        return _MuQRuntime(snapshot, device, self.name)

    def embed_audio(
        self,
        samples: Sequence[float],
        segments: Sequence[Sequence[float]] | None = None,
        *,
        sample_rate: int = TARGET_RATE,
        source_sha256: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> InferenceResult:
        self._validate_audio(samples, sample_rate)
        chosen_segments = list(segments) if segments is not None else [samples]
        if not chosen_segments:
            raise ValueError("MuQ requires at least one segment")
        for segment in chosen_segments:
            self._validate_audio(segment, sample_rate)
        started = perf_counter()
        pooled = self._provenance(
            source_sha256, "track", {"method": "normalized_segment_spherical_mean"}
        )
        if cache is not None and pooled is not None:
            cached = cache.get(pooled)
            if cached is not None:
                return InferenceResult(
                    cached,
                    self._telemetry(items=0, batches=0, started=started, cache_hit=True),
                    pooled,
                )
        vectors: list[list[float] | None] = [None] * len(chosen_segments)
        missing: list[tuple[int, Sequence[float], dict[str, Any] | None]] = []
        for index, segment in enumerate(chosen_segments):
            provenance = self._provenance(
                source_sha256, f"segment-{index}", {"method": "normalized_segment"}
            )
            cached = cache.get(provenance) if cache is not None and provenance is not None else None
            if cached is None:
                missing.append((index, segment, provenance))
            else:
                vectors[index] = cached
        model = self._require_model() if missing else None
        if missing:
            self._begin_vram_measurement()
        batches = 0
        context = self._inference_mode()
        with context:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                assert model is not None
                outputs = model.embed_audio([segment for _, segment, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "MuQ runtime returned a vector count different from its batch"
                    )
                for (index, _, provenance), output in zip(batch, outputs, strict=True):
                    vector = _normalize(output)
                    vectors[index] = vector
                    if cache is not None and provenance is not None:
                        cache.put(provenance, vector)
                batches += 1
        complete = [vector for vector in vectors if vector is not None]
        vector = pooled_spherical_mean(complete, [1] * len(complete))
        if cache is not None and pooled is not None:
            cache.put(pooled, vector)
        return InferenceResult(
            vector,
            self._telemetry(items=len(chosen_segments), batches=batches, started=started),
            pooled,
        )

    @staticmethod
    def _validate_audio(samples: Sequence[float], sample_rate: int) -> None:
        if sample_rate != TARGET_RATE:
            raise ValueError("MuQ accepts only mono 24 kHz samples")
        if not samples or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) for value in samples
        ):
            raise ValueError("MuQ accepts only non-empty flat mono samples")

    def _inference_mode(self) -> Any:
        torch = _torch()
        return torch.inference_mode() if torch is not None else nullcontext()

    def _provenance(
        self, source_sha256: str | None, segment_id: str, pooling: dict[str, str]
    ) -> dict[str, Any] | None:
        if source_sha256 is None:
            return None
        return embedding_provenance(
            source_sha256=source_sha256,
            model_repository=self.repository,
            model_revision=self.revision,
            preprocessing={
                "sample_rate": TARGET_RATE,
                "mono": True,
                "dtype": "fp32",
                "batch_size": self.batch_size,
            },
            pooling=pooling,
            feature_view=self.name,
            segment_id=segment_id,
        )


class QwenLyricsAdapter(ProductionAdapter):
    """Qwen-owned tokenizer with exact-line dedupe and token-weighted pooling."""

    def _effective_dtype(self, device: str) -> str:
        torch = _torch()
        return (
            "bf16"
            if device.startswith("cuda") and torch is not None and torch.cuda.is_bf16_supported()
            else "fp32"
        )

    def embed_lyrics(
        self,
        text: str | None,
        *,
        source_sha256: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> InferenceResult:
        started = perf_counter()
        cleaned = _exact_unique_lines(text or "")
        if not cleaned:
            return InferenceResult(None, self._telemetry(items=0, batches=0, started=started))
        pooled = self._provenance(
            source_sha256, "lyrics-pooled", {"method": "token_weighted_spherical_mean"}
        )
        if cache is not None and pooled is not None:
            cached = cache.get(pooled)
            if cached is not None:
                return InferenceResult(
                    cached,
                    self._telemetry(items=0, batches=0, started=started, cache_hit=True),
                    pooled,
                )
        model = self._require_model()
        chunks = chunk_lyrics(model.tokenize(cleaned), 384, 64)
        if not chunks:
            return InferenceResult(
                None, self._telemetry(items=0, batches=0, started=started), pooled
            )
        vectors: list[list[float] | None] = [None] * len(chunks)
        missing: list[tuple[int, Sequence[int], dict[str, Any] | None]] = []
        for index, chunk in enumerate(chunks):
            provenance = self._provenance(
                source_sha256, f"lyrics-chunk-{index}", {"method": "normalized_chunk"}
            )
            cached = cache.get(provenance) if cache is not None and provenance is not None else None
            if cached is None:
                missing.append((index, chunk, provenance))
            else:
                vectors[index] = cached
        batches = 0
        if missing:
            self._begin_vram_measurement()
        torch = _torch()
        with torch.inference_mode() if torch is not None else nullcontext():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                outputs = model.embed_text([chunk for _, chunk, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "Qwen runtime returned a vector count different from its batch"
                    )
                for (index, _, provenance), output in zip(batch, outputs, strict=True):
                    vector = _normalize(output)
                    vectors[index] = vector
                    if cache is not None and provenance is not None:
                        cache.put(provenance, vector)
                batches += 1
        complete = [vector for vector in vectors if vector is not None]
        vector = pooled_spherical_mean(complete, [len(chunk) for chunk in chunks])
        if cache is not None and pooled is not None:
            cache.put(pooled, vector)
        return InferenceResult(
            vector, self._telemetry(items=len(chunks), batches=batches, started=started), pooled
        )

    def _provenance(
        self, source_sha256: str | None, segment_id: str, pooling: dict[str, str]
    ) -> dict[str, Any] | None:
        if source_sha256 is None:
            return None
        return embedding_provenance(
            source_sha256=source_sha256,
            model_repository=self.repository,
            model_revision=self.revision,
            preprocessing={
                "dedupe": "exact_nonempty_line",
                "chunk_tokens": 384,
                "overlap_tokens": 64,
                "batch_size": self.batch_size,
            },
            pooling=pooling,
            feature_view="lyrics",
            segment_id=segment_id,
        )


def adapter_for(name: str) -> ProductionAdapter:
    try:
        repository, revision, dtype = MODEL_REVISIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown embedding adapter: {name}") from exc
    return (
        MuQAdapter(name, repository, revision, dtype)
        if name in {"muq", "mulan"}
        else QwenLyricsAdapter(name, repository, revision, dtype)
    )
