# Implementation and qualification status

This candidate is based on `c5d2ca48afdffae091e914cc12c803bdd31814b7`. It is
not a release, permission to scan private music, or evidence that a live model
run occurred.

## Implemented

- Python `>=3.12,<3.13` package metadata and locked dependencies.
- Runtime doctor/resource checks and pinned model/license registry.
- Read-only deterministic catalog scanning, exact SHA-256 identity, optional
  Chromaprint evidence, WAV metadata, structured corrupt-input failures.
- Versioned Pydantic contracts with fail-closed versions/fields.
- Deterministic descriptor, graph, discovery, guidance, naming, evaluation,
  telemetry, review, and preview-first atomic export services.
- Generated-audio scan-to-export integration plus metamorphic path/order,
  duplicate-byte, missing-lyrics, corrupt, short-track, and perturbation cases.
- WSL2/Python 3.12 catalog-only 20/200 gate scripts with local evidence output.
- Real CLI orchestration for catalog, cached descriptor features, gated discovery,
  localhost review, previewed export, and leakage-controlled benchmark scoring.
- Chromaprint-plus-duration musical-variant grouping while preserving SHA-256
  file identity.
- Rights-aware, manifest-driven YouTube acquisition for individually authorized
  videos, with yt-dlp/FFmpeg provenance, SHA-256 receipts, and verified rerun
  skips. Search, playlists, authentication, cookies, and bypasses are excluded.

## Known interface limits

`descriptors` is the runnable automatic-discovery profile. `research` is a
separate exact profile: it uses pinned, lazy MuQ, MuQ-MuLan, and Qwen adapters
from local snapshots and does not silently replace a missing model view. It adds
acoustic, semantic-audio, and optional lyrics views; absent lyrics abstain.
Research cache hits skip inference and are bound to complete provenance. The
default pipeline cache is JSON; Safetensors/Parquet remains optional.

## Explicitly unqualified

- model download, load, or MuQ/MuLan/Qwen inference;
- model-backed multimodal quality and numeric equivalence;
- private lyrics, private historical labels, or a private music library;
- the required one-song model smoke and conditional 20-song model smoke on the
  explicitly authorized private subset;
- operational 20-song and 200-song catalog-only gates on user data;
- fresh 10,000-song time/RAM/VRAM/cache/ANN qualification;
- guided private held-out evaluation and 100 blind human comparisons;
- streaming-service writes, deployment, release, or commercial use.
- live YouTube acquisition; automated tests use a synthetic downloader boundary.

Unavailable telemetry is recorded as `null`/`unknown`, never guessed or treated
as zero. No persistent server, model weight, private scan, or external write is
part of offline acceptance. WSL2/Python 3.12/CUDA is the live-model qualification
environment; the PowerShell smoke/benchmark wrappers are catalog-only. A current
resource preflight is mandatory before model load, inference, or download, and
this campaign does not authorize a 98-song run.
