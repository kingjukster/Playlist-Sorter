# Implementation and qualification status

The integration stage started from combined commit
`08aad41cdbbd7a2804bf7f3ab2df936d9d9316a8`. The final stage commit and evidence
are recorded by the campaign coordinator after all checks complete. This is a
candidate, not a release or authorization to scan private music.

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

The runnable automatic-discovery MVP currently uses the explicit descriptor
lens. The pinned MuQ, MuQ-MuLan, and Qwen entries and cache boundaries exist, but
their heavyweight inference adapters have not been qualified end to end under
WSL2/CUDA. Parquet/Safetensors writers also remain optional runtime backends.

## Explicitly unqualified

- model download, load, or MuQ/MuLan/Qwen inference;
- model-backed multimodal quality and numeric equivalence;
- private lyrics, private historical labels, or a private music library;
- operational 20-song and 200-song gates on user data;
- fresh 10,000-song time/RAM/VRAM/cache/ANN qualification;
- guided private held-out evaluation and 100 blind human comparisons;
- streaming-service writes, deployment, release, or commercial use.
- live YouTube acquisition; automated tests use a synthetic downloader boundary.

Unavailable telemetry is recorded as `null`/`unknown`, never guessed or treated
as zero. No persistent server, model weight, private scan, or external write is
part of the offline integration acceptance.
