# Implementation and qualification status

This candidate is based on `7cbba9e`. It is
not a release or standing permission to scan or acquire private music. A bounded,
explicitly authorized 20-song live-model qualification has occurred.

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

- Qwen lyric inference (the qualified private song had no attached lyrics);
- model-backed multimodal quality and numeric equivalence;
- private lyrics, private historical labels, or a private music library;
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

## Live one-song qualification

On 2026-09-16, an authorized one-song WSL2/CUDA research-profile smoke produced
finite, unit-normalized MuQ acoustic (1024-dimensional) and MuQ-MuLan
semantic-audio (512-dimensional) vectors with zero item failures. Cold feature
execution took 17.19 seconds. Observed total GPU memory peaked at 6869 MiB,
temperature at 42 C, and power at 94.19 W; there was no OOM, NaN, driver reset,
or host paging. The unchanged rerun served both feature views from cache in
4.10 seconds. Evidence remains in the private run directory and is not committed.

## Live 20-song qualification

On 2026-09-16, the authorized WSL2/CUDA research-profile gate processed 20
songs into 40 model-backed feature artifacts and 111 segments with zero item
failures. The 20 MuQ acoustic vectors (1024 dimensions) and 20 MuQ-MuLan
semantic-audio vectors (512 dimensions) were finite and unit normalized. Cold
feature execution took 59.65 seconds, or 0.335 songs/second. Observed peak total
GPU memory was 9209 MiB, peak GPU utilization was 90%, minimum available host
RAM was 6374.6 MiB, temperature peaked at 47 C, and power at 103.49 W. There was
no observed OOM, NaN, host paging, driver reset, data loss, or unhandled song
failure.

The unchanged rerun served all 40 artifacts from provenance-validated cache:
100% cache reuse, 0.055 seconds of internal feature work, and 0.65 seconds of
command wall time, approximately 91.8x faster than the cold run. Discovery used
the acoustic, semantic-audio, and fused lenses and safely abstained with zero
qualified candidates for this small subset. Private logs and run artifacts remain
outside Git. This qualifies the bounded 20-song execution gate, not the 200-song,
approximately 782-song, 10,000-song, lyric-model, or quality-evaluation gates.

## Live 775-file qualification

On 2026-09-16, the staged full-library campaign acquired 775 of 782 explicitly
authorized URLs. Seven remained fail-closed: five required authentication for
age gates and two were unavailable. No cookies, authentication, search, playlist
scraping, or bypass was used.

The research profile cataloged 775 files as 774 unique SHA-derived musical
identities, produced 1548 finite unit-normalized feature rows and 4229 segments,
and reported zero catalog or feature failures. The 200-song seed embeddings were
reused. Cold full-run feature work took 1780.92 internal seconds; the unchanged
rerun served every pre-normalization row from cache in 0.643 internal seconds and
1.55 seconds wall time. Peak total GPU memory was 9235 MiB, minimum available
host RAM was 7904.7 MiB, peak temperature was 56 C, and peak power was 122.84 W.
No OOM, NaN, paging, driver reset, or data loss was observed.

The original discovery accidentally ran the connected-components fallback
because the research environment omitted Leiden, retaining only one trivial
11-song single-artist group. The corrected run fails closed without the pinned
Leiden backend, calibrates each lens before applying the separation gate, adds
post-clustering artist/album dominance gates, and extends the resolution sweep.
It retained five diverse candidates containing 353 placements and 343 unique
songs (44.3% library coverage). The 774-song cached discovery took 212.64 wall
seconds, used about 1.16 GiB peak RSS with no swap, and reported `leiden` with no
fallback reason. Atomic HTML, JSON, CSV, and M3U8 exports each contain 353 rows.
The private song names and artifacts are not committed. This qualifies staged
local execution at this library size; it does not qualify lyric inference,
subjective playlist quality, private-label evaluation, blind human comparison,
or the fresh 10,000-song gate.
