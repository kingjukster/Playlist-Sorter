# Validated Playlist-Sorter Build Plan

Approved on 2026-09-15 for implementation in `kingjukster/Playlist-Sorter`.

## Implementation and qualification status

The combined foundation/features/discovery/review-export snapshot is implemented at
`08aad41cdbbd7a2804bf7f3ab2df936d9d9316a8`. Offline unit, contract, performance,
integration, and metamorphic checks exercise deterministic services and generated
fixtures. The Typer commands now orchestrate the descriptor-only automatic-discovery
MVP through canonical artifacts. Model weights/inference, private labels or libraries,
the 20/200 operational scripts on user data, the 10,000-song run, and blind human
preference gates remain explicitly unqualified. This document is the frozen validated
plan, not a claim that every planned runtime path is complete.

## Outcome and boundaries

Build a local-first Python 3.12 application that discovers useful,
stable, overlapping playlists from a private library of up to 10,000 songs.
Automatic zero-shot discovery is the MVP; optional guidance may use at most 19
unique songs per run. The implementation may install ordinary project
dependencies and run synthetic/redistributable fixtures, but this campaign must
not download model weights, scan the private library, run the full 10,000-song
qualification, write to streaming services, or modify source audio.

Original source is Apache-2.0. MuQ/MuQ-MuLan weights are CC-BY-NC 4.0 and the
default research profile is personal/non-commercial. Qwen3-Embedding is
Apache-2.0. Model weights are never redistributed.

## Runtime and safety

- Target Ubuntu 24.04 under WSL2 with Python 3.12, `uv`, `pyproject.toml`, and a
  committed `uv.lock`.
- Keep the Git repository on D:, but document/recommend WSL ext4 locations for
  the virtual environment, model cache, decoded intermediates, and embedding
  cache.
- Implement `playlist-sorter doctor` to report runtime readiness, including
  Python, CUDA PyTorch, pinned model revisions, FFmpeg, disk, and permissions.
  Ubuntu 24.04 under WSL2 is the target qualification environment; invoking the
  CLI on another platform does not qualify a live gate.
- Expensive work fails closed below 8 GB host-available RAM, 12 GB free VRAM, or
  required disk headroom, and when unsafe unrelated GPU contention is detected.
  Never terminate unrelated processes.
- Every run records exact command, timestamps, Git state, versions, throughput,
  RAM/VRAM, GPU utilization, temperature, power, failures, and a summary.

## Catalog and representations

- Read-only scan MP3, OGG, FLAC, WAV, and M4A. Use SHA-256 for exact files and
  Chromaprint plus duration/metadata checks for musical variants.
- Decode mono float32 at 24 kHz. After silence trimming, create 10-second windows
  centered at 10/30/50/70/90 percent plus one non-overlapping highest-RMS
  window; deduplicate overlaps and pad short tracks.
- Pin model revisions exactly:
  - OpenMuQ/MuQ-large-msd-iter@0562a57814f6f8bbd9fdea0a25921a2fce1a841a
  - OpenMuQ/MuQ-MuLan-large@2e01c796b71dca71b45251384c04cd7b237c9020
  - Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
- MuQ adapters default to FP32; Qwen3 defaults to BF16 inference mode.
- Lyrics use exact duplicate-line removal, 384-token chunks with 64-token
  overlap, and token-weighted spherical-mean pooling.
- Extract tempo, key, loudness, duration, dynamics, spectral, and rhythmic
  descriptors through replaceable Essentia/librosa adapters.
- Cache segment/pooled vectors and complete provenance. Unchanged songs must
  require no model inference on valid reruns.

## Multi-lens discovery

Build independent cosine k-NN graphs for MuQ-MuLan semantics, MuQ acoustic/MIR,
Qwen lyric semantics, explicit descriptors, and late-fused multimodal
similarity. The exact MVP uses blockwise GPU PyTorch cosine top-k with k=30 and
stability variants k=15 and 60. Default fusion weights are 0.35 semantic audio,
0.30 acoustic, 0.20 lyrics, and 0.15 descriptors, renormalized for missing
views. Artist and album are excluded from similarity and used only for leakage
diagnostics.

Run seeded Leiden separately for every lens over a logarithmic resolution sweep.
Overlap comes from combining stable candidates across lenses/resolutions. Use 12
bounded perturbation trials covering segment resampling, plus/minus 5 percent
edge weights, and fixed seeds. Align communities by maximum Jaccard.

Candidate size classes are broad (5-25 percent of library), medium (1-10
percent), and narrow (8 songs to 3 percent). Retain only candidates with mean
aligned Jaccard at least 0.75, core-member recurrence at least 80 percent,
calibrated within-versus-outside median margin at least 0.05, at least eight
songs, and no higher-ranked duplicate. Duplicate threshold is member Jaccard at
least 0.80 and centroid cosine at least 0.95 unless a meaningful parent/child
size ratio exists. Rank by 0.40 stability, 0.25 cohesion, 0.20 distinctiveness,
and 0.15 novelty. Return at most 40 candidates, targeting 8 broad, 16 medium,
and 16 narrow without filling failed quotas. Memberships overlap and expose a
`membership_score`, never probability/confidence.

## Guidance, review, export

- Guidance supports positive/negative seeds, must-link/cannot-link pairs, and
  optional names; at most 19 unique songs total, at least two positive seeds for
  named concepts, and fail-closed validation for duplicates, contradictions,
  unknown songs, or a twentieth unique example.
- Build positive/negative prototypes without backbone training. Guidance is a
  0.25-weight lens and unguided views retain 0.75. Must-links are at least 0.95
  and cannot-link edges are removed. Preserve unguided output.
- Implement a Streamlit 1.64 review app bound only to 127.0.0.1. It reads
  canonical artifacts and appends approve, reject, rename, split, merge,
  intrusion, and omission actions to `feedback.jsonl`. Business logic stays
  outside Streamlit.
- Export previewed, atomic JSON, CSV, and M3U8; never modify source media.

## Public interfaces and artifacts

CLI commands:

```
playlist-sorter doctor
playlist-sorter catalog scan --library <path> --output <run>
playlist-sorter features build --run <run> --profile descriptors
playlist-sorter discover --run <run> [--guidance <yaml>]
playlist-sorter review --run <run>
playlist-sorter export --run <run> --format json|csv|m3u8
playlist-sorter evaluate --run <run> --benchmark <path>
```

Versioned Pydantic contracts: SongRecord, AudioSegment, FeatureArtifact,
GuidanceSet, PlaylistCandidate, Membership, ReviewAction, DiscoveryRun, and
RunManifest. Unknown schema versions fail closed.

Canonical artifacts use Parquet for catalog/segments/memberships/metrics,
Safetensors plus a Parquet index for embeddings, JSON for configuration,
manifest/playlists/summaries, append-only JSONL for feedback, and CSV/M3U8 only
as derived exports. Git ignores private paths, lyrics, audio, caches, feedback,
weights, and run outputs.

## Verification and acceptance

- Unit/property tests cover identity, invalidation, missing modalities,
  normalization, fusion, overlap, constraints, abstention, and export
  idempotence. Integration covers scan-to-export with redistributable audio.
  Metamorphic tests cover renamed paths, duplicate encodes, missing lyrics,
  corrupt audio, reordered input, short tracks, and small perturbations.
- Synthetic gates: non-overlap ARI >=0.85, overlap pairwise F1 >=0.90, outlier
  F1 >=0.80, and fixed-seed repeated-run ARI >=0.99.
- Provide baseline/evaluation interfaces for metadata-only clustering,
  descriptor k-means, pooled MuQ-MuLan plus HDBSCAN, single-lens Leiden, fused
  Leiden without consensus, and the full system.
- Private historical labels are evaluation-only with artist-disjoint
  selection/evaluation. They cannot tune representations or discovery.
- Qualification gates (implemented and documented, but not claimed without the
  private/full runs): full-system macro NDCG@20 +5 percent relative over the
  strongest baseline; Precision@20 regression on at most one historical
  playlist; at least 60 percent preference in 100 blind comparisons with Wilson
  lower bound above 50 percent; guided held-out NDCG@20 +5 percent with mean
  stability drop no more than 0.03.
- Performance gates: 20- and 200-song smoke tests precede a fresh 10,000-song
  run; peak VRAM below 90 percent and host RAM below 85 percent; unchanged rerun
  skips at least 99 percent of embedding work and is at least 5x faster;
  approximate k-NN is allowed only at recall@30 >=0.98 and >=50 percent measured
  speedup.

Implementation is complete only when safe synthetic/redistributable checks pass,
the code/docs expose all qualification gates, and unrun private/model/full-scale
gates are explicitly reported as unqualified rather than implied to pass.
