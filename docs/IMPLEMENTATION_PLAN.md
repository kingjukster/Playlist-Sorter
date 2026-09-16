# Playlist Sorter implementation plan

- Status: proposed for approval
- Repository: `kingjukster/Playlist-Sorter`
- Primary runtime: local Windows 11 with NVIDIA RTX 4090 (24 GB VRAM)
- Implementation state: planning only

## 1. Outcome

Build a local-first application that scans a user-controlled music library and
automatically discovers coherent, useful, overlapping playlists. It must work
without predefined labels and must improve when the user supplies fewer than
20 positive, negative, must-link, or cannot-link examples.

The system must preserve raw media, cache expensive derived artifacts, explain
why songs were grouped, and export deterministic playlist manifests without
writing to a streaming service in the MVP.

## 2. Product principles

1. **Discovery first.** The default workflow discovers latent playlist ideas;
   it does not require the user to name categories in advance.
2. **Multiple valid views.** Genre, mood, energy, lyrical theme, era, and
   production style can all produce useful but different playlists.
3. **Overlap is expected.** A song may belong to several playlists with
   independent membership scores.
4. **Abstention is valid.** Weak or unstable assignments remain unassigned.
5. **Local and inspectable.** Audio, lyrics, embeddings, and feedback remain on
   the local machine unless a future integration is explicitly enabled.
6. **Measured quality.** Model or clustering changes are accepted only when
   they improve predefined evaluation and preference metrics.
7. **Measured performance.** Runtime, throughput, RAM, VRAM, GPU utilization,
   and cache effectiveness are recorded instead of estimated.
8. **Replaceable components.** Model, feature, graph, clustering, naming, and
   export backends use narrow interfaces so licenses or hardware constraints do
   not lock the product to one provider.

## 3. MVP scope

### Included

- Scan local audio files and ingest CSV/JSON manifests.
- Read tags without modifying source files.
- Normalize stable song identities and detect duplicates.
- Segment tracks and extract cached musical-audio embeddings.
- Embed available lyrics and metadata independently.
- Extract explicit tempo, key, loudness, duration, and related descriptors.
- Construct a fused multi-view nearest-neighbor graph.
- Discover communities at broad, medium, and narrow resolutions.
- Select stable communities through repeated-run consensus.
- Produce soft, overlapping memberships and an unassigned state.
- Accept fewer than 20 examples per guidance set.
- Generate transparent playlist names and explanations.
- Export JSON, CSV, and M3U8 manifests.
- Provide a CLI and machine-readable run report.

### Excluded from MVP

- Modifying, moving, deleting, or retagging source audio.
- Writing directly to Spotify, Apple Music, YouTube Music, or another service.
- Collaborative filtering from other users' behavior.
- Training or fine-tuning a large audio or language-model backbone.
- Cloud inference or upload of private music and lyrics.
- A production web application or mobile application.
- Claims of commercial readiness while default research weights have
  non-commercial license terms.

## 4. Core user workflows

### Zero-shot discovery

```powershell
playlist-sorter discover `
  --library "D:\Music" `
  --profile balanced `
  --output ".\runs\library-v1"
```

Expected outputs:

- discovered playlists at multiple granularities;
- per-song memberships and confidence;
- representative and borderline songs;
- explanations and naming evidence;
- unassigned songs and exclusion reasons;
- run configuration, model revisions, timings, telemetry, and quality metrics.

### Few-shot guidance

```powershell
playlist-sorter discover `
  --library "D:\Music" `
  --guidance ".\examples\guidance.yaml" `
  --output ".\runs\guided-v1"
```

Guidance can express:

- songs that should be together;
- songs that should be separated;
- positive seeds for a desired concept;
- negative examples that define a boundary;
- optional human playlist-name hints.

Every guidance group is rejected with a clear validation error when it exceeds
19 examples.

## 5. Data contracts

### Song record

Each normalized song record contains:

- `song_id`: content-derived stable identifier;
- `source_path`: local path retained only in private local artifacts;
- title, artist, album, track number, date, and genre tags when present;
- duration, codec, sample rate, and channel count;
- lyrics plus provenance and language when supplied;
- optional external identifiers supplied by the user;
- file size, modification time, and an integrity fingerprint;
- preprocessing state and error fields.

The content identifier must not depend only on a mutable filename. The design
will distinguish exact duplicates, alternate encodes, remasters, live versions,
and truly different tracks.

### Embedding artifact

Every embedding artifact records:

- song and segment identifiers;
- modality and feature-view name;
- model repository and immutable revision;
- preprocessing configuration and code version;
- vector dtype, dimensions, and normalization;
- source fingerprint;
- creation timestamp and elapsed time;
- cache validity status.

### Playlist candidate

A candidate records:

- stable candidate identifier;
- granularity and discovery resolution;
- member songs with soft scores;
- representative, core, boundary, and excluded songs;
- stability, cohesion, separation, novelty, and coverage metrics;
- naming evidence and generated labels;
- lineage to parent, child, overlapping, or competing communities;
- any few-shot constraints that affected it.

Use versioned Pydantic schemas and reject unknown incompatible schema versions.

## 6. Technical architecture

```text
local audio / manifest / lyrics
              |
              v
      immutable catalog index
              |
       +------+-------+----------------+
       |              |                |
       v              v                v
 audio segments   lyric chunks    metadata/MIR features
       |              |                |
       v              v                v
 audio embeddings text embeddings  numeric descriptors
       +--------------+----------------+
                      |
                      v
          per-view normalized k-NN graphs
                      |
                      v
        calibrated late-fusion similarity graph
                      |
                      v
      multi-resolution community discovery
                      |
                      v
      perturbation and bootstrap consensus
                      |
                      v
 soft overlap + abstention + hierarchy/lineage
                      |
                      v
       naming, explanations, evaluation, export
```

### 6.1 Catalog and preprocessing

- Scan incrementally and never alter source media.
- Decode to a temporary or streamed normalized waveform.
- Detect silence and invalid/corrupt inputs.
- Segment by musical structure when reliable; otherwise use deterministic,
  overlapping windows representing the intro, early body, middle/high-energy
  region, late body, and outro.
- Bound queues so decoding cannot exhaust the 32 GB system RAM.
- Cache by content fingerprint plus preprocessing version.
- Preserve failures as structured records so one bad file does not abort a run.

### 6.2 Representation views

The default research profile will evaluate:

1. `OpenMuQ/MuQ-large-msd-iter` for music-specific acoustic representation.
2. `OpenMuQ/MuQ-MuLan-large` for joint audio-text semantics.
3. `Qwen/Qwen3-Embedding-0.6B` for lyrics and normalized metadata text.
4. Deterministic musical descriptors for tempo, key, loudness, duration,
   dynamics, and spectral/rhythmic characteristics.

Backbones remain frozen. Run audio and text models sequentially by default to
leave VRAM headroom. MuQ inference begins in the precision recommended by its
upstream implementation; lower precision is adopted only after a numerical
equivalence and NaN-free smoke test.

Lyrics are chunked by structural markers or bounded token windows. Store chunk
embeddings, pooled song embeddings, and pooling metadata. Missing lyrics must
not produce zero vectors that masquerade as meaningful evidence.

### 6.3 Similarity and late fusion

- L2-normalize embeddings within each view.
- Build a top-k neighbor graph independently for each view.
- Retain per-view similarity rather than concatenating unlike vectors.
- Calibrate each view's score distribution before fusion.
- Fuse edges using configurable weights and missing-view masks.
- Prevent artist, album, or duplicate leakage from dominating communities.
- Store the contribution of every view to every retained edge.

Initial weights are explicit configuration, not hidden constants. Few-shot
feedback may fit a strongly regularized weight adjustment, but it may not
fine-tune a backbone in the MVP.

### 6.4 Community discovery

- Run Leiden over a documented resolution sweep to discover broad, medium, and
  narrow communities.
- Use deterministic seeds where supported and record all seeds.
- Run HDBSCAN as an independent baseline and outlier signal.
- Do not cluster on a two-dimensional visualization projection.
- Preserve parent/child and overlap relationships among retained communities.
- Avoid forced assignment when a song has no strong, stable membership.

### 6.5 Stability consensus

Perturb one factor at a time across bounded trials:

- segment sampling;
- graph neighborhood size;
- fusion weights;
- clustering resolution;
- supported random seeds.

Align communities across trials and compute member-level and
community-level stability. A playlist is publishable only if it passes minimum
stability, size, cohesion, and distinctiveness gates. Near-duplicates are merged
or ranked as alternatives rather than exported as separate playlists.

### 6.6 Overlap and abstention

Convert graph/community evidence into calibrated membership scores. Permit a
song to join multiple communities when each score independently clears its
threshold. Record reasons for abstention, including weak similarity,
instability, insufficient view coverage, or conflict with guidance.

### 6.7 Few-shot guidance

- Validate a maximum of 19 examples per guidance group.
- Represent positive and negative prototypes with uncertainty.
- Apply must-link and cannot-link constraints as transparent graph adjustments.
- Use graph propagation or constrained assignment after unguided discovery.
- Keep unguided results available for before/after comparison.
- Use cross-validation or leave-one-out evaluation when enough examples exist.
- Reject learned adjustments that improve seed fit but damage held-out quality
  or global stability.

### 6.8 Naming and explanations

The deterministic naming layer first summarizes:

- dominant audio-text tags;
- lyrical themes;
- musical descriptors;
- representative songs and artists;
- contrast against nearest competing communities.

An optional music-language model may refine names for representative songs only.
It is never required for discovery, and its output is stored as generated text
with model provenance. A name cannot change playlist membership.

### 6.9 Export

- JSON is the canonical, lossless result.
- CSV provides song-membership tables.
- M3U8 uses normalized absolute or configured relative paths.
- Export is preview-first and atomic.
- Re-running export does not duplicate or alter source media.
- Streaming-service adapters, if later approved, consume canonical manifests
  and require a separate explicit write step.

## 7. Licensing and privacy boundaries

- Do not commit model weights, copyrighted audio, private lyrics, caches, or
  generated library manifests.
- Add explicit ignore rules before any data-bearing command is introduced.
- Pin model revisions and record each upstream license.
- MuQ and MuQ-MuLan released weights are CC-BY-NC 4.0; therefore they belong in
  a clearly labeled non-commercial research profile.
- Keep model adapters replaceable so a commercially compatible profile can be
  qualified separately.
- Never load executable remote model code without a pinned revision and an
  explicit trust decision documented in the model registry.
- Treat lyrics and local path information as private by default.
- Telemetry must contain resource measurements, not song content.

## 8. Proposed repository structure

```text
Playlist-Sorter/
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- .gitignore
|-- configs/
|   |-- profiles/
|   `-- models.yaml
|-- docs/
|   |-- IMPLEMENTATION_PLAN.md
|   |-- ARCHITECTURE.md
|   |-- DATA_CONTRACTS.md
|   |-- EVALUATION.md
|   `-- MODEL_LICENSES.md
|-- examples/
|   |-- library.example.csv
|   `-- guidance.example.yaml
|-- src/playlist_sorter/
|   |-- catalog/
|   |-- audio/
|   |-- embeddings/
|   |-- features/
|   |-- graph/
|   |-- discovery/
|   |-- guidance/
|   |-- naming/
|   |-- evaluation/
|   |-- export/
|   |-- telemetry/
|   `-- cli.py
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- contracts/
|   |-- performance/
|   `-- fixtures/
`-- scripts/
    |-- smoke.ps1
    `-- benchmark.ps1
```

## 9. Implementation stages and gates

Each stage is independently reviewable. A later stage does not begin until its
dependency gates pass.

### Stage 0: repository and safety foundation

Deliverables:

- Python package skeleton and pinned supported interpreter range;
- linting, formatting, type checking, tests, and CI;
- ignore rules for media, lyrics, credentials, caches, models, and runs;
- configuration loader and model/license registry;
- run manifest, structured logging, and artifact directory conventions.

Acceptance:

- clean installation in a fresh environment;
- all checks pass locally and in CI;
- a fixture run cannot commit private or oversized artifacts accidentally;
- every command has `--help` and dry-run behavior where writes occur.

### Stage 1: catalog and immutable ingestion

Deliverables:

- local scanner and manifest importer;
- metadata extraction and stable song identity;
- duplicate/variant handling;
- versioned catalog schema and failure ledger.

Acceptance:

- incremental rescan changes only affected records;
- source files remain byte-identical;
- corrupt and unsupported files are reported without aborting the scan;
- catalog output is deterministic for fixed inputs.

### Stage 2: audio segmentation and deterministic descriptors

Deliverables:

- decoding and resampling pipeline;
- bounded deterministic segmentation;
- explicit musical descriptors;
- content-addressed cache.

Acceptance:

- segment boundaries are reproducible;
- cache hit/miss behavior is covered by tests;
- invalid audio and very short tracks have defined behavior;
- smoke-run timing, throughput, RAM, and errors are recorded.

### Stage 3: multimodal embedding pipeline

Deliverables:

- model adapter interface;
- MuQ, MuQ-MuLan, and Qwen3 embedding adapters;
- batching, cache invalidation, and missing-modality handling;
- model revision and license manifest.

Acceptance:

- a small end-to-end fixture completes without NaNs or OOM;
- reruns reuse valid embeddings and avoid model inference;
- embedding dimensions, norms, and fingerprints are validated;
- exact commands, load time, songs/second, peak VRAM, and wall time are saved;
- model outputs are stable within documented numerical tolerance.

### Stage 4: multi-view graph construction

Deliverables:

- per-view nearest-neighbor indices;
- score calibration and late fusion;
- edge provenance and graph serialization;
- artist/album/duplicate leakage controls.

Acceptance:

- synthetic neighborhoods recover known relationships;
- missing views do not bias scores as zero evidence;
- every fused edge can be decomposed into view contributions;
- exact and approximate neighbor search are compared before choosing defaults.

### Stage 5: automatic discovery and overlap

Deliverables:

- Leiden resolution sweep;
- HDBSCAN baseline and outlier signal;
- community alignment and stability consensus;
- hierarchy, overlap, soft membership, and abstention.

Acceptance:

- planted synthetic communities are recovered above a defined adjusted mutual
  information threshold;
- unstable candidates are filtered;
- output is reproducible for fixed configuration and seeds;
- no song is forced into a playlist below threshold;
- duplicate candidate playlists are detected and resolved.

### Stage 6: few-shot guidance

Deliverables:

- guidance schema and validator;
- positive/negative prototypes;
- must-link/cannot-link constraints;
- regularized view-weight adaptation and graph propagation;
- guided-versus-unguided comparison report.

Acceptance:

- 20 or more examples in one guidance group fails validation;
- contradictory constraints fail with actionable diagnostics;
- seed memorization alone cannot satisfy the quality gate;
- held-out agreement improves without unacceptable global stability loss.

### Stage 7: naming, explanations, and exports

Deliverables:

- deterministic cluster summaries and names;
- optional pluggable language-model refinement;
- representative/borderline evidence;
- JSON, CSV, and M3U8 exporters with preview.

Acceptance:

- membership is invariant to naming backend;
- every playlist and song membership has inspectable evidence;
- exports are atomic, deterministic, and round-trip validated;
- no source files are changed.

### Stage 8: integrated quality and performance qualification

Deliverables:

- gold and synthetic evaluation sets;
- ablation study by representation view;
- zero-shot and 1/3/5/10/19-shot comparisons;
- full telemetry and cache benchmarks;
- human preference review workflow;
- documented recommended profile and fallback profile.

Acceptance:

- the recommended system beats metadata-only, single-embedding HDBSCAN, and
  single-view Leiden baselines on the agreed quality scorecard;
- gains survive repeated seeds and bootstrap samples;
- cache reuse materially reduces rerun wall time;
- the full target-library run finishes without OOM, paging, data loss, or
  unhandled failures;
- limitations and unqualified use cases are documented.

## 10. Evaluation protocol

No single intrinsic clustering metric decides success.

### Test sets

1. Synthetic songs/embeddings with planted hierarchy, overlap, outliers, and
   missing modalities.
2. A small redistributable audio fixture for CI.
3. A private local evaluation set with user-confirmed playlist relationships.
4. Existing playlists used only as comparison labels, not assumed ground truth.

### Baselines

- metadata-only clustering;
- explicit descriptors plus k-means;
- single pooled embedding plus HDBSCAN;
- single-view graph plus Leiden;
- full multi-view graph without consensus;
- full system.

### Metrics

- adjusted mutual information and adjusted Rand index where labels exist;
- pairwise precision, recall, and F1 for overlapping relationships;
- constraint satisfaction on held-out guidance;
- community stability under perturbation;
- cohesion and separation in each original representation view;
- coverage and abstention rate;
- candidate redundancy and playlist-size distribution;
- human pairwise preference and edit distance to accepted playlists;
- songs/second, wall time, peak RAM/VRAM, and cache hit rate.

### Human acceptance

The review UI or report must hide method names during comparison. The user
chooses between candidate playlists, marks intruders and omissions, and may
approve, rename, split, merge, or reject a community. Those actions are stored
as provenance-rich feedback, never silently converted into training authority.

## 11. Performance and telemetry plan

The first expensive run at each stage begins with a representative smoke test.
The full run is blocked if the smoke test has OOM, NaNs, driver errors,
corruption, or invalid artifacts.

For every expensive command, save:

- exact command and resolved configuration;
- Git commit and dirty-state indicator;
- model names, revisions, precision, and licenses;
- input and output paths plus content fingerprints;
- start/end timestamps and wall time;
- songs, segments, audio minutes, and tokens processed per second;
- model load and cache time;
- average/peak CPU and GPU utilization when available;
- peak RAM and VRAM;
- GPU temperature, power, and throttle state when available;
- failures, retries, skipped records, and final status;
- a short `run_summary.md`.

Optimization sequence:

1. Establish a correct CPU/GPU baseline.
2. Tune batch size without OOM-driven search.
3. Tune decode workers and prefetch against GPU idle time.
4. Enable pinned memory and asynchronous transfers where measured useful.
5. Compare precision changes only with numerical and quality validation.
6. Stop tuning when improvement is below approximately 3% or stability falls.

The likely initial constraint is the CPU/RAM audio-decode pipeline feeding the
GPU. Keep intermediate queues bounded and avoid nested thread oversubscription.

## 12. Test strategy

- **Unit:** schemas, fingerprints, normalization, score calibration, fusion,
  constraints, membership thresholds, exporters.
- **Property-based:** idempotence, ordering invariance, bounded scores, cache
  invalidation, and no mutation of inputs.
- **Contract:** model adapters, artifact schemas, and CLI JSON output.
- **Integration:** scan through export on tiny audio fixtures.
- **Golden:** versioned expected graphs and manifests where determinism applies.
- **Metamorphic:** duplicate insertion, tag removal, lyric removal, path rename,
  and minor audio perturbation.
- **Performance:** smoke and benchmark profiles with non-regression thresholds.
- **Privacy/safety:** secret scanning, ignored-artifact checks, path redaction,
  malformed manifest handling, and unsafe model-loading rejection.

## 13. Quality gates

A stage is complete only when:

- required tests and static checks pass;
- outputs are schema-valid and provenance-complete;
- no raw/private artifact is committed;
- exact validation commands and outcomes are recorded;
- new dependencies and model weights have license review;
- performance is measured for expensive changes;
- failure and recovery paths are tested;
- documentation matches implemented behavior.

Passing source tests does not qualify model quality. Passing model evaluation
does not qualify streaming-service writes, commercial use, or deployment.

## 14. Major risks and mitigations

| Risk | Mitigation |
|---|---|
| Communities reflect artist/album identity instead of playlist character | Leakage controls, per-view ablations, artist-disjoint evaluation |
| One embedding dominates fusion | Distribution calibration, explicit weights, edge provenance |
| Full-song pooling erases structure | Segment embeddings and stability across segments |
| Clusters are mathematically clean but not useful | Blind human preference review and edit-cost metrics |
| Too many near-duplicate playlists | Candidate similarity, lineage, merge/rank policy |
| Few-shot examples overfit | Frozen backbones, strong regularization, held-out checks |
| Missing lyrics or tags distort similarity | Missing-view masks and modality-specific confidence |
| Non-commercial model licenses block later use | Research/commercial profiles and replaceable adapters |
| Windows dependency friction | Pinned environment, startup diagnostics, CPU fallback tests |
| Audio decoding exhausts RAM or starves GPU | Bounded queues, measured workers, cached preprocessing |
| GPU acceleration changes clustering results | Record backend, compare quality, preserve deterministic reference path |

## 15. Definition of MVP done

The MVP is done when a clean checkout can, using documented commands:

1. scan a local test library without altering it;
2. generate or reuse provenance-complete multimodal features;
3. discover stable broad, medium, and narrow overlapping playlists;
4. abstain on uncertain songs;
5. accept valid guidance sets of at most 19 examples;
6. show why each playlist and membership exists;
7. export validated JSON, CSV, and M3U8 artifacts;
8. beat the defined baseline systems on the agreed scorecard;
9. complete the qualified target-library run with saved timing and resource
   telemetry and without OOM or data loss;
10. document model licenses, privacy boundaries, limitations, and recovery.

## 16. Approval boundary

Approval of this plan authorizes implementation only when the user explicitly
approves it. Planning does not authorize downloading model weights, scanning a
private music library, running long GPU jobs, installing dependencies, or
writing to external music services.
