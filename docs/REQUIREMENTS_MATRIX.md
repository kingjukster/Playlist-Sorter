# Frozen requirements matrix

Statuses are deliberately distinct:

- **Implemented** means a code/interface exists.
- **Synthetic-validated** means offline tests passed against generated or
  redistributable inputs.
- **Unqualified** means the required private, model-backed, full-scale, or human
  evidence was not authorized or produced.

The campaign grouped the frozen plan into requirements `r1` through `r10` as
shown below. Passing a lower-cost check never upgrades an unrun gate.

| ID | Frozen requirement | Code and artifacts | Test/evidence | Status and remaining gate |
|---|---|---|---|---|
| r1 | Python 3.12/WSL2 package, lazy CLI, fail-closed doctor and resource safety | `pyproject.toml`, `src/playlist_sorter/cli.py`, `core/runtime.py` | `tests/unit/foundation/test_runtime.py`, CLI help smoke | Implemented and synthetic-validated. Installed WSL/GPU/model environment remains unqualified. |
| r2 | Versioned public contracts and canonical artifact conventions | `core/contracts.py`, `core/artifacts.py` | `tests/contracts/test_contracts.py` | Implemented and synthetic-validated. No complete producer writes every planned Parquet/Safetensors artifact. |
| r3 | Read-only catalog, content identity, decode/segment behavior, corrupt/short input handling | `catalog/scan.py`, feature adapters | feature unit tests; catalog metamorphic and variant tests | Implemented and synthetic-validated for generated WAV and adapter fixtures, including Chromaprint-plus-duration variant grouping. Private formats/library remain unqualified. |
| r4 | Descriptors, lyrics handling, pinned lazy model adapters, provenance cache and no-inference reruns | `features/`, `configs/models.yaml` | `tests/unit/features/test_feature_pipeline.py` | Deterministic adapters/cache are implemented and synthetic-validated. Actual MuQ/MuLan/Qwen inference, numerical equivalence, and weight licenses in use remain unqualified. |
| r5 | Separate-lens exact top-k graphs, missing-view fusion, leakage diagnostics, multi-resolution consensus and overlap | `graph/`, `discovery/` | discovery unit tests | Implemented and synthetic-validated, including optional Torch checks from the accepted component stage. Full-library graph quality remains unqualified. |
| r6 | Guidance limited to 19 unique songs total, constraints, 0.25 guided/0.75 unguided comparison | `guidance/policy.py` | guidance and contract tests; `examples/guidance.example.yaml` | Implemented and synthetic-validated. Guided private held-out NDCG/stability gate remains unqualified. |
| r7 | Deterministic naming, localhost-only review, provenance feedback, previewed atomic JSON/CSV/M3U8 exports | `naming/`, `review/`, `review_app.py`, `export/service.py` | review/export unit tests and generated scan-to-export integration | Implemented and synthetic-validated. CLI review/export invoke the real services; no persistent server was launched during validation. |
| r8 | Local-first privacy, ignore rules, pinned model/license boundaries, no source mutation or external writes | `.gitignore`, `LICENSE`, `configs/models.yaml`, runtime registry | source-byte invariance and repository diff inspection | Implemented policy and synthetic source-mutation checks. Commercial model profile and streaming writes are out of scope/unqualified. |
| r9 | Synthetic quality gates, baselines, private-label separation, resource/cache/ANN/full qualification gates | `evaluation/`, `tests/performance/test_evaluation_gates.py` | synthetic ARI/F1 and deterministic gate tests | Gate logic is implemented and synthetic-validated. Private NDCG, 100 blind comparisons, 20/200 operational runs, fresh 10,000-song run, RAM/VRAM/cache-speed, and ANN performance remain unqualified. |
| r10 | Redistributable integration/metamorphic coverage, safe operational scripts, accurate docs and final combined checks | `pipeline/`, `tests/integration/`, `tests/metamorphic/`, `scripts/`, `README.md`, `docs/` | real scan-to-export integration and final combined checks | Implemented and synthetic-validated for the descriptor MVP. Model-backed and private/full-scale qualification remains open. |

## CLI/service boundary

All advertised commands now call shared services. The generated-audio integration
test uses those same services and validates the canonical boundary. Review stays
localhost-only, exports are derived writes, and source audio remains read-only.

## Frozen numeric gates

- Synthetic: non-overlap ARI at least `0.85`, overlap pairwise F1 at least
  `0.90`, outlier F1 at least `0.80`, repeated-seed ARI at least `0.99`.
- Candidates: aligned Jaccard at least `0.75`, core recurrence at least `0.80`,
  margin at least `0.05`, at least 8 songs, no higher-ranked duplicate.
- Approximate search: recall@30 at least `0.98` and at least 50% measured speedup.
- Resources/cache: peak VRAM below 90%, host RAM below 85%, at least 99% embedding
  skips and 5x faster unchanged rerun.
- Private/human: full-system macro NDCG@20 at least 5% relative over the strongest
  baseline; Precision@20 regression on at most one playlist; at least 60% wins in
  100 blind comparisons with Wilson lower bound above 50%; guided NDCG@20 at least
  5% with mean stability drop no more than `0.03`.

The last two groups remain unqualified until their exact authorized runs occur.
