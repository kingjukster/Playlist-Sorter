# Playlist Sorter

Playlist Sorter is a local-first Python 3.12 foundation for discovering stable,
overlapping playlists without modifying source media. The repository implements
read-only cataloging, versioned contracts, deterministic descriptors and graph
utilities, guidance policy, evaluation gates, review artifacts, and atomic
exports. The CLI now runs a complete descriptor-lens scan-to-export pipeline.
Pinned MuQ/MuLan/Qwen execution is an optional, local-snapshot-only `research`
profile. It remains unqualified until it completes the approved WSL2/CUDA
resource gates on an explicitly authorized private subset.

## What works now

The shared service APIs are the current integration surface:

```text
audio files -> read-only catalog -> cached descriptors -> stable discovery
            -> canonical run artifacts -> local review -> atomic derived export
```

The redistributable integration test generates WAV bytes and exercises the real
scan, feature, discovery, canonical-artifact, and export services without
changing the audio. Discovery may intentionally abstain when fewer than eight
songs or insufficiently stable evidence are available.

The Typer interface exposes the frozen command shape:

```bash
playlist-sorter doctor --path <output-root>
playlist-sorter acquire youtube --manifest <yaml|json|csv> --output <directory>
playlist-sorter catalog scan --library <path> --output <run>
playlist-sorter features build --run <run> --profile descriptors
playlist-sorter discover --run <run> [--guidance <yaml>]
playlist-sorter review --run <run>
playlist-sorter export --run <run> --format json|csv|m3u8
playlist-sorter evaluate --run <run> --benchmark <path>
```

Each command invokes the corresponding service. `review` launches Streamlit on
`127.0.0.1` only. `export` writes to `<run>/exports/playlists.<format>` by default
or an explicit `--output`, after producing the exact preview bytes. Offline tests
do not launch a persistent server or download model weights.

### Feature profiles and cache behavior

`--profile` accepts exactly two lowercase values; unknown, spaced, or
case-normalized values fail closed.

- `descriptors` is the runnable offline compatibility profile. It writes only
  the descriptor view and never creates or loads a model adapter. Reruns reuse
  matching source/preprocessing rows in that run's `features.json`.
- `research` requests MuQ acoustic, MuQ-MuLan semantic-audio, and (when
  nonblank lyrics exist) Qwen lyrics views. It never falls back to descriptors.
  It requires the optional `research` dependencies and exact model snapshots
  already present locally; adapters use `local_files_only=True`. Missing lyrics
  abstain rather than invent a vector.

Research entries are content/provenance-addressed under that run's
`embedding_cache`: source SHA-256, repository and revision, preprocessing,
pooling, view, and segment identity determine the cache key. A cache hit skips
adapter inference. The default pipeline cache is JSON; the separately available
Safetensors/Parquet backend requires NumPy, PyArrow, and Safetensors.

## Authorized YouTube acquisition

The optional acquisition command accepts only explicit individual YouTube URLs
that you are authorized to download. Every item must set `rights_basis`, explain
the basis in `rights_note`, and set `authorization_confirmed: true`. Playlist or
mix URLs, non-YouTube hosts, unknown fields, and missing declarations fail
closed. The wrapper disables inherited yt-dlp configuration, playlists, and
remote components; it never searches, imports cookies, logs in, or bypasses DRM
or geographic restrictions.

Copy `examples/youtube_manifest.example.yaml` outside the repository, fill it
with authorized URLs, then run under WSL2 after installing FFmpeg:

```bash
uv run playlist-sorter acquire youtube \
  --manifest "$HOME/private/muy-fuego-authorized.yaml" \
  --output "$HOME/.local/share/playlist-sorter/acquired/muy-fuego" \
  --audio-format flac
```

The output contains `media/`, the yt-dlp archive, resolved manifest, exact
commands, append-only receipts, SHA-256 values, and JSON/Markdown run summaries.
An unchanged verified rerun skips completed files. Feed the resulting `media/`
directory to `catalog scan`. Acquisition outputs and private manifests must stay
outside Git.

The policy and synthetic downloader boundary are tested, but no live acquisition
was run in this campaign; it remains unqualified.

## Exact offline quickstart

Use Ubuntu 24.04 on WSL2 and keep the repository on the Windows drive if desired.
Put the virtual environment and caches on WSL ext4 to avoid `/mnt/<drive>` metadata
overhead:

```bash
sudo apt-get install -y python3.12 python3.12-venv ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /mnt/c/Users/kingj/projects/b9dd/Playlist-Sorter
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/playlist-sorter/venv"
export UV_CACHE_DIR="$HOME/.cache/playlist-sorter/uv"
export XDG_CACHE_HOME="$HOME/.cache/playlist-sorter"
uv sync --frozen
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run playlist-sorter --help
```

From Windows PowerShell, `scripts/smoke.ps1` accepts only an explicit directory
containing exactly 20 supported audio files. `scripts/benchmark.ps1` requires a
successful smoke receipt and a separate explicit directory containing exactly
200 files. Both enter WSL2, invoke Python 3.12, save command/stdout/stderr/timing
and resource telemetry, and write `run_summary.md` under the chosen output. They
perform catalog-only gates: no model inference, model download, library mutation,
or persistent server.

```powershell
.\scripts\smoke.ps1 -Library <exact-20-file-directory> -Output <private-output>
.\scripts\benchmark.ps1 -Library <exact-200-file-directory> -SmokeReceipt <receipt> -Output <private-output>
```

These are catalog-only WSL2 wrapper gates, not model-backed inference. The
offline suite validates generated/synthetic inputs; it does not scan private
media, acquire videos, download/load/infer models, or qualify 20/200/10,000-song
operational results.

### Model-backed qualification sequence

Before any model load, inference, or download, save a fresh machine preflight
with RAM, VRAM, GPU utilization/temperature/power, and competing-process state.
Only with an explicitly authorized private subset and practical local snapshots
may the one-song smoke run; proceed to at most 20 songs only if it has no NaN,
OOM, driver reset, or resource-gate failure. Retain the command, timestamps,
model revisions, cache evidence, throughput, and telemetry in a private output.
Never substitute the 98-song library for this gate.

## Privacy, models, and qualification

- Source audio, private paths, lyrics, feedback, caches, weights, and run outputs
  stay local and are ignored by Git.
- OpenMuQ/MuQ weights are CC-BY-NC 4.0 and belong only to non-commercial
  research. The pinned identifiers are
  `OpenMuQ/MuQ-large-msd-iter@0562a57814f6f8bbd9fdea0a25921a2fce1a841a`,
  `OpenMuQ/MuQ-MuLan-large@2e01c796b71dca71b45251384c04cd7b237c9020`, and
  `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
  `configs/license_registry.toml` records Qwen's declared Apache-2.0 boundary;
  users must verify and obey upstream terms before use.
- No weights are redistributed or downloaded by this stage. MuQ/MuLan use FP32;
  Qwen requests BF16 on supported CUDA and otherwise uses FP32.
- Synthetic tests do not qualify model download/load/inference, private historical
  labels, model-backed quality or numeric equivalence, private 1/20-song runs,
  a 10,000-song run, or the required 100 blind human comparisons. WSL2/Python
  3.12/CUDA is the live-model qualification environment; Windows scripts are
  catalog-only wrappers.

See [current status](docs/STATUS.md), the
[requirements matrix](docs/REQUIREMENTS_MATRIX.md), the
[validated plan](docs/IMPLEMENTATION_PLAN.md), and
[telemetry guidance](docs/TELEMETRY.md).
