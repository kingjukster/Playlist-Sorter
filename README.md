# Playlist Sorter

Playlist Sorter is a local-first Python 3.12 foundation for discovering stable,
overlapping playlists without modifying source media. The repository implements
read-only cataloging, versioned contracts, deterministic descriptors and graph
utilities, guidance policy, evaluation gates, review artifacts, and atomic
exports. The CLI now runs a complete descriptor-lens scan-to-export pipeline.
Pinned MuQ/MuLan/Qwen model execution remains an optional, unqualified research
extension until it is run under the approved WSL2/CUDA resource gates.

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
playlist-sorter catalog scan --library <path> --output <run>
playlist-sorter features build --run <run> --profile research
playlist-sorter discover --run <run> [--guidance <yaml>]
playlist-sorter review --run <run>
playlist-sorter export --run <run> --format json|csv|m3u8
playlist-sorter evaluate --run <run> --benchmark <path>
```

Each command invokes the corresponding service. `review` launches Streamlit on
`127.0.0.1` only. `export` writes to `<run>/exports/playlists.<format>` by default
or an explicit `--output`, after producing the exact preview bytes. Offline tests
do not launch a persistent server or download model weights.

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

## Privacy, models, and qualification

- Source audio, private paths, lyrics, feedback, caches, weights, and run outputs
  stay local and are ignored by Git.
- OpenMuQ/MuQ weights are CC-BY-NC 4.0 and belong only to the non-commercial
  research profile; Qwen3-Embedding code/weights are Apache-2.0 as documented.
- Model revisions are pinned in configuration, but no weights are redistributed
  or downloaded by this stage.
- Synthetic tests do not qualify private historical labels, model-backed quality,
  a 10,000-song run, or the required 100 blind human comparisons.

See [current status](docs/STATUS.md), the
[requirements matrix](docs/REQUIREMENTS_MATRIX.md), the
[validated plan](docs/IMPLEMENTATION_PLAN.md), and
[telemetry guidance](docs/TELEMETRY.md).
