# Playlist Sorter

Playlist Sorter is a local-first Python 3.12 foundation for discovering stable,
overlapping playlists without modifying source media. The repository implements
read-only cataloging, versioned contracts, deterministic descriptors and graph
utilities, guidance policy, evaluation gates, review artifacts, and atomic
exports. Model-backed feature orchestration and a complete scan-to-export CLI
pipeline are not implemented or qualified.

## What works now

The shared service APIs are the current integration surface:

```text
audio files -> catalog.scan_library -> deterministic feature/graph/discovery helpers
            -> canonical run JSON -> review/load services -> previewed atomic export
```

The redistributable integration test generates WAV bytes, scans them with
`scan_library`, builds versioned canonical artifacts, loads them through the
export service, and verifies deterministic export without changing the audio.
There is no service that automatically connects all of those steps yet.

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

Only `doctor` invokes a runtime service. The other advertised commands currently
emit validated JSON shell records and do **not** scan, build features, discover,
launch Streamlit, export files, or evaluate a run. Use the Python services and
tests for implemented behavior; do not treat a successful shell response as a
completed pipeline stage. Review server helpers are fixed to `127.0.0.1`, but no
persistent server is launched by the offline checks.

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
