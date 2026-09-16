from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from playlist_sorter.acquisition import (
    AcquisitionEntry,
    AcquisitionError,
    acquire_youtube_audio,
    build_ytdlp_command,
    load_acquisition_manifest,
)

URL = "https://www.youtube.com/watch?v=BaW_jenozKc"


def _write_manifest(path: Path, **overrides) -> Path:
    item = {
        "url": URL,
        "rights_basis": "permission",
        "rights_note": "Written permission retained by the user.",
        "authorization_confirmed": True,
        **overrides,
    }
    path.write_text(json.dumps({"schema_version": "1.0", "items": [item]}), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "overrides",
    [
        {"authorization_confirmed": False},
        {"rights_note": ""},
        {"url": "https://example.com/video"},
        {"url": f"{URL}&list=PL123456"},
    ],
)
def test_manifest_fails_closed_without_rights_or_for_non_individual_urls(tmp_path, overrides):
    manifest = _write_manifest(tmp_path / "manifest.json", **overrides)

    with pytest.raises(AcquisitionError):
        load_acquisition_manifest(manifest)


def test_command_disables_config_auth_adjacent_and_playlist_behavior(tmp_path):
    entry = AcquisitionEntry(
        url=URL,
        rights_basis="owned",
        rights_note="Original recording owned by the user.",
        authorization_confirmed=True,
    )

    command = build_ytdlp_command(entry, output_dir=tmp_path, audio_format="flac")

    assert "--no-config" in command
    assert "--no-playlist" in command
    assert "--no-remote-components" in command
    assert "--download-archive" in command
    assert "--extract-audio" in command
    assert "--cookies-from-browser" not in command
    assert "--username" not in command
    assert "ytsearch:" not in " ".join(command)
    assert command[-1] == URL


def test_acquisition_writes_receipt_and_verified_rerun_skips(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.json")
    output = tmp_path / "output"
    calls: list[tuple[str, ...]] = []

    def fake_runner(command, **_kwargs):
        calls.append(command)
        template = Path(command[command.index("--output") + 1])
        target = Path(str(template).replace("%(ext)s", "flac"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"authorized fixture audio")
        return subprocess.CompletedProcess(command, 0, stdout=str(target), stderr="")

    cold = acquire_youtube_audio(
        manifest, output, runner=fake_runner, check_runtime=False, audio_format="flac"
    )
    warm = acquire_youtube_audio(
        manifest, output, runner=fake_runner, check_runtime=False, audio_format="flac"
    )

    assert cold["completed"] == 1
    assert warm["skipped"] == 1
    assert len(calls) == 1
    receipt = json.loads((output / "receipts.jsonl").read_text(encoding="utf-8"))
    assert receipt["authorization_confirmed"] is True
    assert receipt["rights_basis"] == "permission"
    assert len(receipt["sha256"]) == 64
    assert (output / "commands.jsonl").is_file()
    assert (output / "run_summary.md").is_file()


def test_duplicate_urls_are_rejected(tmp_path):
    source = json.loads(_write_manifest(tmp_path / "manifest.json").read_text(encoding="utf-8"))
    source["items"].append(dict(source["items"][0]))
    (tmp_path / "manifest.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(AcquisitionError):
        load_acquisition_manifest(tmp_path / "manifest.json")
