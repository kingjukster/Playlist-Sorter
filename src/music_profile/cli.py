from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from music_profile.builder import build_profile, write_outputs
from music_profile.musickit import fetch_playlist, playlist_id_from_reference, playlist_reference
from music_profile.util import read_json, write_json
from music_profile.validation import validate_file, validate_profile

STATE_DIR = Path(".music-profile")
REGISTRY = STATE_DIR / "sources.json"


def _load_registry() -> dict[str, list[dict[str, str]]]:
    return read_json(REGISTRY) if REGISTRY.exists() else {"sources": []}


def _register(path: Path, *, reference: str = "") -> Path:
    resolved = path.resolve()
    registry = _load_registry()
    record = {"path": str(resolved), "reference": reference}
    if record not in registry["sources"]:
        registry["sources"].append(record)
    write_json(REGISTRY, registry)
    return resolved


def _cmd_add_playlist(args: argparse.Namespace) -> int:
    references = list(args.reference)
    if args.capture:
        if len(references) != 1:
            print("--capture accepts exactly one playlist reference", file=sys.stderr)
            return 2
        path = _register(Path(args.capture), reference=references[0])
        print(f"Registered playlist capture: {path}")
        return 0
    developer_token = os.environ.get("APPLE_MUSIC_DEVELOPER_TOKEN", "")
    user_token = os.environ.get("APPLE_MUSIC_USER_TOKEN", "")
    kinds = [playlist_reference(reference)[1] for reference in references]
    if not developer_token or ("library" in kinds and not user_token):
        print(
            "Apple Music URLs require --capture PATH or a developer token; "
            "private library playlists also require a Music User Token. "
            "No credentials or browser session data were collected.",
            file=sys.stderr,
        )
        return 2
    for reference in references:
        playlist_id = playlist_id_from_reference(reference)
        payload = fetch_playlist(reference, developer_token, user_token, country=args.country)
        capture = STATE_DIR / "imports" / f"{playlist_id}.json"
        write_json(capture, payload)
        path = _register(capture, reference=reference)
        print(f"Fetched and registered {payload['source_track_count']} tracks: {path}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    path = _register(Path(args.path))
    print(f"Registered import: {path}")
    return 0


def _cmd_import_clipboard(args: argparse.Namespace) -> int:
    try:
        import tkinter
    except ImportError as error:  # pragma: no cover - platform Python packaging
        raise ValueError("This Python installation does not include tkinter") from error
    root = tkinter.Tk()
    root.withdraw()
    try:
        text = root.clipboard_get()
    finally:
        root.destroy()
    payload = json.loads(text)
    playlist = payload.get("playlist", {})
    playlist_id = str(playlist.get("playlist_id") or "clipboard")
    safe_name = "".join(
        character if character.isalnum() or character in "-." else "-" for character in playlist_id
    )
    capture = STATE_DIR / "imports" / f"{safe_name}.json"
    write_json(capture, payload)
    _register(capture, reference=str(playlist.get("url") or ""))
    track_count = len(payload.get("tracks", []))
    print(f"Imported and registered {track_count} clipboard tracks: {capture.resolve()}")
    return 0


def _input_paths(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(item) for item in explicit]
    return [Path(item["path"]) for item in _load_registry()["sources"]]


def _cmd_build(args: argparse.Namespace) -> int:
    paths = _input_paths(args.input)
    if not paths:
        print("No sources registered. Use add-playlist, import-file, or --input.", file=sys.stderr)
        return 2
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print(f"Missing input files: {', '.join(missing)}", file=sys.stderr)
        return 2
    profile, warnings, unresolved = build_profile(
        paths,
        manual_preferences=Path(args.manual_preferences) if args.manual_preferences else None,
        enrich=args.enrich,
        country=args.country,
    )
    output_dir = Path(args.output_dir)
    write_outputs(profile, output_dir)
    report = validate_profile(profile, warnings=warnings, unresolved_items=unresolved)
    write_json(output_dir / "validation_report.json", report)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "playlists": len(profile["playlists"]),
                "tracks": len(profile["tracks"]),
                "albums": len(profile["albums"]),
                "artists": len(profile["artists"]),
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_file(Path(args.path), Path(args.report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


def _cmd_init_preferences(args: argparse.Namespace) -> int:
    destination = Path(args.path)
    if destination.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {destination}", file=sys.stderr)
        return 2
    template = Path(__file__).resolve().parents[2] / "manual_preferences.json"
    shutil.copyfile(template, destination)
    print(f"Created {destination.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="music-profile")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add-playlist", help="fetch or register an Apple Music playlist")
    add.add_argument("reference", nargs="+", help="one or more Apple Music playlist URLs or IDs")
    add.add_argument("--capture", help="browser-capture JSON file")
    add.add_argument("--country", default="us", help="two-letter Apple storefront country")
    add.set_defaults(func=_cmd_add_playlist)

    imported = subparsers.add_parser("import-file", help="register JSON, CSV, XML, or plist data")
    imported.add_argument("path")
    imported.set_defaults(func=_cmd_import)

    clipboard = subparsers.add_parser(
        "import-clipboard",
        help="save and register a browser-capture JSON object from the clipboard",
    )
    clipboard.set_defaults(func=_cmd_import_clipboard)

    build = subparsers.add_parser("build", help="build the canonical profile and validation report")
    build.add_argument("--input", action="append", default=[], help="input file; repeatable")
    build.add_argument("--output-dir", default=".")
    build.add_argument("--manual-preferences", default="manual_preferences.json")
    build.add_argument("--enrich", action="store_true", help="query Apple's public iTunes catalog")
    build.add_argument("--country", default="us", help="two-letter Apple storefront country")
    build.set_defaults(func=_cmd_build)

    validate = subparsers.add_parser("validate", help="validate a canonical profile JSON file")
    validate.add_argument("path", nargs="?", default="music_profile_source.json")
    validate.add_argument("--report", default="validation_report.json")
    validate.set_defaults(func=_cmd_validate)

    preferences = subparsers.add_parser(
        "init-preferences", help="create a manual preference template"
    )
    preferences.add_argument("--path", default="manual_preferences.json")
    preferences.add_argument("--force", action="store_true")
    preferences.set_defaults(func=_cmd_init_preferences)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
