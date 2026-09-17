"""Lazy CLI entry points for the local canonical pipeline."""

from __future__ import annotations
import json
from pathlib import Path
import subprocess
from typing import cast
import typer

app = typer.Typer(no_args_is_help=True)
catalog_app = typer.Typer(no_args_is_help=True)
features_app = typer.Typer(no_args_is_help=True)
acquire_app = typer.Typer(no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")
app.add_typer(features_app, name="features")
app.add_typer(acquire_app, name="acquire")


def _emit(value: object) -> None:
    typer.echo(json.dumps(value, sort_keys=True, default=str))


@app.command()
def doctor(path: Path = typer.Option(Path("."), "--path")) -> None:
    from .core.runtime import doctor as run_doctor

    _emit(run_doctor(path))


@catalog_app.command("scan")
def catalog_scan(
    library: Path = typer.Option(..., "--library"), output: Path = typer.Option(..., "--output")
) -> None:
    from .pipeline import scan_to_run

    _emit(scan_to_run(library, output))


@features_app.command("build")
def features_build(
    run: Path = typer.Option(..., "--run"), profile: str = typer.Option(..., "--profile")
) -> None:
    from .pipeline import build_features

    _emit(build_features(run, profile))


@acquire_app.command("youtube")
def acquire_youtube(
    manifest: Path = typer.Option(..., "--manifest"),
    output: Path = typer.Option(..., "--output"),
    audio_format: str = typer.Option("flac", "--audio-format"),
) -> None:
    from .acquisition import SUPPORTED_AUDIO_FORMATS, acquire_youtube_audio

    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        raise typer.BadParameter(
            f"audio format must be one of: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )
    result = acquire_youtube_audio(manifest, output, audio_format=audio_format)
    _emit(result)
    if result["failed"]:
        raise typer.Exit(1)


@app.command()
def discover(
    run: Path = typer.Option(..., "--run"),
    guidance: Path | None = typer.Option(None, "--guidance"),
) -> None:
    from .pipeline import discover_run

    _emit(discover_run(run, guidance))


@app.command()
def review(run: Path = typer.Option(..., "--run")) -> None:
    from .review_app import review_server_command

    raise typer.Exit(subprocess.run(review_server_command(str(run)), check=False).returncode)


@app.command()
def export(
    run: Path = typer.Option(..., "--run"),
    format: str = typer.Option(..., "--format"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    if format not in {"json", "csv", "m3u8", "html"}:
        raise typer.BadParameter("format must be one of: json, csv, m3u8, html")
    from .export import ExportFormat, export_run

    destination = output or run / "exports" / f"playlists.{format}"
    preview = export_run(run, cast(ExportFormat, format), destination)
    _emit(
        {
            "destination": str(destination.resolve()),
            "format": format,
            "rows": preview.row_count,
            "sha256": preview.sha256,
        }
    )


@app.command()
def evaluate(
    run: Path = typer.Option(..., "--run"),
    benchmark: Path = typer.Option(..., "--benchmark"),
) -> None:
    from .pipeline import evaluate_run

    _emit(evaluate_run(run, benchmark))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
