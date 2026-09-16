"""Small, lazy CLI shells; no pipeline/model modules import at startup."""

from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True)
catalog_app = typer.Typer(no_args_is_help=True)
features_app = typer.Typer(no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")
app.add_typer(features_app, name="features")


def _shell(command: str, **values: object) -> None:
    typer.echo(json.dumps({"command": command, **values}, sort_keys=True))


@app.command()
def doctor(path: Path = typer.Option(Path("."), "--path")) -> None:
    from .core.runtime import doctor as run_doctor

    typer.echo(json.dumps(run_doctor(path), sort_keys=True))


@catalog_app.command("scan")
def catalog_scan(
    library: Path = typer.Option(..., "--library"), output: Path = typer.Option(..., "--output")
) -> None:
    _shell("catalog scan", library=str(library), output=str(output))


@features_app.command("build")
def features_build(
    run: str = typer.Option(..., "--run"), profile: str = typer.Option(..., "--profile")
) -> None:
    _shell("features build", run=run, profile=profile)


@app.command()
def discover(
    run: str = typer.Option(..., "--run"), guidance: str | None = typer.Option(None, "--guidance")
) -> None:
    _shell("discover", run=run, guidance=guidance)


@app.command()
def review(run: str = typer.Option(..., "--run")) -> None:
    _shell("review", run=run)


@app.command()
def export(
    run: str = typer.Option(..., "--run"), format: str = typer.Option(..., "--format")
) -> None:
    if format not in {"json", "csv", "m3u8"}:
        raise typer.BadParameter("format must be one of: json, csv, m3u8")
    _shell("export", run=run, format=format)


@app.command()
def evaluate(
    run: str = typer.Option(..., "--run"), benchmark: str = typer.Option(..., "--benchmark")
) -> None:
    _shell("evaluate", run=run, benchmark=benchmark)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
