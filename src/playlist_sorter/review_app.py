"""Local-only Streamlit presentation for canonical playlist review artifacts.

All validation, artifact IO, feedback persistence, naming, and export rendering
lives in shared services.  This module only maps a reviewer interaction onto
those services and is safe to import without starting a Streamlit server.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import cast
from uuid import uuid4

import streamlit as st

from playlist_sorter.core.contracts import ReviewAction, SongRecord
from playlist_sorter.export import (
    ExportError,
    ExportFormat,
    ExportPreview,
    create_preview,
    write_preview,
)
from playlist_sorter.naming import explain_candidate, name_candidate
from playlist_sorter.review import FeedbackStore, load_canonical_run

LOCALHOST_ADDRESS = "127.0.0.1"
SUPPORTED_ACTIONS = ("approve", "reject", "rename", "split", "merge", "intrusion", "omission")


def review_server_args() -> tuple[str, str]:
    """Return the non-negotiable Streamlit address option for the review CLI."""
    return ("--server.address", LOCALHOST_ADDRESS)


def review_server_command(run_directory: str) -> tuple[str, ...]:
    """Return the only supported local review launch contract; do not execute it."""
    return (
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        *review_server_args(),
        "--",
        "--run",
        str(Path(run_directory)),
    )


def _run_argument() -> str | None:
    """Read the explicit script argument passed after Streamlit's ``--`` separator."""
    try:
        position = sys.argv.index("--run")
        return sys.argv[position + 1]
    except (ValueError, IndexError):
        return None


@st.cache_data(ttl="30s", max_entries=20)
def _load_run(run_directory: str):
    return load_canonical_run(run_directory)


def _display_songs(title: str, song_ids: list[str], songs: dict[str, SongRecord]) -> None:
    st.subheader(title)
    rows = []
    for song_id in song_ids:
        song = songs.get(song_id)
        if song is not None:
            rows.append({"song ID": song_id, "title": song.title, "artist": song.artist})
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("None")


def _action_for(
    candidate_id: str, action: str, actor: str, song_id: str, value: str
) -> ReviewAction:
    payload: dict[str, object] = {
        "action_id": str(uuid4()),
        "candidate_id": candidate_id,
        "action": action,
        "actor": actor,
        "created_at": datetime.now(UTC),
        "provenance": {"surface": "streamlit-review", "blind_comparison": False},
    }
    if action in {"intrusion", "omission"}:
        payload["song_id"] = song_id
    elif action in {"rename", "split", "merge"}:
        payload["value"] = value
    return ReviewAction.model_validate(payload)


def _render_feedback(candidate, artifacts, run_directory: str) -> None:
    st.subheader("Record feedback")
    with st.form(f"feedback-{candidate.candidate_id}"):
        actor = st.text_input("Reviewer", value="local-reviewer")
        action = st.selectbox("Action", SUPPORTED_ACTIONS)
        song_id = st.selectbox(
            "Song for intrusion or omission", [""] + list(candidate.member_song_ids)
        )
        value = st.text_input("Name or target candidate IDs for rename, split, or merge")
        submitted = st.form_submit_button("Append feedback", type="primary")
    if submitted:
        try:
            review_action = _action_for(candidate.candidate_id, action, actor, song_id or "", value)
            FeedbackStore(Path(run_directory) / "feedback.jsonl").append(
                review_action, candidate_ids=artifacts.candidate_ids, song_ids=artifacts.song_ids
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("Feedback appended durably.")


def _render_blind_comparison(candidates, artifacts, run_directory: str) -> None:
    st.subheader("Blind comparison")
    if len(candidates) < 2:
        st.caption("At least two candidates are required for a blind comparison.")
        return
    ordered = sorted(candidates, key=lambda item: item.candidate_id)
    left, right = ordered[0], ordered[1]
    with st.form("blind-comparison"):
        choice = st.segmented_control("Which grouping is more coherent?", ["A", "B", "Neither"])
        submitted = st.form_submit_button("Record blind choice")
    if submitted:
        selection = {"A": left, "B": right}.get(choice or "")
        if selection is None:
            st.info("No preference was recorded.")
        else:
            action = ReviewAction(
                action_id=str(uuid4()),
                candidate_id=selection.candidate_id,
                action="approve",
                actor="local-reviewer",
                created_at=datetime.now(UTC),
                provenance={
                    "surface": "streamlit-review",
                    "blind_comparison": True,
                    "choice": choice,
                },
            )
            FeedbackStore(Path(run_directory) / "feedback.jsonl").append(
                action, candidate_ids=artifacts.candidate_ids, song_ids=artifacts.song_ids
            )
            st.success(f"Selected {choice}; blind comparison feedback appended durably.")
    for label, candidate in (("A", left), ("B", right)):
        st.caption(
            f"{label}: {len(candidate.member_song_ids)} songs, {candidate.granularity} grouping"
        )


def _render_export(run_directory: str) -> None:
    st.subheader("Preview export")
    with st.form("export-preview"):
        format = st.selectbox("Format", ["html", "json", "csv", "m3u8"])
        destination = st.text_input(
            "Derived export destination", value=str(Path(run_directory) / f"preview.{format}")
        )
        preview_requested = st.form_submit_button("Create preview")
    if preview_requested:
        try:
            preview = create_preview(run_directory, cast(ExportFormat, format))
        except ExportError as error:
            st.error(str(error))
            return
        st.session_state["export_preview"] = preview
        st.session_state["export_destination"] = destination
    state_preview = cast(ExportPreview | None, st.session_state.get("export_preview"))
    if state_preview is not None:
        st.code(state_preview.text, language="text")
        st.caption(f"{state_preview.row_count} rows; SHA-256 {state_preview.sha256}")
        if st.button("Write this preview atomically", type="primary"):
            try:
                write_preview(state_preview, st.session_state["export_destination"])
            except ExportError as error:
                st.error(str(error))
            else:
                st.success("Derived export replaced atomically.")


def main(run_directory: str | None = None) -> None:
    """Render the review surface; caller supplies a canonical run directory."""
    st.set_page_config(page_title="Playlist review", page_icon=":material/queue_music:")
    st.title("Playlist review")
    supplied_run = run_directory or st.text_input("Canonical run directory")
    if not supplied_run:
        st.info("Enter a canonical run directory to begin.")
        return
    try:
        artifacts = _load_run(supplied_run)
    except ValueError as error:
        st.error(str(error))
        return
    candidate_ids = [candidate.candidate_id for candidate in artifacts.candidates]
    if not candidate_ids:
        st.info(
            "Discovery abstained: this run contains no playlist candidates that passed the gates."
        )
        _render_export(supplied_run)
        return
    candidate_id = st.selectbox("Candidate", candidate_ids)
    candidate = next(item for item in artifacts.candidates if item.candidate_id == candidate_id)
    result = name_candidate(candidate)
    st.header(result.name)
    st.caption(explain_candidate(candidate))
    songs = {song.song_id: song for song in artifacts.songs}
    _display_songs(
        "Representative evidence",
        candidate.representative_song_ids or candidate.core_song_ids,
        songs,
    )
    _display_songs("Borderline evidence", candidate.boundary_song_ids, songs)
    _display_songs("Rejected evidence", candidate.excluded_song_ids, songs)
    _render_blind_comparison(artifacts.candidates, artifacts, supplied_run)
    _render_feedback(candidate, artifacts, supplied_run)
    _render_export(supplied_run)


if __name__ == "__main__":
    main(_run_argument())
