import json
from datetime import datetime, timezone

import pytest
from playlist_sorter.core.contracts import PlaylistCandidate, ReviewAction
from playlist_sorter.naming import name_candidate
from playlist_sorter.review.artifacts import CanonicalArtifactError, load_canonical_run
from playlist_sorter.review.feedback import FeedbackStore, validate_review_action


def _candidate(evidence: list[str]) -> PlaylistCandidate:
    return PlaylistCandidate(
        candidate_id="cand-1",
        name="original",
        granularity="narrow",
        member_song_ids=["song-1", "song-2"],
        representative_song_ids=["song-1"],
        stability=0.9,
        cohesion=0.8,
        separation=0.7,
        novelty=0.6,
        coverage=0.5,
        naming_evidence=evidence,
    )


def _action(action: str, **extra: str) -> ReviewAction:
    payload = {
        "action_id": f"action-{action}",
        "candidate_id": "cand-1",
        "action": action,
        "actor": "reviewer",
        "created_at": datetime.now(timezone.utc),
    }
    payload.update(extra)
    return ReviewAction(**payload)


def _write_run(tmp_path):
    manifest = {
        "schema_version": "1.0",
        "run_id": "run-1",
        "created_at": "2026-01-01T00:00:00Z",
        "discovery_run": "discovery-1",
        "artifacts": [],
        "candidates": ["cand-1"],
        "generator_version": "1.0",
    }
    song = {
        "schema_version": "1.0",
        "song_id": "song-1",
        "source_path": "C:/music/song.mp3",
        "integrity_fingerprint": "abc",
    }
    second_song = {**song, "song_id": "song-2", "source_path": "C:/music/song-two.mp3"}
    candidate = _candidate(["Night Drive", "Night Drive", "Synth"])
    membership = {
        "schema_version": "1.0",
        "song_id": "song-1",
        "candidate_id": "cand-1",
        "score": 0.9,
        "threshold": 0.4,
    }
    for name, payload in {
        "manifest.json": manifest,
        "playlists.json": {"schema_version": "1.0", "items": [candidate.model_dump(mode="json")]},
        "memberships.json": {"schema_version": "1.0", "items": [membership]},
        "songs.json": {"schema_version": "1.0", "items": [song, second_song]},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")


def test_naming_is_deterministic_and_does_not_mutate_membership():
    candidate = _candidate([" Synth ", "Night Drive", "Night Drive", "Synth"])
    before = list(candidate.member_song_ids)
    result = name_candidate(candidate)

    assert result.name == "Night Drive / Synth"
    assert result.dominant_evidence == ("Night Drive", "Synth")
    assert candidate.member_song_ids == before
    assert name_candidate(_candidate(list(reversed(candidate.naming_evidence)))) == result


def test_naming_fallback_is_stable():
    result = name_candidate(_candidate([]))
    assert result.name == "Playlist cand-1"
    assert "stable fallback" in result.explanation


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("approve", {}),
        ("reject", {}),
        ("rename", {"value": "Late Night"}),
        ("split", {"value": "song-1,song-2"}),
        ("merge", {"value": "cand-2"}),
        ("intrusion", {"song_id": "song-1"}),
        ("omission", {"song_id": "song-1"}),
    ],
)
def test_review_action_accepts_each_valid_action(action, payload):
    assert _action(action, **payload).action == action


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("approve", {"song_id": "song-1"}),
        ("reject", {"value": "why"}),
        ("rename", {}),
        ("rename", {"song_id": "song-1", "value": "name"}),
        ("split", {}),
        ("merge", {"song_id": "song-1", "value": "cand-2"}),
        ("intrusion", {}),
        ("omission", {"song_id": "song-1", "value": "extra"}),
    ],
)
def test_review_action_rejects_invalid_payloads(action, payload):
    with pytest.raises(ValueError):
        validate_review_action(_action(action, **payload))


def test_feedback_is_append_only_and_strict(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    store.append(_action("approve"), candidate_ids={"cand-1"}, song_ids={"song-1"})
    store.append(
        _action("intrusion", song_id="song-1"), candidate_ids={"cand-1"}, song_ids={"song-1"}
    )
    assert [item.action for item in store.read()] == ["approve", "intrusion"]
    assert len((tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError, match="unknown candidate"):
        store.append(_action("approve"), candidate_ids={"other"})
    store.path.write_text(
        _action("approve", song_id="song-1").model_dump_json() + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid feedback"):
        store.read()
    (tmp_path / "feedback.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid feedback"):
        store.read()


def test_canonical_reader_loads_versioned_artifacts(tmp_path):
    _write_run(tmp_path)
    artifacts = load_canonical_run(tmp_path)
    assert artifacts.candidate_ids == {"cand-1"}
    assert artifacts.song_ids == {"song-1", "song-2"}


@pytest.mark.parametrize(
    "filename,payload",
    [
        ("playlists.json", []),
        ("songs.json", {"schema_version": "9.0", "items": []}),
        ("memberships.json", {"schema_version": "1.0", "items": [{"bad": "shape"}]}),
    ],
)
def test_canonical_reader_fails_closed_on_malformed_artifacts(tmp_path, filename, payload):
    _write_run(tmp_path)
    (tmp_path / filename).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CanonicalArtifactError):
        load_canonical_run(tmp_path)


def test_canonical_reader_rejects_inconsistent_references(tmp_path):
    _write_run(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["candidates"] = ["other"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanonicalArtifactError, match="manifest candidates"):
        load_canonical_run(tmp_path)
