import pytest
from pydantic import ValidationError
from playlist_sorter.core.contracts import Membership, GuidanceSet, AudioSegment
from playlist_sorter.core.artifacts import artifact_fingerprint, artifact_key


def test_unknown_schema_version_rejected():
    with pytest.raises(ValidationError):
        Membership(song_id="s", candidate_id="c", score=0.5, threshold=0.4, schema_version="9.0")


def test_scores_are_bounded():
    with pytest.raises(ValidationError):
        Membership(song_id="s", candidate_id="c", score=1.1, threshold=0.4)


def test_guidance_limits_examples():
    with pytest.raises(ValidationError):
        GuidanceSet(guidance_id="g", positive_song_ids=[str(i) for i in range(20)])


def test_segment_must_have_positive_duration():
    with pytest.raises(ValidationError):
        AudioSegment(segment_id="x", song_id="s", start_seconds=3, end_seconds=3, index=0)


def test_artifact_helpers_are_canonical():
    assert artifact_fingerprint({"b": 2, "a": 1}) == artifact_fingerprint({"a": 1, "b": 2})
    assert artifact_key("features", {"song": "s"}) == artifact_key("features", {"song": "s"})
