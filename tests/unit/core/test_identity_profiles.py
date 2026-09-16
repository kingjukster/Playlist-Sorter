import pytest
from pydantic import ValidationError

from playlist_sorter.core.contracts import (
    AcquisitionProvenance,
    FeatureProfile,
    parse_feature_profile,
)


def test_feature_profiles_are_explicit_and_have_closed_execution_requirements():
    descriptors = parse_feature_profile("descriptors")
    research = parse_feature_profile("research")

    assert descriptors is FeatureProfile.DESCRIPTORS
    assert descriptors.descriptors is True
    assert descriptors.requires_model_backed_features is False
    assert research is FeatureProfile.RESEARCH
    assert research.descriptors is False
    assert research.requires_model_backed_features is True


@pytest.mark.parametrize("value", [" Research", "research ", "RESEARCH", "unknown"])
def test_feature_profile_parser_rejects_unknown_or_normalized_values(value):
    with pytest.raises(ValueError):
        parse_feature_profile(value)


@pytest.mark.parametrize(
    "field,value",
    [("acquisition_id", ""), ("url", " "), ("source", ""), ("source_sha256", "not-a-hash")],
)
def test_acquisition_provenance_fails_closed_for_missing_or_malformed_identity(field, value):
    payload = {
        "acquisition_id": "youtube-0123456789abcdef",
        "url": "https://www.youtube.com/watch?v=BaW_jenozKc",
        "source_sha256": "a" * 64,
        "source": "youtube",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        AcquisitionProvenance.model_validate(payload)
