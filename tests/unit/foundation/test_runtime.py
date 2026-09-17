from playlist_sorter.core.runtime import ResourceSnapshot, readiness
from playlist_sorter.core.telemetry import new_run_manifest, record_telemetry


def test_readiness_fails_closed_for_unknown_or_low_resources():
    assert not readiness(ResourceSnapshot()).get("ready")
    assert not readiness(ResourceSnapshot(8, 12, 2, contention=True)).get("ready")
    assert readiness(ResourceSnapshot(8, 12, 2)).get("ready")


def test_manifest_retains_command_config_and_telemetry_provenance():
    manifest = new_run_manifest("r1", command="features build", config={"profile": "fast"})
    event = record_telemetry(manifest, "resource_snapshot", vram_gib=12)
    assert manifest["command"] == "features build"
    assert manifest["config"]["profile"] == "fast"
    assert event["event"] == "resource_snapshot"
    assert "timestamp" in event
