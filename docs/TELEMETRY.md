# Telemetry and evidence

Telemetry is private-by-default and must never contain audio, lyrics, raw
embeddings, or unredacted library paths unless the user explicitly chooses a
local evidence location. `new_run_manifest` creates a versioned manifest and
`record_telemetry` appends UTC events with measurements.

Every smoke or benchmark receipt should include:

- exact command, repository commit, Python version, and platform;
- start/end UTC timestamps and wall seconds;
- item count, completed/failed counts, and items/second;
- available RAM, free disk, and VRAM when discoverable;
- cache hit/request counts and whether the result is cold or warm;
- source SHA-256 values or aggregate hashes, never source content;
- gate result and a human-readable reason for any `unknown` or failure.

The 20-item smoke run is a prerequisite for the 200-item scale run. Resource
measurements that cannot be collected remain null/unknown and fail readiness;
they are not inferred from hardware specifications.
