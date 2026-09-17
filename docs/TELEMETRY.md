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

The catalog-only 20-item smoke run is a prerequisite for the 200-item scale run. Resource
measurements that cannot be collected remain null/unknown and fail readiness;
they are not inferred from hardware specifications.

## Model-backed smoke record

Model-backed qualification is separate from the catalog-only scripts. Before an
authorized one-song attempt, save a fresh local preflight with RAM, GPU
VRAM/utilization/temperature/power, and competing processes. Do not load,
infer, or download a model before it. Record the exact command, profile,
Python/torch/CUDA runtime, commit, pinned revisions, cache directory,
start/end, wall time, item rate, peak RAM/VRAM, and stdout/stderr/telemetry/
summary paths.

Only a clean one-song result without NaN, OOM, or driver reset may start the
private 20-song smoke. Retain cold/warm cache counts and timings plus vector and
discovery artifact hashes. The authorized cap is 20 songs. If snapshots,
dependencies, authority, or resource safety are missing, record `unqualified`
with the exact blocker rather than running a substitute.
