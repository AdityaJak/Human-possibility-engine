# ADR-0002: Determinism, common random numbers and replay
Status: accepted (2026-11-04)

**Decision.** All randomness comes from `hpe.core.rng` streams keyed by `(seed, purpose, entity, ...)` using SHA-256-derived SeedSequence spawn keys. Agents are processed in sorted id order. Twin-society arms share `population_seed` and `task_seed`, and every per-agent stochastic purpose (mortality, activity sampling, school item draws, ...) has its own stream, so an intervention on one agent does not shift another agent's draws (common random numbers). The world maintains a running SHA-256 over the canonical JSON of each emitted event (`event_stream_hash`). Snapshots capture RNG bit-generator states, so resume-from-checkpoint is bit-identical.

**Consequences.** Reproducibility tests compare event-stream hashes across runs and across snapshot/resume. Wall-clock time is reported in manifests but never used in logic.
