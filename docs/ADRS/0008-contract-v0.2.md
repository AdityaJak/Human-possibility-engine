# ADR-0008: Contract v0.2 (proposed — not yet applied)
Status: proposed (2026-11-04). Apply only after every running experiment has been committed, because
`preregistration_hash` currently covers `schema_version`; applying this bump earlier would orphan
in-flight preregistrations.

**Context.** The eight lanes integrated against the frozen v0.1 contract and requested the additive
changes below in their handoffs. None blocks the current build; each removes a workaround.

**Decision (to apply as one minor bump, `0.1.0 → 0.2.0`).**

| Change | Requested by | Kind |
|---|---|---|
| `OutcomeObservation.shots: int \| None` (few-shot transfer curves drawable by the web app) | frontend | additive |
| `ExperimentDefinition.protocol: str \| None`, `protocol_params: dict` (learning-lab experiments creatable through the API, not only from YAML) | evaluation | additive |
| A `HumanDatasetExperimentDefinition` variant (dataset id, split spec, models) in place of forcing a `SimulationConfig` | human-validation | additive |
| `ExperimentResult.evidence_kind` and rename `synthetic_disclaimer → disclaimer` | human-validation | **breaking** (keep `synthetic_disclaimer` as a deprecated alias for one version) |
| Exclude `schema_version` from `preregistration_hash` | orchestrator | behavioural (changes future hashes only; committed results keep their recorded hash) |
| `RunManifest.event_stream_hash` (today stored in `dataset_hashes["event_stream_sha256_chain"]`) | world | additive |
| `SocialEdgeKind.ACQUAINTANCE`; `FeedbackSource` adds `caregiver`, `mentor` | world | additive |
| `WorldParams.prehistory_max_episodes` (default 400) | world | additive |
| `AgentState.name` (today only inside persona text) | world | additive |
| `GET /experiments/{id}/figures/{name}` and `GET /human-validation/{id}/results` | frontend, human-validation | additive endpoints |
| Pin the `ASSESSMENT`, `POLICY_UPDATE` and `DECISION` event payload shapes as typed models (the untyped payload caused the empty-skill-curve defect found end-to-end) | orchestrator | additive models, tightening |

**Procedure.** Bump `CONTRACT_VERSION`, update models, `python scripts/export_contracts.py`,
`python apps/api/export_openapi.py`, run the full gate, re-run the pipeline-validation experiments, and
confirm `python -m hpe.evaluation analyze` still reproduces every committed `result.json` (the analysis
reads stored hashes; it does not recompute them).

**Consequences.** Committed experiments keep their v0.1 preregistration hashes and remain reproducible;
new preregistrations hash only scientific content.
