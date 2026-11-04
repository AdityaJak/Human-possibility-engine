# ADR-0005: Data separation and the simulation/consumer boundary
Status: accepted (2026-11-04)

**Decision.** Every stored row belongs to one `DataNamespace` (synthetic_simulation, external_dataset, user_private, experiment, evidence) in separate SQLite tables. `hpe.futures` must not import `hpe.world`, `hpe.development` or `hpe.evaluation` (enforced by an import-boundary test). Future Self evidence objects can be user facts, deterministic calculations, assumptions, narratives or externally cited studies, never simulated outcomes. Only demo/pseudonymous users are supported until authentication, encryption-at-rest and retention policies exist.
