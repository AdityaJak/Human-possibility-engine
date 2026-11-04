# Changelog

All notable changes are recorded here. Dates are UTC.

## [Unreleased]
### Fixed
- 2026-11-04 Future Self safety: indirect crisis language, medical/legal vocabulary, input size limits (audit H-1, M-1, M-2); API parsing of world assessment payloads; react-router 7.18.5 (GHSA-wrjc-x8rr-h8h6); CI venv reuse.
### Performance
- 2026-11-04 Task sampling ~3× faster with bit-identical draws.

### Added
- 2026-11-04 Developmental learner (research/development lane); society kernel (sim/world); baselines, statistics, preregistered runner and learning lab (science/evaluation); SQLite storage, FastAPI service, Future Self engine (platform/backend); research dashboard + Future Self web app (product/frontend); make demo, Docker files, scale harness, CI hardening (infra/release); Psych-101 and HeartSteps adapters with held-out human evaluation and causal grounding (science/human-validation); independent audit (audit/review).
- 2026-11-04 Results: full-budget H1 (failed), Exp 1, Exp 6, ablations, Exp 2 twin tutoring; stored 100-agent × 80-year run with replay verification; HeartSteps causal grounding; evidence ledger (`python -m hpe.ledger`); RESEARCH_RESULTS, investor demo + diligence docs, model cards, risks.
- 2026-11-04 Phase 0 foundation: frozen v0.1 contracts (Pydantic schemas, Policy protocol, World/EventSink protocols, policy registry, JSON Schema + TypeScript export), deterministic named RNG streams, novel-symbol task world with held-out splits and transfer control, governance docs (AGENTS, ARCHITECTURE, INTERFACES, CLAIMS_POLICY, ADR-0001..0007, task board), CI workflow.
