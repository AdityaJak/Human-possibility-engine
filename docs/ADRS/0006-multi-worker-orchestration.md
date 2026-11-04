# ADR-0006: Multi-worker orchestration
Status: accepted (2026-11-04)

**Decision.** The orchestrator freezes contracts first, then launches five parallel Claude Code subagents (development, world, evaluation, backend, frontend), each in an isolated git worktree on a lane branch (`research/development`, `sim/world`, `science/evaluation`, `platform/backend`, `product/frontend`). Specialist lanes (`infra/release`, `science/human-validation`, `audit/review`) start once their inputs exist. The orchestrator merges lane branches locally in dependency order after lint/type/test gates pass, then pushes the integrated result.

**Constraint.** This cloud session may push only to the branch `claude/human-possibility-engine-yjujoc`. Lane branches are therefore local-only; their history is preserved in the integrated branch through merge commits.
