# ADR-0001: Monorepo, namespace and stack
Status: accepted (2026-11-04)

**Context.** Spec §9/§39 suggest Python 3.12+ / FastAPI / Pydantic / SQLite / pytest and a TypeScript + React (Vite) front end; Part IX suggests `packages/<lane>/` folders and parallel git worktrees.

**Decision.** One Python distribution `human-possibility-engine` exposing the namespace `hpe.*` from `packages/hpe/<module>/` plus `hpe_api` from `apps/api/`. pytest puts `packages/` and `apps/api/` on `sys.path`, so every worktree tests its own code without reinstalling. Dependencies: numpy, scipy, pydantic v2, FastAPI/uvicorn, PyYAML, matplotlib; dev: pytest, hypothesis, ruff, mypy. `uv` manages the venv and the `uv.lock` lockfile. Front end: Vite + React + TypeScript, npm lockfile.

**Consequences.** Lane ownership maps to `packages/hpe/<module>/` instead of `packages/<module>/`. No ORM: stdlib `sqlite3` with versioned SQL migrations keeps a clean path to PostgreSQL (SQL is kept ANSI where possible).
