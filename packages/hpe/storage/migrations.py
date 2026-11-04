"""Versioned SQL migrations.

Each migration is applied once, in order, inside a transaction, and recorded in
``schema_migrations``. SQL is kept close to ANSI so a PostgreSQL port only needs the
``AUTOINCREMENT`` columns rewritten as ``GENERATED ALWAYS AS IDENTITY``.

Tables are grouped by ``DataNamespace`` (ADR-0005). Rows of different namespaces never share a
table, so synthetic simulation data, external datasets, real-user private data, experiments and
evidence can be exported, retained and deleted independently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "initial_namespaced_schema",
        """
-- namespace: synthetic_simulation ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS sim_runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    status        TEXT NOT NULL,
    mode          TEXT NOT NULL,
    source        TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    manifest_json TEXT,
    run_dir       TEXT,
    events_imported_sha256 TEXT,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS sim_events (
    run_id          TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    tick            INTEGER NOT NULL,
    day             INTEGER NOT NULL,
    type            TEXT NOT NULL,
    agent_id        TEXT,
    other_ids       TEXT NOT NULL,
    event_json      TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_sim_events_agent ON sim_events (run_id, agent_id, seq);
CREATE INDEX IF NOT EXISTS ix_sim_events_type ON sim_events (run_id, type, seq);
CREATE INDEX IF NOT EXISTS ix_sim_events_tick ON sim_events (run_id, tick);
CREATE TABLE IF NOT EXISTS sim_ticks (
    run_id    TEXT NOT NULL,
    tick      INTEGER NOT NULL,
    tick_json TEXT NOT NULL,
    PRIMARY KEY (run_id, tick)
);
CREATE TABLE IF NOT EXISTS sim_checkpoints (
    run_id     TEXT NOT NULL,
    tick       INTEGER NOT NULL,
    path       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, tick)
);

-- namespace: experiment ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id        TEXT PRIMARY KEY,
    preregistration_hash TEXT NOT NULL,
    definition_json      TEXT NOT NULL,
    status               TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    output_dir           TEXT,
    error                TEXT
);
CREATE TABLE IF NOT EXISTS experiment_results (
    experiment_id TEXT PRIMARY KEY,
    result_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- namespace: evidence ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    record_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- namespace: external_dataset -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS external_datasets (
    dataset_id       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    source_url       TEXT,
    license          TEXT,
    license_verified INTEGER NOT NULL DEFAULT 0,
    sha256           TEXT,
    meta_json        TEXT NOT NULL,
    registered_at    TEXT NOT NULL
);

-- namespace: user_private (demo / pseudonymous users only, see ADR-0005) ----------------------
CREATE TABLE IF NOT EXISTS user_consent (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    purpose    TEXT NOT NULL,
    granted    INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_consent_user ON user_consent (user_id);
CREATE TABLE IF NOT EXISTS user_decisions (
    decision_id  TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_decisions_user ON user_decisions (user_id);
CREATE TABLE IF NOT EXISTS user_scenarios (
    scenario_id   TEXT PRIMARY KEY,
    decision_id   TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    scenario_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_scenarios_user ON user_scenarios (user_id);
CREATE TABLE IF NOT EXISTS user_evidence (
    evidence_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_evidence_user ON user_evidence (user_id);
CREATE TABLE IF NOT EXISTS user_reflections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    decision_id     TEXT NOT NULL,
    reflection_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_reflections_user ON user_reflections (user_id);

-- audit -------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    namespace   TEXT NOT NULL,
    subject     TEXT,
    detail_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_subject ON audit_log (subject);
""",
    ),
)

# Tables holding real-user private data. Export and delete operate on exactly these.
USER_PRIVATE_TABLES: tuple[str, ...] = (
    "user_consent",
    "user_decisions",
    "user_scenarios",
    "user_evidence",
    "user_reflections",
)
