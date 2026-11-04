"""Repositories: typed access to each namespace's tables. No business logic lives here."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from hpe.contracts.schemas import (
    ConsentRecord,
    DataNamespace,
    DecisionRequest,
    EvidenceRecord,
    ExperienceEvent,
    ExperimentDefinition,
    ExperimentResult,
    Reflection,
    RunManifest,
    Scenario,
    WorldTick,
)
from hpe.storage.database import Database, now_iso
from hpe.storage.migrations import USER_PRIVATE_TABLES


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------
# synthetic_simulation
# ---------------------------------------------------------------------------------------


class RunRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(
        self,
        run_id: str,
        *,
        status: str,
        mode: str,
        source: str,
        config_json: str,
        manifest: RunManifest | None = None,
        run_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        ts = now_iso()
        self.db.execute(
            """INSERT INTO sim_runs (run_id, created_at, updated_at, status, mode, source, config_json,
                                     manifest_json, run_dir, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status,
                   mode=excluded.mode, source=excluded.source, config_json=excluded.config_json,
                   manifest_json=COALESCE(excluded.manifest_json, sim_runs.manifest_json),
                   run_dir=COALESCE(excluded.run_dir, sim_runs.run_dir), error=excluded.error""",
            (
                run_id,
                ts,
                ts,
                status,
                mode,
                source,
                config_json,
                manifest.model_dump_json() if manifest else None,
                run_dir,
                error,
            ),
        )

    def set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE sim_runs SET status=?, error=?, updated_at=? WHERE run_id=?",
            (status, error, now_iso(), run_id),
        )

    def set_manifest(self, run_id: str, manifest: RunManifest) -> None:
        self.db.execute(
            "UPDATE sim_runs SET manifest_json=?, updated_at=? WHERE run_id=?",
            (manifest.model_dump_json(), now_iso(), run_id),
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM sim_runs WHERE run_id=?", (run_id,))
        return dict(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM sim_runs ORDER BY created_at")]

    def imported_sha(self, run_id: str) -> str | None:
        row = self.db.query_one("SELECT events_imported_sha256 FROM sim_runs WHERE run_id=?", (run_id,))
        return row["events_imported_sha256"] if row else None

    def mark_imported(self, run_id: str, sha256: str) -> None:
        self.db.execute("UPDATE sim_runs SET events_imported_sha256=? WHERE run_id=?", (sha256, run_id))

    def delete_run_data(self, run_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM sim_events WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM sim_ticks WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM sim_checkpoints WHERE run_id=?", (run_id,))


class EventRepository:
    """Append-only event store. Events are immutable once written."""

    INSERT = (
        "INSERT OR IGNORE INTO sim_events (run_id, seq, tick, day, type, agent_id, other_ids, event_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def row(event: ExperienceEvent, run_id: str | None = None) -> tuple[Any, ...]:
        return (
            run_id or event.run_id,
            event.seq,
            event.tick,
            event.day,
            str(event.type.value if hasattr(event.type, "value") else event.type),
            event.agent_id,
            _dumps(list(event.other_ids)),
            event.model_dump_json(),
        )

    def append_many(self, events: Iterable[ExperienceEvent], run_id: str | None = None) -> int:
        rows = [self.row(e, run_id) for e in events]
        self.db.executemany(self.INSERT, rows)
        return len(rows)

    def append_rows(self, rows: Sequence[Sequence[Any]]) -> None:
        self.db.executemany(self.INSERT, rows)

    def _where(
        self,
        run_id: str,
        agent_id: str | None,
        types: Sequence[str] | None,
        tick_from: int | None,
        tick_to: int | None,
    ) -> tuple[str, list[Any]]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if agent_id:
            clauses.append("(agent_id = ? OR other_ids LIKE ?)")
            params.extend([agent_id, f'%"{agent_id}"%'])
        if types:
            clauses.append("type IN (" + ",".join("?" * len(types)) + ")")
            params.extend(types)
        if tick_from is not None:
            clauses.append("tick >= ?")
            params.append(tick_from)
        if tick_to is not None:
            clauses.append("tick <= ?")
            params.append(tick_to)
        return " AND ".join(clauses), params

    def query(
        self,
        run_id: str,
        *,
        agent_id: str | None = None,
        types: Sequence[str] | None = None,
        tick_from: int | None = None,
        tick_to: int | None = None,
        offset: int = 0,
        limit: int = 200,
        descending: bool = False,
    ) -> tuple[list[ExperienceEvent], int]:
        where, params = self._where(run_id, agent_id, types, tick_from, tick_to)
        total_row = self.db.query_one(f"SELECT COUNT(*) AS n FROM sim_events WHERE {where}", params)
        total = int(total_row["n"]) if total_row else 0
        order = "DESC" if descending else "ASC"
        rows = self.db.query(
            f"SELECT event_json FROM sim_events WHERE {where} ORDER BY seq {order} LIMIT ? OFFSET ?",
            [*params, int(limit), int(offset)],
        )
        return [ExperienceEvent.model_validate_json(r["event_json"]) for r in rows], total

    def iter_all(self, run_id: str, batch: int = 5000) -> Iterable[str]:
        """Yield raw event JSON in replay (seq) order."""
        last = -1
        while True:
            rows = self.db.query(
                "SELECT seq, event_json FROM sim_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (run_id, last, batch),
            )
            if not rows:
                return
            for r in rows:
                yield r["event_json"]
            last = int(rows[-1]["seq"])

    def count(self, run_id: str) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM sim_events WHERE run_id=?", (run_id,))
        return int(row["n"]) if row else 0


class TickRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append_many(self, run_id: str, ticks: Iterable[WorldTick]) -> None:
        rows = [(run_id, t.tick, t.model_copy(update={"frames": None}).model_dump_json()) for t in ticks]
        self.db.executemany(
            "INSERT OR REPLACE INTO sim_ticks (run_id, tick, tick_json) VALUES (?, ?, ?)", rows
        )

    def list(self, run_id: str) -> list[WorldTick]:
        rows = self.db.query("SELECT tick_json FROM sim_ticks WHERE run_id=? ORDER BY tick", (run_id,))
        return [WorldTick.model_validate_json(r["tick_json"]) for r in rows]


class CheckpointRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, run_id: str, tick: int, path: str, sha256: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO sim_checkpoints (run_id, tick, path, sha256, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, tick, path, sha256, now_iso()),
        )

    def list(self, run_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query("SELECT * FROM sim_checkpoints WHERE run_id=? ORDER BY tick", (run_id,))
        ]


# ---------------------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------------------


class ExperimentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, definition: ExperimentDefinition, status: str = "queued") -> None:
        ts = now_iso()
        self.db.execute(
            """INSERT INTO experiments (experiment_id, preregistration_hash, definition_json, status, created_at,
                                        updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                definition.experiment_id,
                definition.preregistration_hash or "",
                definition.model_dump_json(),
                status,
                ts,
                ts,
            ),
        )

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,))
        return dict(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM experiments ORDER BY created_at")]

    def set_status(
        self, experiment_id: str, status: str, *, error: str | None = None, output_dir: str | None = None
    ) -> None:
        self.db.execute(
            "UPDATE experiments SET status=?, error=?, output_dir=COALESCE(?, output_dir), updated_at=? WHERE experiment_id=?",
            (status, error, output_dir, now_iso(), experiment_id),
        )

    def save_result(self, result: ExperimentResult) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO experiment_results (experiment_id, result_json, created_at) VALUES (?, ?, ?)",
            (result.experiment_id, result.model_dump_json(), now_iso()),
        )

    def get_result(self, experiment_id: str) -> ExperimentResult | None:
        row = self.db.query_one(
            "SELECT result_json FROM experiment_results WHERE experiment_id=?", (experiment_id,)
        )
        return ExperimentResult.model_validate_json(row["result_json"]) if row else None


# ---------------------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------------------


class EvidenceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, record: EvidenceRecord) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO evidence (evidence_id, record_json, created_at) VALUES (?, ?, ?)",
            (record.evidence_id, record.model_dump_json(), now_iso()),
        )

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.db.query_one("SELECT record_json FROM evidence WHERE evidence_id=?", (evidence_id,))
        return EvidenceRecord.model_validate_json(row["record_json"]) if row else None

    def list(self) -> list[EvidenceRecord]:
        return [
            EvidenceRecord.model_validate_json(r["record_json"])
            for r in self.db.query("SELECT record_json FROM evidence ORDER BY evidence_id")
        ]


# ---------------------------------------------------------------------------------------
# external_dataset
# ---------------------------------------------------------------------------------------


class ExternalDatasetRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def register(
        self,
        dataset_id: str,
        name: str,
        *,
        source_url: str | None,
        license: str | None,
        license_verified: bool,
        sha256: str | None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO external_datasets
               (dataset_id, name, source_url, license, license_verified, sha256, meta_json, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dataset_id,
                name,
                source_url,
                license,
                int(license_verified),
                sha256,
                _dumps(meta or {}),
                now_iso(),
            ),
        )

    def list(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM external_datasets ORDER BY dataset_id")]


# ---------------------------------------------------------------------------------------
# user_private + audit
# ---------------------------------------------------------------------------------------


class AuditLog:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(
        self,
        action: str,
        namespace: DataNamespace | str,
        *,
        subject: str | None = None,
        actor: str = "api",
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO audit_log (at, actor, action, namespace, subject, detail_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                now_iso(),
                actor,
                action,
                str(getattr(namespace, "value", namespace)),
                subject,
                _dumps(detail or {}),
            ),
        )

    def for_subject(self, subject: str) -> list[dict[str, Any]]:
        return [
            dict(r) for r in self.db.query("SELECT * FROM audit_log WHERE subject=? ORDER BY id", (subject,))
        ]


class UserRepository:
    """Real-user private data (namespace ``user_private``). Demo/pseudonymous users only."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # consent
    def add_consent(self, consent: ConsentRecord) -> None:
        self.db.execute(
            "INSERT INTO user_consent (user_id, purpose, granted, created_at) VALUES (?, ?, ?, ?)",
            (consent.user_id, consent.purpose, int(consent.granted), consent.created_at.isoformat()),
        )

    def consents(self, user_id: str) -> list[ConsentRecord]:
        rows = self.db.query("SELECT * FROM user_consent WHERE user_id=? ORDER BY id", (user_id,))
        return [
            ConsentRecord(
                user_id=r["user_id"],
                purpose=r["purpose"],
                granted=bool(r["granted"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def has_consent(self, user_id: str, purposes: Sequence[str] | None = None) -> bool:
        """Latest record per purpose wins; True if any listed purpose is currently granted."""
        latest: dict[str, bool] = {}
        for c in self.consents(user_id):
            latest[c.purpose] = c.granted
        if purposes is None:
            return any(latest.values())
        return any(latest.get(p, False) for p in purposes)

    # decisions / scenarios / evidence
    def save_decision(self, decision_id: str, request: DecisionRequest) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO user_decisions (decision_id, user_id, request_json, created_at) VALUES (?, ?, ?, ?)",
            (decision_id, request.user_id, request.model_dump_json(), now_iso()),
        )

    def save_scenarios(self, user_id: str, scenarios: Sequence[Scenario]) -> None:
        rows = [(s.scenario_id, s.decision_id, user_id, s.model_dump_json(), now_iso()) for s in scenarios]
        self.db.executemany(
            "INSERT OR REPLACE INTO user_scenarios (scenario_id, decision_id, user_id, scenario_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def save_evidence(self, user_id: str, decision_id: str, records: Sequence[EvidenceRecord]) -> None:
        rows = [(r.evidence_id, decision_id, user_id, r.model_dump_json(), now_iso()) for r in records]
        self.db.executemany(
            "INSERT OR REPLACE INTO user_evidence (evidence_id, decision_id, user_id, record_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        row = self.db.query_one("SELECT record_json FROM user_evidence WHERE evidence_id=?", (evidence_id,))
        return EvidenceRecord.model_validate_json(row["record_json"]) if row else None

    def decisions(self, user_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self.db.query(
            "SELECT * FROM user_decisions WHERE user_id=? ORDER BY created_at", (user_id,)
        ):
            scen = self.db.query(
                "SELECT scenario_json FROM user_scenarios WHERE decision_id=? ORDER BY scenario_id",
                (r["decision_id"],),
            )
            out.append(
                {
                    "decision_id": r["decision_id"],
                    "created_at": r["created_at"],
                    "request": json.loads(r["request_json"]),
                    "scenarios": [json.loads(s["scenario_json"]) for s in scen],
                }
            )
        return out

    def add_reflection(self, reflection: Reflection) -> None:
        self.db.execute(
            "INSERT INTO user_reflections (user_id, decision_id, reflection_json, created_at) VALUES (?, ?, ?, ?)",
            (
                reflection.user_id,
                reflection.decision_id,
                reflection.model_dump_json(),
                reflection.created_at.isoformat(),
            ),
        )

    def reflections(self, user_id: str) -> list[Reflection]:
        rows = self.db.query(
            "SELECT reflection_json FROM user_reflections WHERE user_id=? ORDER BY id", (user_id,)
        )
        return [Reflection.model_validate_json(r["reflection_json"]) for r in rows]

    # export / delete (spec §17 step 10)
    def export(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for table in USER_PRIVATE_TABLES:
            out[table] = [
                dict(r) for r in self.db.query(f"SELECT * FROM {table} WHERE user_id=?", (user_id,))
            ]
        return out

    def delete(self, user_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.db.transaction() as conn:
            for table in USER_PRIVATE_TABLES:
                cur = conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                counts[table] = int(cur.rowcount)
        return counts
