from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from hpe.contracts.schemas import (
    AgentFrame,
    ConsentRecord,
    DecisionRequest,
    EventType,
    ExperienceEvent,
    ExperimentDefinition,
    Provenance,
    Reflection,
    RunManifest,
    SimulationConfig,
    WorldMap,
    WorldTick,
)
from hpe.storage import (
    USER_PRIVATE_TABLES,
    AuditLog,
    CheckpointStore,
    Database,
    EventRepository,
    ExperimentRepository,
    ExternalDatasetRepository,
    JsonlGzSink,
    RunDirectory,
    RunRepository,
    SqliteSink,
    StoredRunLoader,
    TeeSink,
    TickFrameIndex,
    TickRepository,
    UserRepository,
    export_jsonl,
    export_parquet,
    parquet_available,
    sha256_of_gz_lines,
)


def _event(
    seq: int,
    tick: int,
    agent: str | None = "a1",
    type_: EventType = EventType.LEARNING_EPISODE,
    others: list[str] | None = None,
) -> ExperienceEvent:
    return ExperienceEvent(
        seq=seq,
        run_id="r1",
        tick=tick,
        day=tick * 30,
        type=type_,
        agent_id=agent,
        other_ids=others or [],
        payload={"family": "sort", "x": seq * 0.1},
        provenance=Provenance(source="world", mechanism="test"),
    )


def _tick(t: int, frames: bool = True) -> WorldTick:
    fr = (
        [
            AgentFrame(
                id="a1",
                x=1.0,
                y=2.0,
                place_id=None,
                activity="rest",
                age=t / 12,
                alive=True,
                household_id="h1",
            )
        ]
        if frames
        else None
    )
    return WorldTick(run_id="r1", tick=t, day=t * 30, year=t / 12, population=1, frames=fr)


def test_migrations_create_namespaced_tables_and_are_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    assert db.schema_version() == 1
    tables = set(db.tables())
    expected = {
        "schema_migrations",
        "sim_runs",
        "sim_events",
        "sim_ticks",
        "sim_checkpoints",
        "experiments",
        "experiment_results",
        "evidence",
        "external_datasets",
        "audit_log",
        *USER_PRIVATE_TABLES,
    }
    assert expected <= tables
    assert db.migrate() == []
    db.close()
    db2 = Database(tmp_path / "x.sqlite3")
    assert db2.migrate() == [] and db2.schema_version() == 1
    rows = db2.query("SELECT version, name FROM schema_migrations")
    assert [(r["version"], r["name"]) for r in rows] == [(1, "initial_namespaced_schema")]


def test_transaction_rolls_back(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    with pytest.raises(RuntimeError), db.transaction() as conn:
        conn.execute("INSERT INTO evidence (evidence_id, record_json, created_at) VALUES ('e', '{}', 'now')")
        raise RuntimeError("boom")
    assert db.query("SELECT * FROM evidence") == []


def test_jsonl_sink_roundtrip_sha_and_determinism(tmp_path: Path) -> None:
    events = [_event(i, i // 3) for i in range(30)]
    shas = []
    for name in ("a", "b"):
        sink = JsonlGzSink(tmp_path / name)
        for e in events:
            sink.emit(e)
        for t in range(1, 4):
            sink.emit_tick(_tick(t, frames=t != 2))
        sink.close()
        assert sink.n_events == 30 and sink.n_ticks == 3 and sink.n_frame_sets == 2
        assert sink.sha256 == sha256_of_gz_lines(tmp_path / name / "events.jsonl.gz")
        shas.append(sink.sha256)
    assert shas[0] == shas[1]
    assert (tmp_path / "a" / "events.jsonl.gz").read_bytes() == (
        tmp_path / "b" / "events.jsonl.gz"
    ).read_bytes()
    lines = gzip.decompress((tmp_path / "a" / "events.jsonl.gz").read_bytes()).decode().splitlines()
    assert [ExperienceEvent.model_validate_json(x) for x in lines] == events
    ticks = gzip.decompress((tmp_path / "a" / "ticks.jsonl.gz").read_bytes()).decode().splitlines()
    assert all(json.loads(t)["frames"] is None for t in ticks)
    expected = hashlib.sha256(b"".join(e.model_dump_json().encode() + b"\n" for e in events)).hexdigest()
    assert shas[0] == expected


def test_sqlite_sink_and_event_queries(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    sink = SqliteSink(db, "r1", batch_size=7)
    for i in range(20):
        sink.emit(_event(i, i, agent="a1" if i % 2 else "a2"))
    sink.emit(_event(20, 20, agent="a3", type_=EventType.FRIENDSHIP_FORMED, others=["a1"]))
    sink.emit_tick(_tick(1))
    sink.close()
    repo = EventRepository(db)
    assert repo.count("r1") == 21
    items, total = repo.query("r1", agent_id="a1", limit=3)
    assert total == 11 and len(items) == 3 and all(e.agent_id == "a1" for e in items)
    fr, n = repo.query("r1", agent_id="a1", types=["friendship_formed"])
    assert n == 1 and fr[0].other_ids == ["a1"]
    rng, _ = repo.query("r1", tick_from=5, tick_to=7)
    assert [e.tick for e in rng] == [5, 6, 7]
    desc, _ = repo.query("r1", limit=1, descending=True)
    assert desc[0].seq == 20
    page2, _ = repo.query("r1", offset=10, limit=5)
    assert [e.seq for e in page2] == [10, 11, 12, 13, 14]
    assert len(list(repo.iter_all("r1", batch=4))) == 21
    # immutability: re-inserting the same seq is ignored, not overwritten
    repo.append_many([_event(0, 99)], run_id="r1")
    assert repo.query("r1", limit=1)[0][0].tick == 0
    assert TickRepository(db).list("r1")[0].frames is None


def test_tick_frame_index_and_tee(tmp_path: Path) -> None:
    idx = TickFrameIndex(max_frame_sets=4)
    tee = TeeSink([idx, JsonlGzSink(tmp_path / "t")])
    for t in range(1, 11):
        tee.emit_tick(_tick(t))
    tee.close()
    assert [t.tick for t in idx.ticks()] == list(range(1, 11))
    assert len(idx.frame_ticks()) <= 5 and 10 in idx.frame_ticks()
    assert idx.frames_at(10)[0].id == "a1" and idx.latest() is not None and idx.latest().tick == 10  # type: ignore[union-attr]


def test_checkpoint_store(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "ck")
    import numpy as np

    p, sha = store.save(12, {"a": np.arange(3), "b": {"c": 1.5}})
    assert p.exists() and len(sha) == 64
    store.save(24, {"a": [9]})
    assert store.ticks() == [12, 24]
    assert store.load(12) == {"a": [0, 1, 2], "b": {"c": 1.5}}
    assert store.load() == {"a": [9]}
    with pytest.raises(FileNotFoundError):
        CheckpointStore(tmp_path / "empty").load()


def test_run_directory_and_loader(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    cfg = SimulationConfig(seed=1, n_agents=1, years=1)
    for base, rid in ((reports / "runs", "demo1"), (reports / "experiments" / "e1" / "runs", "exp-run-1")):
        sink = JsonlGzSink(base / rid)
        sink.emit(_event(0, 0))
        sink.emit_tick(_tick(1))
        sink.close()
        rd = RunDirectory(base / rid)
        rd.write_map(WorldMap(width=10, height=10, places=[]))
        rd.write_agents([])
        rd.write_edges([])
        rd.write_outcomes([])
        rd.write_manifest(
            RunManifest(
                run_id=rid,
                config=cfg,
                config_hash="h",
                policy_class=cfg.policy_class,
                event_log_sha256=sink.sha256,
            )
        )
    loader = StoredRunLoader(reports)
    assert loader.run_ids() == ["demo1"] and loader.count_demo_runs() == 1
    assert loader.run_ids(include_experiment_runs=True) == ["demo1", "exp-run-1"]
    run = loader.get("demo1")
    assert run is not None and run.manifest.run_id == "demo1"
    assert run.events_sha256() == run.manifest.event_log_sha256
    assert (
        [t.tick for t in run.ticks()] == [1] and run.frame_ticks() == [1] and run.frames_at(1)[0].id == "a1"
    )
    assert run.world_map().width == 10 and run.agents() == [] and run.outcomes() == []
    assert loader.get("missing") is None


def test_repositories_runs_experiments_datasets(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    runs = RunRepository(db)
    runs.upsert("r1", status="queued", mode="background", source="live", config_json="{}")
    runs.set_status("r1", "completed")
    runs.mark_imported("r1", "abc")
    row = runs.get("r1")
    assert row is not None and row["status"] == "completed" and runs.imported_sha("r1") == "abc"
    defn = ExperimentDefinition.model_validate(
        {
            "experiment_id": "e",
            "title": "t",
            "hypothesis": "h",
            "falsification_criterion": "f",
            "arms": [{"name": "c", "policy_class": "random"}],
            "control_arm": "c",
            "seeds": [0],
            "base_config": {"seed": 0},
            "analysis": {"primary_metric": "m", "primary_metric_definition": "d", "comparison": "c"},
            "preregistration_hash": "ph",
        }
    )
    exps = ExperimentRepository(db)
    exps.create(defn)
    assert exps.get("e")["preregistration_hash"] == "ph"  # type: ignore[index]
    ds = ExternalDatasetRepository(db)
    ds.register(
        "dfe",
        "DfE-DB",
        source_url="https://example.invalid",
        license="unknown",
        license_verified=False,
        sha256=None,
    )
    assert ds.list()[0]["license_verified"] == 0


def test_user_repository_consent_export_delete(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    users = UserRepository(db)
    audit = AuditLog(db)
    users.add_consent(ConsentRecord(user_id="u", purpose="future_self_demo", granted=True))
    assert users.has_consent("u") and users.has_consent("u", ["future_self_demo"])
    users.add_consent(ConsentRecord(user_id="u", purpose="future_self_demo", granted=False))
    assert not users.has_consent("u", ["future_self_demo"])
    req = DecisionRequest.model_validate(
        {
            "user_id": "u",
            "consent_given": True,
            "question": "q",
            "options": [{"option_id": "a", "label": "A"}, {"option_id": "b", "label": "B"}],
            "values": [],
        }
    )
    users.save_decision("d1", req)
    users.add_reflection(Reflection(user_id="u", decision_id="d1"))
    users.add_consent(ConsentRecord(user_id="other", purpose="future_self_demo", granted=True))
    audit.record("x", "user_private", subject="u")
    exp = users.export("u")
    assert (
        len(exp["user_decisions"]) == 1
        and len(exp["user_reflections"]) == 1
        and len(exp["user_consent"]) == 2
    )
    counts = users.delete("u")
    assert counts == {
        "user_consent": 2,
        "user_decisions": 1,
        "user_scenarios": 0,
        "user_evidence": 0,
        "user_reflections": 1,
    }
    assert all(not v for v in users.export("u").values())
    assert users.has_consent("other")  # other users untouched
    assert audit.for_subject("u")[0]["action"] == "x"


def test_exports(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": {"c": [1, 2]}}, {"a": 2, "b": {}}]
    assert export_jsonl(rows, tmp_path / "x.jsonl.gz") == 2
    assert [
        json.loads(x) for x in gzip.decompress((tmp_path / "x.jsonl.gz").read_bytes()).decode().splitlines()
    ] == rows
    assert export_jsonl(['{"raw": true}'], tmp_path / "y.jsonl") == 1
    if parquet_available():
        import pyarrow.parquet as pq

        assert export_parquet(rows, tmp_path / "x.parquet") == 2
        t = pq.read_table(tmp_path / "x.parquet").to_pylist()
        assert t[0]["a"] == 1 and json.loads(t[0]["b"]) == {"c": [1, 2]}
