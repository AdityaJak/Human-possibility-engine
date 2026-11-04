from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from conftest import small_config, wait_for
from fastapi.testclient import TestClient

from hpe.contracts.api import AgentDetail, EventPage, FramesResponse, RunInfo, TimeSeriesResponse
from hpe.contracts.schemas import WorldTick
from hpe.storage import sha256_of_gz_lines


def _create(client: TestClient, mode: str = "background", **cfg: object) -> RunInfo:
    r = client.post("/runs", json={"config": small_config(**cfg), "mode": mode})
    assert r.status_code == 201, r.text
    return RunInfo.model_validate(r.json())


def test_background_run_completes_and_persists(client: TestClient, tmp_path: Path) -> None:
    info = _create(client)
    done = RunInfo.model_validate(wait_for(client, info.run_id))
    assert done.progress == 1.0 and done.source == "live" and done.mode == "background"
    m = done.manifest
    assert m is not None and m.status.value == "completed"
    run_dir = tmp_path / "data" / "runs" / info.run_id
    assert m.event_log_sha256 == sha256_of_gz_lines(run_dir / "events.jsonl.gz")
    for name in (
        "manifest.json",
        "events.jsonl.gz",
        "ticks.jsonl.gz",
        "frames.jsonl.gz",
        "final_agents.json",
        "edges.json",
        "map.json",
        "outcomes.jsonl.gz",
    ):
        assert (run_dir / name).exists(), name
    # 2 years at 30-day ticks, checkpoint every 10 years => no checkpoint; n_events counted
    assert m.n_events > 0


def test_background_checkpoints_resume_identically(client: TestClient, tmp_path: Path) -> None:
    from _fake_world import FakeWorld

    from hpe.contracts.registry import get_factory
    from hpe.contracts.schemas import SimulationConfig
    from hpe.storage import CheckpointStore

    cfg = small_config(log={"checkpoint_every_years": 0.5})
    info = _create(client, **cfg)
    done = wait_for(client, info.run_id)
    cks = done["manifest"]["checkpoints"]
    assert len(cks) >= 3 and all(c.startswith("checkpoints/") for c in cks)
    store = CheckpointStore(tmp_path / "data" / "runs" / info.run_id / "checkpoints")
    snap = store.load(store.ticks()[0])
    # resuming from the checkpoint reproduces the uninterrupted world's final state
    config = SimulationConfig.model_validate(cfg)
    resumed = FakeWorld(config, get_factory("random"))
    resumed.load_snapshot(snap)
    while not resumed.done:
        resumed.step(1)
    full = FakeWorld(config, get_factory("random"))
    while not full.done:
        full.step(1)
    assert resumed.snapshot() == full.snapshot()


def test_agents_detail_events_frames_timeseries(client: TestClient) -> None:
    info = _create(client)
    wait_for(client, info.run_id)
    rid = info.run_id
    page = client.get(f"/runs/{rid}/agents").json()
    assert page["total"] == 4 and page["items"][0]["agent_id"] == "a0000"
    assert page["items"][0]["mean_measured_accuracy"] is not None
    assert client.get(f"/runs/{rid}/agents", params={"q": "a0003"}).json()["total"] == 1

    detail = AgentDetail.model_validate(client.get(f"/runs/{rid}/agents/a0000").json())
    assert detail.skill_curve, "skill curve must come from ASSESSMENT events"
    fams = {p.family for p in detail.skill_curve}
    assert "barter" in fams and any(p.split == "transfer" for p in detail.skill_curve if p.family == "barter")
    first = [p for p in detail.skill_curve if p.family == "sort"]
    assert first[-1].accuracy > first[0].accuracy
    assert detail.parameter_history and detail.parameter_history[-1].summary["n_updates"] > 0
    assert [e.dst for e in detail.relationships] == ["a0001"]
    assert detail.latest_decision_explanation and detail.latest_decision_explanation["action"] == "study"
    assert detail.recent_memories, "live runs expose policy memories or POLICY_UPDATE memories"
    types = {e.type.value for e in detail.timeline}
    assert {"birth", "friendship_formed", "learning_episode"} <= types
    assert client.get(f"/runs/{rid}/agents/nope").status_code == 404

    ev = EventPage.model_validate(client.get(f"/runs/{rid}/events", params={"limit": 5}).json())
    assert (
        ev.total > 5 and len(ev.items) == 5 and [e.seq for e in ev.items] == sorted(e.seq for e in ev.items)
    )
    ev2 = EventPage.model_validate(client.get(f"/runs/{rid}/events", params={"limit": 5, "offset": 5}).json())
    assert ev2.items[0].seq > ev.items[-1].seq
    only = client.get(f"/runs/{rid}/events", params={"types": "assessment,birth", "agent_id": "a0001"}).json()
    assert only["total"] > 0 and {e["type"] for e in only["items"]} <= {"assessment", "birth"}
    # friendship event lists a0001 in other_ids, so it is part of a0001's events
    fr = client.get(f"/runs/{rid}/events", params={"types": "friendship_formed", "agent_id": "a0001"}).json()
    assert fr["total"] == 1
    ranged = client.get(f"/runs/{rid}/events", params={"tick_from": 3, "tick_to": 3}).json()
    assert all(e["tick"] == 3 for e in ranged["items"])

    frames = FramesResponse.model_validate(
        client.get(f"/runs/{rid}/frames", params={"every": 2, "limit": 4}).json()
    )
    assert len(frames.ticks) == 4 and len(frames.frames[0]) == 4 and frames.map.places
    ts = TimeSeriesResponse.model_validate(client.get(f"/runs/{rid}/timeseries").json())
    assert [t.tick for t in ts.ticks] == list(range(1, 25))
    assert all(t.frames is None for t in ts.ticks)


def test_download_is_manifest_plus_events(client: TestClient) -> None:
    info = _create(client)
    done = wait_for(client, info.run_id)
    r = client.get(f"/runs/{info.run_id}/download")
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    lines = gzip.decompress(r.content).decode().splitlines()
    head = json.loads(lines[0])
    assert head["type"] == "manifest" and head["manifest"]["run_id"] == info.run_id
    events = lines[1:]
    assert len(events) == done["manifest"]["n_events"]
    body = ("\n".join(events) + "\n").encode()
    # events in the download are byte-identical to the logged event JSONL
    assert hashlib.sha256(body).hexdigest() == done["manifest"]["event_log_sha256"]


def test_interactive_step_pause_resume_reset_cancel(client: TestClient) -> None:
    info = _create(client, mode="interactive", years=1.0)
    assert info.status.value == "paused" and info.mode == "interactive"
    rid = info.run_id
    t = WorldTick.model_validate(
        client.post(f"/runs/{rid}/step", json={"ticks": 3, "include_frames": True}).json()
    )
    assert t.tick == 3 and t.frames and len(t.frames) == 4
    assert client.get(f"/runs/{rid}").json()["current_tick"] == 3
    # frames recorded for map playback
    assert 3 in client.get(f"/runs/{rid}/frames").json()["ticks"]

    r = client.post(f"/runs/{rid}/control", json={"action": "resume"})
    assert r.status_code == 200
    done = wait_for(client, rid, ("completed",))
    assert done["current_tick"] == 12
    assert client.post(f"/runs/{rid}/step", json={"ticks": 1}).status_code == 409

    r = client.post(f"/runs/{rid}/control", json={"action": "reset"})
    assert r.status_code == 200 and r.json()["current_tick"] == 0 and r.json()["status"] == "paused"
    assert client.get(f"/runs/{rid}/events").json()["total"] == 4  # births only after reset
    assert client.post(f"/runs/{rid}/step", json={"ticks": 2}).json()["tick"] == 2
    assert client.post(f"/runs/{rid}/control", json={"action": "pause"}).json()["status"] == "paused"
    r = client.post(f"/runs/{rid}/control", json={"action": "cancel"})
    assert r.json()["status"] == "cancelled"
    assert client.post(f"/runs/{rid}/step", json={"ticks": 1}).status_code == 409


def test_background_cannot_be_stepped_or_reset(client: TestClient) -> None:
    info = _create(client)
    assert client.post(f"/runs/{info.run_id}/step", json={"ticks": 1}).status_code == 409
    assert client.post(f"/runs/{info.run_id}/control", json={"action": "reset"}).status_code == 409
    wait_for(client, info.run_id)


def test_quota_and_validation_errors(client: TestClient) -> None:
    r = client.post("/runs", json={"config": small_config(n_agents=51)})
    assert r.status_code == 400 and "HPE_MAX_AGENTS_PER_RUN" in r.json()["detail"]
    r = client.post("/runs", json={"config": small_config(bogus=1)})
    assert r.status_code == 422
    r = client.post("/runs", json={"config": small_config(policy_class="developmental")})
    # developmental is provided by another lane; when it is not installed the API says so explicitly
    assert r.status_code in (201, 400)
    assert client.get("/runs/does-not-exist").status_code == 404


def test_world_engine_unavailable_is_503(client_factory) -> None:
    from hpe_api.errors import Unavailable

    def missing(*_a: object, **_k: object) -> object:
        raise Unavailable("world engine not available in this build")

    c = client_factory(world_factory=missing)
    r = c.post("/runs", json={"config": small_config()})
    assert r.status_code == 503


def test_spec_aliases(client: TestClient) -> None:
    r = client.post("/worlds", json={"config": small_config(years=0.5), "mode": "interactive"})
    assert r.status_code == 201
    rid = r.json()["run_id"]
    assert client.get(f"/worlds/{rid}").json()["run_id"] == rid
    client.post(f"/runs/{rid}/step", json={"ticks": 1})
    assert client.get(f"/worlds/{rid}/agents").json()["total"] == 4
    assert client.post(f"/worlds/{rid}/run").status_code == 200
    wait_for(client, rid)
    assert client.get("/agents/a0002", params={"run_id": rid}).json()["state"]["agent_id"] == "a0002"
    assert client.get("/agents/a0002").status_code == 200  # resolved by searching live runs
    tl = client.get("/agents/a0000/timeline", params={"run_id": rid}).json()
    assert tl["total"] >= 1 and tl["items"][0]["type"] == "birth"


def test_list_runs(client: TestClient) -> None:
    a = _create(client)
    b = _create(client, mode="interactive")
    ids = {r["run_id"] for r in client.get("/runs").json()}
    assert {a.run_id, b.run_id} <= ids
    wait_for(client, a.run_id)
