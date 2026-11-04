from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from _fake_world import FakeWorld
from conftest import small_config
from fastapi.testclient import TestClient

from hpe.contracts.registry import get_factory
from hpe.contracts.schemas import (
    ArmSummary,
    EffectEstimate,
    ExperimentDefinition,
    ExperimentResult,
    JobStatus,
    OutcomeObservation,
    SimulationConfig,
)
from hpe.storage import JsonlGzSink, RunDirectory, export_jsonl
from hpe_api.experiment_service import preregistration_hash

REQUIRED_PATHS = {
    "/health",
    "/runs",
    "/runs/{run_id}",
    "/runs/{run_id}/step",
    "/runs/{run_id}/control",
    "/runs/{run_id}/agents",
    "/runs/{run_id}/agents/{agent_id}",
    "/runs/{run_id}/events",
    "/runs/{run_id}/frames",
    "/runs/{run_id}/timeseries",
    "/runs/{run_id}/download",
    "/experiments",
    "/experiments/{experiment_id}/run",
    "/experiments/{experiment_id}/results",
    "/research/metrics",
    "/research/claims",
    "/benchmarks",
    "/scenarios",
    "/decisions/scenarios",
    "/decisions/reflect",
    "/evidence",
    "/evidence/{evidence_id}",
    "/privacy/consent",
    "/privacy/export",
    "/privacy/delete",
    "/journey/{user_id}",
    "/worlds",
    "/worlds/{run_id}",
    "/worlds/{run_id}/run",
    "/worlds/{run_id}/agents",
    "/agents/{agent_id}",
    "/agents/{agent_id}/timeline",
}


def _definition(exp_id: str = "exp-test") -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "title": "Tutoring access (test)",
        "hypothesis": "h",
        "falsification_criterion": "f",
        "arms": [
            {"name": "control", "policy_class": "random"},
            {"name": "treatment", "policy_class": "random"},
        ],
        "control_arm": "control",
        "seeds": [0, 1],
        "base_config": {"seed": 0, "n_agents": 4, "years": 1},
        "analysis": {
            "primary_metric": "acc",
            "primary_metric_definition": "d",
            "comparison": "treatment - control",
        },
    }


def _result(defn: ExperimentDefinition) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=defn.experiment_id,
        preregistration_hash=defn.preregistration_hash or "",
        status="inconclusive",
        status_reason="test result written by an injected runner",
        arms=[ArmSummary(arm="control", n_runs=2), ArmSummary(arm="treatment", n_runs=2)],
        effects=[
            EffectEstimate(
                metric="acc",
                contrast="treatment - control",
                estimate=0.0,
                ci_low=-0.1,
                ci_high=0.1,
                n_units=2,
                method="paired_bootstrap",
            )
        ],
        run_ids=["r1", "r2"],
    )


def test_openapi_has_every_contract_path(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert REQUIRED_PATHS <= paths, REQUIRED_PATHS - paths


def test_health(client: TestClient) -> None:
    h = client.get("/health").json()
    assert (
        h["status"] in ("ok", "degraded") and h["contract_version"] == "0.1.0" and h["llm_provider"] == "none"
    )
    assert h["offline_demo_runs"] == 0


def test_experiment_create_run_501_without_runner(client: TestClient) -> None:
    r = client.post("/experiments", json={"definition": _definition()})
    assert r.status_code == 201, r.text
    info = r.json()
    defn = ExperimentDefinition.model_validate(info["definition"])
    assert defn.preregistration_hash == preregistration_hash(defn)
    assert info["status"] == "queued" and info["result"] is None
    assert client.post("/experiments", json={"definition": _definition()}).status_code == 409
    run = client.post("/experiments/exp-test/run")
    assert run.status_code in (200, 501)
    if run.status_code == 501:
        assert "not available" in run.json()["detail"]
    assert client.get("/experiments/missing/results").status_code == 404


def test_experiment_runs_with_injected_runner(client_factory) -> None:
    def resolver():
        def runner(defn: ExperimentDefinition, out_dir: Path) -> ExperimentResult:
            return _result(defn)

        return runner

    c: TestClient = client_factory(app_kwargs={"experiment_runner_resolver": resolver})
    c.post("/experiments", json={"definition": _definition("exp-run")})
    assert c.post("/experiments/exp-run/run").status_code == 200
    deadline = time.monotonic() + 10
    info: dict[str, Any] = {}
    while time.monotonic() < deadline:
        info = c.get("/experiments/exp-run/results").json()
        if info["status"] == JobStatus.COMPLETED.value:
            break
        time.sleep(0.02)
    assert info["status"] == "completed" and info["result"]["status"] == "inconclusive"
    assert info["result"]["preregistration_hash"] == info["definition"]["preregistration_hash"]
    metrics = c.get("/research/metrics").json()
    assert metrics["experiments"][0]["primary_effects"][0]["metric"] == "acc"


def _write_stored_run(reports: Path, run_id: str = "demo-run-1") -> None:
    cfg = SimulationConfig.model_validate(small_config(years=1.0))
    run_dir = reports / "runs" / run_id
    sink = JsonlGzSink(run_dir)
    world = FakeWorld(cfg, get_factory("random"), sink=sink, run_id=run_id)
    while not world.done:
        world.step(4)
    sink.close()
    rd = RunDirectory(run_dir)
    rd.write_agents(world.agents())
    rd.write_edges(world.edges())
    rd.write_map(world.world_map())
    rd.write_outcomes(world.outcomes())
    m = world.manifest().model_copy(
        update={"status": JobStatus.COMPLETED, "event_log_sha256": sink.sha256, "n_events": sink.n_events}
    )
    rd.write_manifest(m)


def test_stored_runs_are_served_as_real_outputs(client_factory, tmp_path: Path) -> None:
    _write_stored_run(tmp_path / "reports")
    c: TestClient = client_factory()
    assert c.get("/health").json()["offline_demo_runs"] == 1
    runs = {r["run_id"]: r for r in c.get("/runs").json()}
    info = runs["demo-run-1"]
    assert (
        info["source"] == "stored_real_output" and info["mode"] == "stored" and info["status"] == "completed"
    )
    assert c.get("/runs/demo-run-1/agents").json()["total"] == 4
    detail = c.get("/runs/demo-run-1/agents/a0001").json()
    assert detail["skill_curve"] and detail["recent_memories"] and detail["relationships"]
    ev = c.get("/runs/demo-run-1/events", params={"types": "birth"}).json()
    assert ev["total"] == 4
    assert len(c.get("/runs/demo-run-1/timeseries").json()["ticks"]) == 12
    assert c.get("/runs/demo-run-1/frames").json()["ticks"]
    assert c.post("/runs/demo-run-1/step", json={"ticks": 1}).status_code == 404
    assert c.get("/agents/a0003").json()["state"]["agent_id"] == "a0003"
    dl = c.get("/runs/demo-run-1/download")
    assert dl.status_code == 200
    # importing twice is a no-op (sha-checked)
    assert (
        c.get("/runs/demo-run-1/events").json()["total"]
        == ev["total"]
        + c.get(
            "/runs/demo-run-1/events",
            params={"types": "learning_episode,decision,assessment,policy_update,friendship_formed"},
        ).json()["total"]
    )


def test_stored_experiments_claims_benchmarks(client_factory, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    exp_dir = reports / "experiments" / "exp-stored"
    exp_dir.mkdir(parents=True)
    defn = ExperimentDefinition.model_validate(_definition("exp-stored"))
    defn = defn.model_copy(update={"preregistration_hash": preregistration_hash(defn)})
    (exp_dir / "definition.json").write_text(defn.model_dump_json())
    (exp_dir / "result.json").write_text(_result(defn).model_dump_json())
    export_jsonl(
        [OutcomeObservation(metric="acc", value=0.5).model_dump(mode="json")], exp_dir / "outcomes.jsonl.gz"
    )
    (reports / "evidence").mkdir(parents=True)
    (reports / "evidence" / "claims.json").write_text(
        json.dumps(
            {
                "claims": [
                    {"claim_id": "c1", "claim": "x", "status": "not_tested", "evidence_kind": "none"},
                    {"bad": 1},
                ]
            }
        )
    )
    (reports / "scale").mkdir(parents=True)
    (reports / "scale" / "bench_100.json").write_text(
        json.dumps(
            {
                "benchmark_id": "b100",
                "n_agents": 100,
                "years": 80,
                "dt_days": 30,
                "policy_class": "random",
                "wall_time_s": 1.0,
                "peak_rss_mb": None,
                "agent_years_per_s": 8000,
                "ticks_per_s": 900,
                "events": 10,
                "hardware": "test",
                "status": "completed",
            }
        )
    )
    c: TestClient = client_factory()
    exps = {e["experiment_id"]: e for e in c.get("/experiments").json()}
    stored = exps["exp-stored"]
    assert stored["source"] == "stored_real_output" and stored["status"] == "completed"
    assert stored["outcomes_sample"][0]["metric"] == "acc"
    assert c.post("/experiments/exp-stored/run").status_code == 409
    assert c.post("/experiments", json={"definition": _definition("exp-stored")}).status_code == 409
    claims = c.get("/research/claims").json()
    assert [x["claim_id"] for x in claims] == ["c1"]  # invalid record skipped, nothing invented
    assert c.get("/benchmarks").json()[0]["benchmark_id"] == "b100"
    m = c.get("/research/metrics").json()
    assert m["claims"] == 1 and m["benchmarks"] == 1


def test_claims_and_benchmarks_empty_when_absent(client: TestClient) -> None:
    assert client.get("/research/claims").json() == []
    assert client.get("/benchmarks").json() == []


def test_rate_limit_returns_429(client_factory) -> None:
    c: TestClient = client_factory(rate_limit_per_minute=0.001, rate_limit_burst=2)
    codes = [c.post("/runs", json={"config": small_config(years=0.2)}).status_code for _ in range(3)]
    assert codes[:2] == [201, 201] and codes[2] == 429
    r = c.post("/runs", json={"config": small_config(years=0.2)})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_cors_header(client: TestClient) -> None:
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
