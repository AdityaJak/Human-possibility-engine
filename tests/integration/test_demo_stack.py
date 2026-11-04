"""Integration: the investor demo flow against the REAL ``hpe.world.World`` (no fakes).

Exercises the same sequence as ``make demo-check`` but in-process with FastAPI's TestClient:
health, stored runs, claims, benchmarks, evidence, a consented Future Self scenario request
(spec §19 example), and an interactive 25-agent society that is stepped and inspected.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hpe_api.main import create_app
from hpe_api.settings import Settings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def spec_19_request(user_id: str = "demo-check-user") -> dict[str, Any]:
    """Spec §19: higher-paying job vs current role vs startup."""
    return {
        "user_id": user_id,
        "consent_given": True,
        "question": "Should I take a higher-paying job, continue in my current role, or attempt a startup?",
        "options": [
            {
                "option_id": "stay",
                "label": "Continue in my current role",
                "kind": "current_path",
                "inputs": {"monthly_income": 5000},
            },
            {
                "option_id": "job",
                "label": "Take the higher-paying job",
                "kind": "new_job",
                "inputs": {"monthly_income": 6200},
            },
            {
                "option_id": "startup",
                "label": "Attempt a startup",
                "kind": "start_project",
                "inputs": {"monthly_income": 0},
            },
        ],
        "values": [
            {"key": "income", "label": "Income", "weight": 0.6},
            {"key": "stability", "label": "Security", "weight": 0.7},
            {"key": "autonomy", "label": "Autonomy", "weight": 0.9},
            {"key": "relationships", "label": "Relationships", "weight": 0.5},
            {"key": "learning", "label": "Growth", "weight": 0.8},
        ],
        "constraints": {"savings": 24000, "monthly_expenses": 3500},
        "horizon_years": 3,
    }


@pytest.fixture
def client(tmp_path: Path) -> Any:
    settings = Settings(
        data_dir=tmp_path / "data",
        reports_dir=ROOT / "reports",
        interactive_tick_delay_s=0.0,
        rate_limit_per_minute=600.0,
        rate_limit_burst=100,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_reports_real_world_engine(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok", body  # "degraded" would mean hpe.world is missing
    assert body["llm_provider"] == "none"


def test_read_endpoints_respond(client: TestClient) -> None:
    for path in ("/runs", "/research/claims", "/benchmarks", "/evidence", "/experiments"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.text)
        assert isinstance(r.json(), list), path


def test_benchmarks_endpoint_serves_committed_records(client: TestClient) -> None:
    committed = sorted(p.stem for p in (ROOT / "reports" / "scale").glob("*.json"))
    served = sorted(r["benchmark_id"] for r in client.get("/benchmarks").json())
    assert served == committed


def test_future_self_spec_19_scenarios(client: TestClient) -> None:
    r = client.post("/scenarios", json=spec_19_request())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["scenarios"]) == 3
    for s in body["scenarios"]:
        assert s["label"] == "illustrative scenario"
        assert "%" not in s["narrative"]
    no_consent = {**spec_19_request(), "consent_given": False}
    assert client.post("/scenarios", json=no_consent).status_code == 400


def test_interactive_society_with_real_world(client: TestClient) -> None:
    cfg = {"seed": 7, "n_agents": 25, "years": 5, "policy_class": "developmental"}
    r = client.post("/runs", json={"config": cfg, "mode": "interactive"})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]

    st = client.post(f"/runs/{run_id}/step", json={"ticks": 3, "include_frames": True})
    assert st.status_code == 200, st.text
    tick = st.json()
    assert tick["tick"] >= 3 and tick["population"] >= 20
    assert tick["frames"] and len(tick["frames"]) == tick["population"]

    agents = client.get(f"/runs/{run_id}/agents").json()
    assert agents["total"] >= 25
    agent_id = agents["items"][0]["agent_id"]
    detail = client.get(f"/runs/{run_id}/agents/{agent_id}")
    assert detail.status_code == 200, detail.text
    d = detail.json()
    assert d["state"]["agent_id"] == agent_id
    assert d["state"]["policy"]["policy_class"] == "developmental"

    frames = client.get(f"/runs/{run_id}/frames")
    assert frames.status_code == 200 and frames.json()["map"]["places"]
    assert client.post(f"/runs/{run_id}/control", json={"action": "pause"}).status_code == 200


def test_background_run_completes_with_real_world(client: TestClient) -> None:
    cfg = {"seed": 3, "n_agents": 10, "years": 1, "policy_class": "static_persona"}
    r = client.post("/runs", json={"config": cfg, "mode": "background"})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    deadline = time.monotonic() + 120
    info: dict[str, Any] = {}
    while time.monotonic() < deadline:
        info = client.get(f"/runs/{run_id}").json()
        if info["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert info.get("status") == "completed", info
    ev = client.get(f"/runs/{run_id}/events", params={"limit": 5}).json()
    assert ev["total"] > 0
    dl = client.get(f"/runs/{run_id}/download")
    assert dl.status_code == 200 and dl.content[:2] == b"\x1f\x8b"  # gzip
