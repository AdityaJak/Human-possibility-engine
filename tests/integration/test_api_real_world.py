"""Cross-lane integration: the API assembles agent detail from a REAL hpe.world.World run."""

import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HPE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HPE_REPORTS_DIR", str(tmp_path / "reports"))
    from hpe_api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_real_world_run_exposes_skill_curves_and_parameter_history(client):
    cfg = {
        "seed": 3,
        "n_agents": 12,
        "years": 3,
        "policy_class": "developmental",
        "log": {"frames_every_ticks": 6, "checkpoint_every_years": None},
    }
    r = client.post("/runs", json={"config": cfg, "mode": "background"})
    assert r.status_code in (200, 201, 202), r.text
    run_id = r.json()["run_id"]
    for _ in range(300):
        info = client.get(f"/runs/{run_id}").json()
        if info["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert info["status"] == "completed", info
    agents = client.get(f"/runs/{run_id}/agents", params={"limit": 50}).json()["items"]
    adult = next(a for a in agents if a["age_years"] >= 6)
    detail = client.get(f"/runs/{run_id}/agents/{adult['agent_id']}").json()
    families = {p["family"] for p in detail["skill_curve"]}
    assert {"sort", "compare", "signal", "barter"} <= families, families
    assert all(p["split"] == "transfer" for p in detail["skill_curve"] if p["family"] == "barter")
    assert len(detail["parameter_history"]) >= 2
    assert all(0.0 <= p["accuracy"] <= 1.0 for p in detail["skill_curve"])
