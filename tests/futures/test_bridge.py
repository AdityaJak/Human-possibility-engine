"""The browser bridge must produce exactly what the API produces (same engine, same semantics)."""

import json
import re

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hpe.futures import bridge

SPEC_19_REQUEST = {
    "user_id": "bridge-test",
    "consent_given": True,
    "question": "Should I take a higher-paying job, continue in my current role, or attempt a startup?",
    "options": [
        {
            "option_id": "job",
            "label": "Take the higher-paying job",
            "kind": "new_job",
            "inputs": {"monthly_income": 6000},
        },
        {
            "option_id": "stay",
            "label": "Stay in my current role",
            "kind": "current_path",
            "inputs": {"monthly_income": 5000},
        },
        {
            "option_id": "startup",
            "label": "Start a startup",
            "kind": "start_project",
            "inputs": {"savings": 20000, "monthly_expenses": 3000},
        },
    ],
    "values": [
        {"key": "income", "label": "Income", "weight": 0.8},
        {"key": "autonomy", "label": "Autonomy", "weight": 0.9},
        {"key": "stability", "label": "Stability", "weight": 0.5},
    ],
    "constraints": {"available_hours_per_week": 10},
    "horizon_years": 3,
}

DEC_ID = re.compile(r"\b(dec|sc)-[0-9a-f]{12}\b")  # random ids (decision, scenario)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HPE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HPE_REPORTS_DIR", str(tmp_path / "reports"))
    from hpe_api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def _normalise(obj: dict) -> dict:
    return json.loads(DEC_ID.sub(r"\1-X", json.dumps(obj, sort_keys=True)))


def test_bridge_matches_api_response_for_spec_example(client):
    api = client.post("/scenarios", json=SPEC_19_REQUEST)
    assert api.status_code == 200, api.text
    ours = json.loads(bridge.generate(json.dumps(SPEC_19_REQUEST)))
    assert _normalise(ours) == _normalise(api.json())
    assert len(ours["scenarios"]) == 3
    assert all(s["label"] == "illustrative scenario" for s in ours["scenarios"])


def test_bridge_fixed_decision_id_is_honoured():
    out = json.loads(bridge.generate(json.dumps(SPEC_19_REQUEST), "dec-fixed000001"))
    assert out["decision_id"] == "dec-fixed000001"
    assert all(s["decision_id"] == "dec-fixed000001" for s in out["scenarios"])


def test_bridge_consent_and_validation_errors():
    with pytest.raises(bridge.ConsentRequired, match="consent_given must be true"):
        bridge.generate(json.dumps({**SPEC_19_REQUEST, "consent_given": False}))
    with pytest.raises(ValidationError):
        bridge.generate(json.dumps({**SPEC_19_REQUEST, "options": []}))


def test_bridge_crisis_routing_matches_api(client):
    req = {**SPEC_19_REQUEST, "question": "I don't see the point of going on anymore. Job or startup?"}
    api = client.post("/scenarios", json=req).json()
    ours = json.loads(bridge.generate(json.dumps(req)))
    assert ours["scenarios"] == [] and ours["safety"]["category"] == "crisis"
    assert _normalise(ours) == _normalise(api)


def test_evidence_catalog_matches_api(client):
    api = client.get("/evidence").json()
    ours = json.loads(bridge.evidence_catalog())
    assert sorted(r["evidence_id"] for r in ours) == sorted(r["evidence_id"] for r in api)
