from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from hpe.contracts.schemas import ScenarioResponse
from hpe.futures import check_scenario


def spec_request(
    user_id: str = "demo-user-1", consent: bool = True, question: str | None = None
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "consent_given": consent,
        "question": question
        or "Should I take a higher-paying job, continue in my current role, or attempt a startup?",
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


def test_scenarios_flow_and_journey(client: TestClient) -> None:
    r = client.post("/scenarios", json=spec_request())
    assert r.status_code == 200, r.text
    resp = ScenarioResponse.model_validate(r.json())
    assert len(resp.scenarios) == 3 and all(s.label == "illustrative scenario" for s in resp.scenarios)
    assert all(not check_scenario(s) for s in resp.scenarios)
    assert resp.simulation_separation_note.startswith("No result from the synthetic agent society")
    alias = client.post("/decisions/scenarios", json=spec_request())
    assert alias.status_code == 200

    # user evidence is retrievable only with the owning user id
    calc_id = next(c.evidence_id for s in resp.scenarios for c in s.calculations)
    assert client.get(f"/evidence/{calc_id}").status_code == 404
    assert (
        client.get(f"/evidence/{calc_id}", params={"user_id": "demo-user-1"}).json()["source_type"]
        == "deterministic_calculation"
    )
    assert client.get(f"/evidence/{calc_id}", params={"user_id": "someone-else"}).status_code == 404
    lib = client.get("/evidence").json()
    assert {e["evidence_id"] for e in lib} >= {
        "ext-affective-forecasting-2005",
        "method-no-individual-forecast",
    }
    assert all(e["source_type"] != "simulated_outcome" for e in lib)
    assert client.get("/evidence/ext-affective-forecasting-2005").json()["validation_status"] == "unvalidated"

    refl = {
        "user_id": "demo-user-1",
        "decision_id": resp.decision_id,
        "chosen_option_id": "job",
        "what_happened": "Talked to two people.",
    }
    assert client.post("/decisions/reflect", json=refl).status_code == 200
    assert client.post("/decisions/reflect", json={**refl, "decision_id": "nope"}).status_code == 404
    j = client.get("/journey/demo-user-1").json()
    assert len(j["decisions"]) == 2 and len(j["reflections"]) == 1
    assert len(j["decisions"][0]["scenarios"]) == 3


def test_consent_required(client: TestClient) -> None:
    r = client.post("/scenarios", json=spec_request(consent=False))
    assert r.status_code == 400 and "consent" in r.json()["detail"]
    refl = {"user_id": "no-consent-user", "decision_id": "d1"}
    assert client.post("/decisions/reflect", json=refl).status_code == 403


def test_crisis_routing_returns_no_scenarios(client: TestClient) -> None:
    r = client.post("/scenarios", json=spec_request(question="I want to end my life, should I quit my job?"))
    body = r.json()
    assert r.status_code == 200 and body["scenarios"] == [] and body["safety"]["category"] == "crisis"
    assert "emergency" in body["safety"]["referral"]


def test_high_stakes_adds_referral(client: TestClient) -> None:
    r = client.post("/scenarios", json=spec_request(question="Should I take out a loan to start a business?"))
    body = r.json()
    assert body["safety"]["category"] == "financial" and body["safety"]["referral"]
    assert len(body["scenarios"]) == 3


def test_privacy_consent_export_delete(client: TestClient) -> None:
    uid = "demo-privacy"
    assert (
        client.post(
            "/privacy/consent", json={"user_id": uid, "purpose": "reflection_storage", "granted": True}
        ).status_code
        == 200
    )
    resp = client.post("/scenarios", json=spec_request(user_id=uid)).json()
    client.post("/decisions/reflect", json={"user_id": uid, "decision_id": resp["decision_id"]})
    exp = client.post("/privacy/export", json={"user_id": uid}).json()
    rec = exp["records"]
    assert (
        len(rec["user_decisions"]) == 1
        and len(rec["user_scenarios"]) == 3
        and len(rec["user_reflections"]) == 1
    )
    assert len(rec["user_consent"]) >= 1 and rec["user_evidence"] and rec["audit_log"]
    deleted = client.post("/privacy/delete", json={"user_id": uid}).json()["deleted"]
    assert (
        deleted["user_decisions"] == 1 and deleted["user_scenarios"] == 3 and deleted["user_reflections"] == 1
    )
    after = client.post("/privacy/export", json={"user_id": uid}).json()["records"]
    assert all(
        not after[t]
        for t in ("user_consent", "user_decisions", "user_scenarios", "user_evidence", "user_reflections")
    )
    # revocation is respected: no consent -> reflections refused
    assert (
        client.post(
            "/decisions/reflect", json={"user_id": uid, "decision_id": resp["decision_id"]}
        ).status_code
        == 403
    )


def test_consent_revocation_latest_wins(client: TestClient) -> None:
    uid = "demo-revoke"
    client.post("/privacy/consent", json={"user_id": uid, "purpose": "reflection_storage", "granted": True})
    client.post("/privacy/consent", json={"user_id": uid, "purpose": "reflection_storage", "granted": False})
    assert client.post("/decisions/reflect", json={"user_id": uid, "decision_id": "d"}).status_code == 403


def test_scenario_rate_limit(client_factory) -> None:
    c: TestClient = client_factory(rate_limit_per_minute=0.001, rate_limit_burst=1)
    assert c.post("/scenarios", json=spec_request()).status_code == 200
    r = c.post("/scenarios", json=spec_request())
    assert r.status_code == 429 and "Retry-After" in r.headers
