"""HPE_READ_ONLY=1 (public deployments): simulation/experiment starters answer 405, everything else works."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

RUN_CONFIG = {"seed": 1, "n_agents": 5, "years": 1, "policy_class": "random"}


def _scenario_request() -> dict:
    return {
        "user_id": "ro-user",
        "consent_given": True,
        "question": "Take the higher-paying job, stay, or start a startup?",
        "options": [
            {
                "option_id": "a",
                "label": "Higher-paying job",
                "kind": "new_job",
                "inputs": {"monthly_income": 6000},
            },
            {"option_id": "b", "label": "Stay", "kind": "current_path", "inputs": {"monthly_income": 5000}},
        ],
        "values": [{"key": "income", "label": "Income", "weight": 0.7}],
    }


def test_default_is_not_read_only(client: TestClient) -> None:
    assert client.get("/deployment").json()["read_only"] is False
    r = client.post("/runs", json={"config": RUN_CONFIG, "mode": "interactive"})
    assert r.status_code == 201, r.text


def test_read_only_blocks_simulation_and_experiment_starters(
    client_factory: Callable[..., TestClient],
) -> None:
    c = client_factory(read_only=True)
    dep = c.get("/deployment").json()
    assert dep["read_only"] is True and dep["mode"] == "stored_real_outputs_only"
    assert c.get("/research/metrics").json()["read_only"] is True
    assert c.get("/health").status_code == 200

    blocked = [
        ("/runs", {"config": RUN_CONFIG, "mode": "interactive"}),
        ("/worlds", {"config": RUN_CONFIG, "mode": "interactive"}),
        ("/runs/any/step", {"ticks": 1}),
        ("/runs/any/control", {"action": "pause"}),
        ("/worlds/any/run", None),
        ("/experiments/any/run", None),
        (
            "/experiments",
            {
                "definition": {
                    "experiment_id": "x",
                    "title": "t",
                    "hypothesis": "h",
                    "falsification_criterion": "f",
                    "arms": [{"name": "control", "policy_class": "random"}],
                    "control_arm": "control",
                    "seeds": [0, 1],
                    "base_config": {"seed": 0},
                    "analysis": {"primary_metric": "m", "primary_metric_definition": "d", "comparison": "c"},
                }
            },
        ),
    ]
    for path, body in blocked:
        r = c.post(path, json=body) if body is not None else c.post(path)
        assert r.status_code == 405, f"{path}: {r.status_code} {r.text}"
        assert "read-only" in r.json()["detail"]
        assert r.headers.get("allow") == "GET"
    # nothing was created
    assert c.get("/runs").json() == []
    assert c.get("/experiments").json() == []


def test_read_only_keeps_future_self_and_reads_working(client_factory: Callable[..., TestClient]) -> None:
    c = client_factory(read_only=True)
    r = c.post("/scenarios", json=_scenario_request())
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["scenarios"]) == 2 and all(
        s["label"] == "illustrative scenario" for s in body["scenarios"]
    )
    decision_id = body["decision_id"]
    r = c.post(
        "/privacy/consent",
        json={"user_id": "ro-user", "purpose": "reflection_storage", "granted": True},
    )
    assert r.status_code == 200
    r = c.post(
        "/decisions/reflect",
        json={
            "user_id": "ro-user",
            "decision_id": decision_id,
            "chosen_option_id": "a",
            "what_happened": "took it",
        },
    )
    assert r.status_code == 200, r.text
    assert len(c.get("/journey/ro-user").json()["reflections"]) == 1
    assert c.post("/privacy/export", json={"user_id": "ro-user"}).status_code == 200
    assert c.post("/privacy/delete", json={"user_id": "ro-user"}).status_code == 200
    for path in ("/research/claims", "/benchmarks", "/evidence", "/runs", "/experiments"):
        assert c.get(path).status_code == 200, path


def test_read_only_env_parsing(monkeypatch) -> None:
    from hpe_api.settings import Settings

    for raw, expected in (
        ("1", True),
        ("true", True),
        ("YES", True),
        ("0", False),
        ("off", False),
        ("", False),
    ):
        monkeypatch.setenv("HPE_READ_ONLY", raw)
        assert Settings().read_only is expected, raw
    monkeypatch.delenv("HPE_READ_ONLY")
    assert Settings().read_only is False
