from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hpe.contracts.schemas import DecisionRequest, EvidenceSourceType, Scenario
from hpe.futures import (
    ConsentRequiredError,
    ForbiddenLanguageError,
    NarrationGate,
    ScenarioEngine,
    check_scenario,
    check_text,
    classify,
)
from hpe.futures import calculators as calc
from hpe.futures.narration import MockNarration, NoNarration
from hpe.futures.validator import check_external_narration

ROOT = Path(__file__).resolve().parents[2]
FUTURES_DIR = ROOT / "packages" / "hpe" / "futures"


def spec_request(**over: object) -> DecisionRequest:
    d: dict[str, object] = {
        "user_id": "demo-1",
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
                "inputs": {"monthly_income": 6200, "hours_per_week": 45},
            },
            {
                "option_id": "startup",
                "label": "Attempt a startup",
                "kind": "start_project",
                "inputs": {"monthly_income": 0, "hours_per_week": 60},
            },
        ],
        "values": [
            {"key": "income", "label": "Income", "weight": 0.6},
            {"key": "stability", "label": "Security", "weight": 0.7},
            {"key": "autonomy", "label": "Autonomy", "weight": 0.9},
            {"key": "relationships", "label": "Relationships", "weight": 0.5},
            {"key": "learning", "label": "Growth", "weight": 0.8},
        ],
        "constraints": {"savings": 24000, "monthly_expenses": 3500, "available_hours_per_week": 50},
        "horizon_years": 3,
    }
    d.update(over)
    return DecisionRequest.model_validate(d)


def _by_option(scenarios: list[Scenario]) -> dict[str, Scenario]:
    return {s.option_id: s for s in scenarios}


def test_spec_example_three_illustrative_scenarios() -> None:
    resp = ScenarioEngine().generate(spec_request(), decision_id="dec-x")
    assert len(resp.scenarios) == 3
    ids = {e.evidence_id for e in resp.evidence}
    for s in resp.scenarios:
        assert s.label == "illustrative scenario"
        assert "not a prediction" in s.disclaimer and "not a forecast" in s.narrative
        assert s.title.startswith("Illustrative path:")
        assert not check_scenario(s), check_scenario(s)
        assert any("Insufficient evidence to predict" in c for c in s.cannot_predict)
        assert len(s.reversible_experiments) >= 2
        assert all(
            e.reversibility in ("fully_reversible", "mostly_reversible") for e in s.reversible_experiments
        )
        assert set(s.evidence_ids) <= ids, "every evidence id must resolve in the response ledger"
        for c in s.calculations:
            assert c.evidence_id in ids
        assert {t.value_key for t in s.tradeoffs} == {
            "income",
            "stability",
            "autonomy",
            "relationships",
            "learning",
        }
        assert "you will become" not in s.narrative.lower()
    assert all(e.source_type != EvidenceSourceType.SIMULATED_OUTCOME for e in resp.evidence)
    assert "No path is recommended" in resp.value_alignment_note
    assert resp.safety is None


def test_calculations_are_exact_arithmetic() -> None:
    sc = _by_option(ScenarioEngine().generate(spec_request(), decision_id="d").scenarios)
    startup = {c.name: c for c in sc["startup"].calculations}
    assert startup["cash_runway_months"].result == pytest.approx(24000 / 3500, abs=0.01)
    assert startup["income_difference_over_horizon"].result == pytest.approx((0 - 5000) * 12 * 3)
    assert startup["weekly_hours_remaining"].result == -10
    job = {c.name: c for c in sc["job"].calculations}
    assert job["income_difference_over_horizon"].result == pytest.approx(1200 * 36)
    stay = {c.name: c for c in sc["stay"].calculations}
    assert stay["cash_runway_months"].result is None  # income covers expenses: no runway number is invented
    assert "income_difference_over_horizon" not in stay
    trade = {t.value_key: t for t in sc["job"].tradeoffs}
    assert trade["income"].direction == "likely_better" and trade["income"].evidence_ids[0].endswith(
        "calc-income_difference_over_horizon"
    )
    st = {t.value_key: t for t in sc["startup"].tradeoffs}
    assert st["stability"].direction == "likely_worse" and st["autonomy"].direction == "likely_better"
    assert any("venture would succeed" in c for c in sc["startup"].cannot_predict)


def test_missing_inputs_produce_hints_not_numbers() -> None:
    req = spec_request(constraints={})
    sc = _by_option(ScenarioEngine().generate(req, decision_id="d").scenarios)
    assert not any(c.name == "cash_runway_months" for c in sc["startup"].calculations)
    assert any("cash-runway" in u for u in sc["startup"].uncertainties)


def test_calculator_edge_cases() -> None:
    assert calc.cash_runway({"savings": 100}) is None
    zero = calc.cash_runway({"savings": 0, "monthly_expenses": 10, "monthly_income": 0})
    assert zero is not None and zero.result == 0.0
    assert calc.cash_runway({"savings": "abc", "monthly_expenses": 1, "monthly_income": 0}) is None
    assert calc.cash_runway({"savings": float("nan"), "monthly_expenses": 1, "monthly_income": 0}) is None
    be = calc.training_break_even({"training_cost": 1200, "expected_monthly_income_increase": 300})
    assert be is not None and be.result == 4.0 and "YOUR assumption" in be.note
    nz = calc.training_break_even({"training_cost": 1200, "expected_monthly_income_increase": 0})
    assert nz is not None and nz.result is None
    tb = calc.time_budget({"hours_per_week": 10}, None, 2)
    assert tb is not None and tb.name == "total_hours_over_horizon" and tb.result == 1040


def test_learn_skill_assumption_is_labelled() -> None:
    req = spec_request(
        options=[
            {"option_id": "stay", "label": "Keep my job", "kind": "current_path"},
            {
                "option_id": "skill",
                "label": "Learn data analysis",
                "kind": "learn_skill",
                "inputs": {
                    "training_cost": 1200,
                    "expected_monthly_income_increase": 300,
                    "hours_per_week": 8,
                },
            },
        ]
    )
    resp = ScenarioEngine().generate(req, decision_id="d")
    skill = _by_option(resp.scenarios)["skill"]
    assert any(c.name == "training_break_even_months" and c.result == 4.0 for c in skill.calculations)
    assumption = [a for a in skill.assumptions if a.evidence_id.endswith("expected_monthly_income_increase")]
    assert assumption and assumption[0].source_type == EvidenceSourceType.ASSUMPTION
    assert any("payoff" in c for c in skill.cannot_predict)


@pytest.mark.parametrize(
    "text",
    [
        "You have an 82% chance of success.",
        "There is a 3 in 4 likelihood this works.",
        "The probability of success is 0.8.",
        "Success is guaranteed.",
        "You will become a millionaire.",
        "You will definitely be happier.",
        "odds of 70 percent",
        "your destiny is set",
    ],
)
def test_validator_catches_forbidden_language(text: str) -> None:
    assert check_text(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Savings of 24,000 cover a monthly shortfall of 3,500 for about 6.9 months.",
        "Imagine it is 3 years from now. This is one possibility among many, not a forecast.",
        "Insufficient evidence to predict whether the venture would succeed.",
    ],
)
def test_validator_allows_arithmetic_and_caveats(text: str) -> None:
    assert not check_text(text)


def test_engine_raises_if_template_ever_emits_forbidden_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from hpe.futures import scenarios as mod

    monkeypatch.setitem(mod._KIND_SENTENCE, "new_job", "You will become rich with a 90% chance.")
    with pytest.raises(ForbiddenLanguageError):
        ScenarioEngine().generate(spec_request(), decision_id="d")


def test_consent_required() -> None:
    with pytest.raises(ConsentRequiredError):
        ScenarioEngine().generate(spec_request(consent_given=False))


def test_safety_classifier() -> None:
    assert classify("I keep thinking about suicide").category == "crisis"
    assert classify("Should I stop taking my medication?").category == "health"
    assert classify("Should I sue my landlord?").category == "legal"
    assert classify("Should I put my savings in crypto?").category == "financial"
    assert classify("Should I divorce my spouse?").category == "relationship"
    assert classify("Should I take a higher-paying job or attempt a startup?").category == "none"
    crisis = ScenarioEngine().generate(spec_request(question="I want to hurt myself"))
    assert crisis.scenarios == [] and crisis.safety is not None and crisis.safety.category == "crisis"


def test_narration_gate_external_path() -> None:
    req = spec_request()
    gate = NarrationGate(MockNarration(), max_calls_per_hour=2)
    resp = ScenarioEngine(narration=gate).generate(req, decision_id="d")
    sources = [s.narrative_source for s in resp.scenarios]
    assert (
        sources[:2] == ["llm_external", "llm_external"] and sources[2] == "template"
    )  # rate limit respected
    assert gate.calls == 2
    off = ScenarioEngine(narration=NarrationGate(MockNarration(), max_calls_per_hour=0)).generate(
        req, decision_id="d"
    )
    assert all(s.narrative_source == "template" for s in off.scenarios)
    none = ScenarioEngine(narration=NarrationGate(NoNarration(), max_calls_per_hour=5)).generate(
        req, decision_id="d"
    )
    assert all(s.narrative_source == "template" for s in none.scenarios)


def test_external_narration_with_numbers_is_rejected() -> None:
    class Leaky(MockNarration):
        def narrate(self, scenario, request):  # type: ignore[no-untyped-def]
            return "In 3 years you will be fine."

    gate = NarrationGate(Leaky(), max_calls_per_hour=10)
    resp = ScenarioEngine(narration=gate).generate(spec_request(), decision_id="d")
    assert all(s.narrative_source == "template" for s in resp.scenarios) and gate.rejected == 3
    assert check_external_narration("no digits here, an imaginative sketch") == []


FORBIDDEN_IMPORTS = ("hpe.world", "hpe.development", "hpe.evaluation")


def test_import_boundary_static() -> None:
    """hpe.futures must never import simulation/research packages (ADR-0005)."""
    for path in FUTURES_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not n.startswith(FORBIDDEN_IMPORTS), f"{path.name} imports {n}"
        assert "importlib" not in path.read_text(), f"{path.name} must not import dynamically"


def test_import_boundary_runtime() -> None:
    code = (
        "import sys, json; import hpe.futures; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('hpe.world','hpe.development','hpe.evaluation')))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(ROOT / "packages"), "PATH": ""},
    )
    assert json.loads(out.stdout) == []
