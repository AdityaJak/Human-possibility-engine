"""Regression cases from the independent audit (reports/audit/AUDIT.md findings H-1, M-1, M-2)."""

import pytest
from pydantic import ValidationError

from hpe.contracts.schemas import DecisionOption, DecisionRequest, ValueWeight
from hpe.futures.safety import classify


@pytest.mark.parametrize(
    "text",
    [
        "I don't see the point of going on anymore. Job or startup?",
        "Everyone would be better off without me",
        "I've been thinking about ending it",
        "I can't go on like this",
        "I wish I wasn't here",
        "nothing left to live for",
    ],
)
def test_indirect_crisis_language_is_routed_to_crisis(text):
    assert classify(text).category == "crisis"


@pytest.mark.parametrize(
    "text,cat",
    [
        ("my doctor says I have a heart condition; should I take the job", "health"),
        ("I'm being sued by my landlord", "legal"),
        ("I was arrested last year; will that matter", "legal"),
        ("I'm in chronic pain and considering retraining", "health"),
    ],
)
def test_high_stakes_terms_from_audit(text, cat):
    assert classify(text).category == cat


def test_ordinary_career_question_is_not_flagged():
    assert classify("Should I take a higher-paying job, stay, or start a startup?").category == "none"


def _req(**kw):
    base = dict(
        user_id="u1",
        consent_given=True,
        question="job or startup?",
        options=[DecisionOption(option_id="a", label="A"), DecisionOption(option_id="b", label="B")],
        values=[ValueWeight(key="income", label="Income", weight=0.5)],
    )
    base.update(kw)
    return DecisionRequest(**base)


def test_input_size_limits():
    with pytest.raises(ValidationError):
        _req(question="x" * 200_000)
    with pytest.raises(ValidationError):
        _req(user_id="u" * 10_000)
    with pytest.raises(ValidationError):
        _req(values=[ValueWeight(key=f"k{i}", label="l", weight=0.1) for i in range(17)])
    assert _req(question="x" * 4000).question
