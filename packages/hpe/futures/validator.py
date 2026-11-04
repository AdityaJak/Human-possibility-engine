"""Language guard for Future Self content (spec §8, §19, §36D).

Rejects unsupported numeric probabilities, guarantees and destiny language in any generated
scenario text. It is applied to every scenario before it leaves the engine, and to any optional
external narration (which must additionally contain no digits at all).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hpe.contracts.schemas import Scenario

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("percentage", re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|percent\b|per\s+cent\b)", re.I)),
    ("odds_ratio", re.compile(r"\b\d+\s*(?:in|out\s+of)\s+\d+\b", re.I)),
    (
        "numeric_probability",
        re.compile(r"\b(?:probability|chance|chances|odds|likelihood)\b[^.\n]{0,40}?\d", re.I),
    ),
    (
        "numeric_probability",
        re.compile(r"\d[^.\n]{0,25}?\b(?:probability|chance|chances|odds|likelihood)\b", re.I),
    ),
    ("probability_word", re.compile(r"\bprobabilit(?:y|ies)\s+(?:of|that|is|=)", re.I)),
    ("guarantee", re.compile(r"\bguarantee(?:d|s)?\b", re.I)),
    ("destiny", re.compile(r"\byou\s+will\s+become\b|\byou\s+are\s+destined\b|\byour\s+destiny\b", re.I)),
    ("certainty", re.compile(r"\byou\s+will\s+(?:definitely|certainly|surely|undoubtedly)\b", re.I)),
    ("certainty", re.compile(r"\b(?:certain|sure)\s+to\s+(?:succeed|fail|be\s+happ)", re.I)),
)


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    field: str
    excerpt: str


def check_text(text: str, field: str = "text") -> list[Violation]:
    out: list[Violation] = []
    for rule, pat in _RULES:
        for m in pat.finditer(text or ""):
            lo, hi = max(0, m.start() - 20), min(len(text), m.end() + 20)
            out.append(Violation(rule, field, text[lo:hi]))
    return out


def scenario_texts(s: Scenario) -> list[tuple[str, str]]:
    """Every free-text field of a scenario that a user reads (disclaimer excluded: it is fixed copy)."""
    fields: list[tuple[str, str]] = [("title", s.title), ("narrative", s.narrative)]
    fields += [(f"tradeoffs[{i}].rationale", t.rationale) for i, t in enumerate(s.tradeoffs)]
    fields += [(f"uncertainties[{i}]", u) for i, u in enumerate(s.uncertainties)]
    fields += [(f"cannot_predict[{i}]", u) for i, u in enumerate(s.cannot_predict)]
    for i, e in enumerate(s.reversible_experiments):
        fields += [
            (f"reversible_experiments[{i}].title", e.title),
            (f"reversible_experiments[{i}].description", e.description),
        ]
    for i, a in enumerate(s.assumptions):
        fields += [(f"assumptions[{i}].claim", a.claim)]
    for i, c in enumerate(s.calculations):
        fields += [(f"calculations[{i}].note", c.note)]
    return fields


def check_scenario(s: Scenario) -> list[Violation]:
    out: list[Violation] = []
    for field, text in scenario_texts(s):
        out.extend(check_text(text, field))
    return out


def check_external_narration(text: str) -> list[Violation]:
    """Optional LLM narration is held to a stricter rule: no digits at all (Tier 3 never produces numbers)."""
    out = check_text(text, "narration")
    if re.search(r"\d", text or ""):
        out.append(Violation("digits_in_external_narration", "narration", text[:60]))
    return out


class ForbiddenLanguageError(ValueError):
    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        super().__init__("; ".join(f"{v.rule} in {v.field}: {v.excerpt!r}" for v in violations))
