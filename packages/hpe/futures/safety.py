"""High-stakes routing (spec §8): crisis, health, legal, financial, relationship.

A transparent keyword classifier, deliberately conservative. It is not a diagnostic tool and makes
no claim to detect risk reliably; it only decides when to stop (crisis) or add a professional
referral. False positives are acceptable; the cost is an extra notice.
"""

from __future__ import annotations

import re

from hpe.contracts.schemas import SafetyNotice

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "crisis": (
        r"suicid\w*",
        r"kill (?:my|him|her)sel(?:f|ves)",
        r"end (?:my|it all|my life)",
        r"self[- ]harm\w*",
        r"hurt(?:ing)? myself",
        r"want to die",
        r"don'?t want to (?:live|be alive)",
        r"overdos\w*",
        r"abus(?:e|ive|ed) (?:me|at home)",
        r"domestic violence",
        r"not safe at home",
        r"unsafe at home",
    ),
    "health": (
        r"diagnos\w*",
        r"medicat\w*",
        r"symptom\w*",
        r"cancer",
        r"surger\w*",
        r"pregnan\w*",
        r"chemo\w*",
        r"prescription\w*",
        r"mental health",
        r"depress\w*",
        r"anxiety disorder",
        r"treatment",
        r"disease",
        r"illness",
        r"therap(?:y|ist)",
        r"stop taking",
    ),
    "legal": (
        r"lawsuit",
        r"\bsue\b",
        r"custody",
        r"visa",
        r"immigration",
        r"criminal",
        r"lawyer",
        r"attorney",
        r"\bcourt\b",
        r"legal action",
        r"contract dispute",
        r"eviction",
    ),
    "financial": (
        r"invest\w*",
        r"stocks?\b",
        r"crypto\w*",
        r"mortgage",
        r"\bloan\b",
        r"\bdebt\b",
        r"bankrupt\w*",
        r"401\s?k",
        r"\bira\b",
        r"retirement (?:savings|account|fund)",
        r"\btax(?:es)?\b",
        r"remortgage",
    ),
    "relationship": (
        r"divorc\w*",
        r"break ?up",
        r"separat(?:e|ion) from",
        r"marri(?:age|ed|y)",
        r"\bspouse\b",
        r"affair",
        r"\bpartner\b(?! company)",
        r"boyfriend",
        r"girlfriend",
    ),
}

_ORDER = ("crisis", "health", "legal", "financial", "relationship")
_COMPILED = {k: re.compile("|".join(f"(?:{p})" for p in v), re.I) for k, v in _KEYWORDS.items()}

_REFERRALS = {
    "health": "a licensed medical or mental-health professional",
    "legal": "a qualified lawyer or legal-aid service in your jurisdiction",
    "financial": "a qualified, independent financial adviser",
    "relationship": "a licensed counsellor or a trusted person in your life",
}

CRISIS_MESSAGE = (
    "It sounds like you may be going through something very hard. This tool cannot help with a crisis, "
    "and you deserve support from a person right now."
)
CRISIS_REFERRAL = (
    "If you are in immediate danger, contact your local emergency services. You can also reach a crisis line "
    "in your country (for example, call or text 988 in the United States)."
)


def classify(*texts: str) -> SafetyNotice:
    blob = " \n ".join(t for t in texts if t)
    for cat in _ORDER:
        if _COMPILED[cat].search(blob):
            if cat == "crisis":
                return SafetyNotice(category="crisis", message=CRISIS_MESSAGE, referral=CRISIS_REFERRAL)
            return SafetyNotice(
                category=cat,  # type: ignore[arg-type]
                message=(
                    f"This decision touches on {cat} matters. These scenarios are educational and exploratory; "
                    f"they are not {cat} advice and cannot account for your full situation."
                ),
                referral=f"Consider talking with {_REFERRALS[cat]} before acting.",
            )
    return SafetyNotice(
        category="none", message="No high-stakes topic detected by keyword screening.", referral=None
    )
