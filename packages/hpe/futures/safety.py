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
        # indirect hopelessness / passive ideation (audit finding H-1)
        r"(?:no|don'?t see (?:the|a)) point (?:in|of) (?:going on|living|carrying on|continuing)",
        r"(?:can'?t|cannot) go on",
        r"(?:better|easier) (?:off )?(?:if i (?:was|were) )?(?:dead|gone|not (?:here|around))",
        r"(?:everyone|they|people) would be better off without me",
        r"(?:wish|wished) i (?:was|were|wasn'?t|weren'?t) (?:dead|alive|born|here)",
        r"(?:thinking|thought|think) about (?:ending|taking) (?:it|my life|everything)",
        r"(?:ending|end) things",
        r"(?:ready|want) to (?:give up on|give up) (?:life|everything|living)",
        r"(?:nothing|no reason) (?:to live|left to live) for",
        r"(?:plan|planning|planned) to (?:hurt|harm|kill)",
        r"(?:take|took) (?:all )?(?:my|the) pills",
        r"(?:cut|cutting) myself",
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
        # medical terms (audit finding M-1)
        r"\bdoctor\b",
        r"physician",
        r"hospital\w*",
        r"\bheart (?:condition|disease|attack|failure)\b",
        r"\bdiabet\w*",
        r"\bstroke\b",
        r"\bchronic\b",
        r"\bdisabilit\w*",
        r"\binjur\w*",
        r"\bpain\b",
        r"\bcondition\b",
        r"\bclinic\w*",
        r"\bmedical\b",
        r"\bhealth(?: issue| problem| condition| scare)",
        r"\baddict\w*",
        r"\balcohol\w*",
        r"\bdrug(?:s| use| problem)\b",
        r"\beating disorder",
        r"\bburnout\b",
        r"\bpanic attack",
    ),
    "legal": (
        r"lawsuit",
        r"\bsu(?:e|ed|ing)\b",
        r"\blitigat\w*",
        r"\bsubpoena\w*",
        r"\barrest\w*",
        r"\bcharged with\b",
        r"\bfelony\b",
        r"\bmisdemeanou?r\b",
        r"\bprobation\b",
        r"\bdeport\w*",
        r"\bgreen card\b",
        r"\bwork permit\b",
        r"\bnon-?compete\b",
        r"\bsettlement\b",
        r"\bharass\w*",
        r"\bdiscriminat\w*",
        r"\bwrongful\b",
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
