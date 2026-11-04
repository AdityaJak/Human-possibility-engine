"""HeartSteps V1 micro-randomised trial (Klasnja et al., 2019; data CC-BY-4.0).

At up to five user-chosen times per day, if the participant was *available*, HeartSteps
randomised whether to send an activity suggestion (walking or anti-sedentary). The outcome used
here is the Jawbone step count in the 30 minutes after the decision point.

Column semantics follow the dataset documentation (github.com/klasnja/HeartStepsV1 wiki):
``avail`` availability at the decision point, ``send`` suggestion delivered, ``send.active`` /
``send.sedentary`` suggestion type, ``jbsteps30.zero`` / ``jbsteps30pre.zero`` Jawbone steps in the
30 min after / before with missing minutes counted as zero, ``jbsteps30`` complete-case version.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class DecisionPoint:
    user: int
    decision_index: int
    study_day: int
    available: bool
    send: bool | None
    send_walking: bool | None
    send_sedentary: bool | None
    steps30: float | None  # zero-imputed post-decision 30-min steps
    steps30_pre: float | None  # zero-imputed pre-decision 30-min steps
    steps30_complete: float | None  # NaN-preserving version (None if minutes missing)


def _b(v: str) -> bool | None:
    return {"True": True, "False": False}.get(v)


def _f(v: str) -> float | None:
    return float(v) if v not in ("", "NA") else None


def load_decision_points(path: Path) -> list[DecisionPoint]:
    out: list[DecisionPoint] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            di = int(r["decision.index"])
            out.append(
                DecisionPoint(
                    user=int(r["user.index"]),
                    decision_index=di,
                    study_day=di // 5,
                    available=r["avail"] == "True",
                    send=_b(r["send"]),
                    send_walking=_b(r["send.active"]),
                    send_sedentary=_b(r["send.sedentary"]),
                    steps30=_f(r["jbsteps30.zero"]),
                    steps30_pre=_f(r["jbsteps30pre.zero"]),
                    steps30_complete=_f(r["jbsteps30"]),
                )
            )
    out.sort(key=lambda d: (d.user, d.decision_index))
    return out


def analysis_sample(points: list[DecisionPoint]) -> list[DecisionPoint]:
    """Available decision points with a recorded randomisation outcome and outcome data."""
    return [
        d
        for d in points
        if d.available and d.send is not None and d.steps30 is not None and d.steps30_pre is not None
    ]


def validate(points: list[DecisionPoint]) -> dict:
    sample = analysis_sample(points)
    send = np.array([d.send for d in sample], dtype=float)
    walk = np.array([bool(d.send and d.send_walking) for d in sample], dtype=float)
    sed = np.array([bool(d.send and d.send_sedentary) for d in sample], dtype=float)
    inconsistent = sum(
        1 for d in sample if d.send and d.send_walking is not None and d.send_walking == d.send_sedentary
    )
    return {
        "rows": len(points),
        "users": len({d.user for d in points}),
        "available_decision_points": sum(d.available for d in points),
        "analysis_sample": len(sample),
        "analysis_users": len({d.user for d in sample}),
        "p_send_given_available": float(send.mean()) if len(send) else None,
        "p_walking_given_available": float(walk.mean()) if len(walk) else None,
        "p_sedentary_given_available": float(sed.mean()) if len(sed) else None,
        "sent_with_unknown_or_inconsistent_type": inconsistent,
        "mean_steps30_zero_imputed": float(np.mean([d.steps30 for d in sample])) if sample else None,
        "mean_steps30_complete_case": float(
            np.mean([d.steps30_complete for d in sample if d.steps30_complete is not None])
        )
        if sample
        else None,
        "complete_case_n": sum(d.steps30_complete is not None for d in sample),
        "max_study_day": max((d.study_day for d in points), default=None),
    }
