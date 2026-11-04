"""Gate verdicts in code (critic finding S-1), implementing docs/phase3/PHASE3_RESEARCH_THESIS.md §6 as amended.

Inputs are per-family paired contrasts (``first − second``; negative favours the first model) as produced
by ``core.evaluate.paired``. The counted families and the primary family come from the **freeze record**,
never from the families present in a run: a counted family missing from the results enters the Holm
correction with p = 1 and fails (so dropping a family cannot shrink the correction).

p-value rule (critic finding S-4): a family's p is the larger of the one-sided bootstrap p and the
one-sided paired t-test p, so a borderline bootstrap p cannot pass on resampling noise alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from hpe.population.core.evaluate import holm

ALPHA = 0.05
MARGIN_C = 0.02
MARGIN_B1 = 0.01
MARGIN_B2 = 0.005
NOT_WORSE_C = 0.02


def conservative_p(contrast: Mapping[str, Any]) -> float:
    """max(one-sided bootstrap p, one-sided paired t p) for H1: mean difference < 0."""
    p_boot = float(contrast["p_one_sided"])
    n = int(contrast["n"])
    sd = float(contrast.get("sd", float("nan")))
    est = float(contrast["estimate"])
    if n < 2 or not np.isfinite(sd) or sd == 0.0:
        return p_boot
    t = est / (sd / np.sqrt(n))
    p_t = float(stats.t.cdf(t, df=n - 1))
    return max(p_boot, p_t)


@dataclass
class GateVerdict:
    gate: str
    passed: bool
    families: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "passed": self.passed, "families": self.families, "reasons": self.reasons}


def _family_table(
    contrasts: Mapping[str, Mapping[str, Any]], counted: list[str], margin: float
) -> dict[str, dict[str, Any]]:
    ps = {f: (conservative_p(contrasts[f]) if f in contrasts else 1.0) for f in counted}
    adj = holm(ps)
    out: dict[str, dict[str, Any]] = {}
    for f in counted:
        c = contrasts.get(f)
        est = float(c["estimate"]) if c is not None else float("nan")
        meets = c is not None and est <= -margin and adj[f] < ALPHA
        out[f] = {
            "present": c is not None,
            "estimate": est,
            "ci": [c["ci_low"], c["ci_high"]] if c is not None else None,
            "p": ps[f],
            "p_holm": adj[f],
            "meets_margin": bool(meets),
            "n": int(c["n"]) if c is not None else 0,
        }
    return out


def gate_c(
    vs_best_comparator: Mapping[str, Mapping[str, Any]], counted: list[str], primary: str
) -> GateVerdict:
    """≥ 0.02 better (Holm) in ≥ 2 counted families incl. the primary; not worse by > 0.02 in any counted family."""
    fams = _family_table(vs_best_comparator, counted, MARGIN_C)
    wins = [f for f, v in fams.items() if v["meets_margin"]]
    worse = [f for f, v in fams.items() if v["present"] and v["estimate"] > NOT_WORSE_C]
    missing = [f for f, v in fams.items() if not v["present"]]
    reasons = []
    if primary not in wins:
        reasons.append(f"primary family {primary} not won")
    if len(wins) < 2:
        reasons.append(f"won {len(wins)} counted families (< 2)")
    if worse:
        reasons.append(f"worse by > {NOT_WORSE_C} in {worse}")
    if missing:
        reasons.append(f"counted families without results: {missing}")
    return GateVerdict("C", not reasons, fams, reasons)


def gate_b(
    vs_twin: Mapping[str, Mapping[str, Any]],
    vs_best_adapter: Mapping[str, Mapping[str, Any]],
    counted: list[str],
    primary: str,
) -> GateVerdict:
    """B1 (vs separately trained no-adaptation twin, ≥ 0.01) and B2 (vs best conventional adapter, ≥ 0.005),
    each in ≥ 2 counted families including the primary (amendment A3-1 aligns B2 with B1)."""
    b1 = _family_table(vs_twin, counted, MARGIN_B1)
    b2 = _family_table(vs_best_adapter, counted, MARGIN_B2)
    reasons = []
    for name, tab in (("B1", b1), ("B2", b2)):
        wins = [f for f, v in tab.items() if v["meets_margin"]]
        if primary not in wins:
            reasons.append(f"{name}: primary family {primary} not won")
        if len(wins) < 2:
            reasons.append(f"{name}: won {len(wins)} counted families (< 2)")
    fams = {f: {"B1": b1[f], "B2": b2[f]} for f in counted}
    return GateVerdict("B", not reasons, fams, reasons)


def gate_a2(vs_mismatched: Mapping[str, Mapping[str, Any]], counted: list[str]) -> GateVerdict:
    """Own-history adaptation beats length-matched donor adaptation (Holm p < 0.05, estimate < 0) in ≥ 2 families."""
    fams = _family_table(vs_mismatched, counted, 0.0)
    wins = [f for f, v in fams.items() if v["present"] and v["estimate"] < 0 and v["p_holm"] < ALPHA]
    for f in fams:
        fams[f]["meets_margin"] = f in wins
    reasons = [] if len(wins) >= 2 else [f"own > donor in {len(wins)} counted families (< 2)"]
    return GateVerdict("A2", not reasons, fams, reasons)


def tune_stop_rule_c(
    vs_best_comparator: Mapping[str, Mapping[str, Any]], counted: list[str], primary: str
) -> bool:
    """Thesis §7: freeze for Gate C only if, on tune, ≥ 0.02 better in ≥ 2 counted families incl. the primary
    (point estimates; tune is selection-optimistic so no p-value is required here)."""
    wins = [
        f
        for f in counted
        if f in vs_best_comparator and float(vs_best_comparator[f]["estimate"]) <= -MARGIN_C
    ]
    return primary in wins and len(wins) >= 2


def tune_stop_rule_b1(vs_twin: Mapping[str, Mapping[str, Any]], counted: list[str]) -> bool:
    wins = [f for f in counted if f in vs_twin and float(vs_twin[f]["estimate"]) <= -MARGIN_B1]
    return len(wins) >= 2
