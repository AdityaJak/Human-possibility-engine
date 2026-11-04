"""Audit 2b: non-learning policies must not change under any feedback; the frozen developmental
ablation must not change; learning policies must change. Practice uses full + bandit feedback.

Usage: python reports/audit/scripts/frozen_policies.py
"""

from __future__ import annotations

import json

from hpe.contracts.conformance import make_init
from hpe.contracts.registry import get_factory
from hpe.core import stream
from hpe.tasks import SymbolWorld, assess, policy_state_hash, practice


def main() -> int:
    w = SymbolWorld(seed=99)
    cases = [
        ("static_persona", {}),
        ("memory_only", {}),
        ("plain_rl", {}),
        ("developmental", {}),
        ("developmental", {"use_learning": False}),
    ]
    out = {}
    for pc, params in cases:
        pol = get_factory(pc, params)(make_init(w, seed=5))
        h0 = policy_state_hash(pol)
        a0 = assess(pol, w, "compare", 20.0).accuracy
        rng = stream(1, "audit", pc)
        for mode in ("full", "bandit"):
            practice(pol, w, "compare", 150, rng, agent_id="a0000", tick=1, age_years=20.0, mode=mode)  # type: ignore[arg-type]
        h1 = policy_state_hash(pol)
        a1 = assess(pol, w, "compare", 20.0).accuracy
        key = pc + ("" if not params else json.dumps(params))
        out[key] = {"state_changed": h0 != h1, "heldout_before": round(a0, 3), "heldout_after": round(a1, 3)}
        print(key, out[key])
    ok = (
        not out["developmental{\"use_learning\": false}"]["state_changed"]
        and out["developmental"]["state_changed"]
        and out["plain_rl"]["state_changed"]
        and out["static_persona"]["heldout_before"] == out["static_persona"]["heldout_after"]
    )
    print("OK" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
