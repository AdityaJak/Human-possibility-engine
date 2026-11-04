"""Director's re-test of the critic's blocking design findings with the critic's own attack models.

Uses ``reports/phase3/critic/causal_attacks.py`` (written by the critic) unchanged as a library: every
cheating model is run through the production check exactly as the runner calls it (20 sequences, donors
on), and every adaptive model that does not use the harness is checked against the runner's G-1 rule.
Output: reports/phase3/director/verify_audit_fixes.json. This is not the critic's independent sign-off;
the critic re-runs its own scripts when its lane can run again.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "critic_attacks", ROOT / "reports/phase3/critic/causal_attacks.py"
)
ca = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["critic_attacks"] = ca
spec.loader.exec_module(ca)  # type: ignore[union-attr]

from hpe.population.core.adaptation import AdaptiveBase  # noqa: E402
from hpe.population.core.conformance import check_causal  # noqa: E402


def main() -> int:
    out: dict = {}
    for k, T in ((2, 40), (4, 60)):
        seqs = ca.toy_seqs(24, k=k, T=T) if hasattr(ca, "toy_seqs") else None
        res = {}
        for cls in (
            ca.Honest,
            ca.PeekAll,
            ca.ProbeDodger,
            ca.KeyCache,
            ca.Transductive,
            ca.InvariantFields,
            ca.DonorPathPeek,
            ca.SabotagedAblations,
        ):
            m = cls()
            viol = check_causal(m, seqs[:20], seed=20261108)
            g1 = bool(getattr(m, "adaptive", False)) and not isinstance(m, AdaptiveBase)
            res[cls.__name__] = {
                "check_causal_flags": bool(viol),
                "runner_G1_flag": g1,
                "excluded_by_runner": bool(viol) or g1,
                "first_violation": viol[:1],
            }
        out[f"k{k}_T{T}"] = res
    dest = ROOT / "reports/phase3/director/verify_audit_fixes.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for cfg, res in out.items():
        print(cfg)
        for name, r in res.items():
            print(
                f"  {name:20s} causal={r['check_causal_flags']!s:5s} G1={r['runner_G1_flag']!s:5s} excluded={r['excluded_by_runner']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
