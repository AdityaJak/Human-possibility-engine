"""Audit 2+3: independent re-implementation (numpy only) of the H1 and Exp6 primary contrasts from the
committed RAW outcome logs, with and without evaluation seed 7 (seed 7 was one of the developmental
lane's two hyperparameter-tuning task seeds; in the learning lab task seed == evaluation seed).

Per seed: metric = mean over agents of the per-agent metric (one value per seed, as preregistered).
Paired bootstrap: resample seeds with replacement, 2000 resamples (own RNG, so CIs differ from the
runner's in the 3rd decimal at most), percentile 95% CI of the mean paired difference.

Usage: python reports/audit/scripts/independent_stats.py [out.json]
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "reports" / "experiments"


def per_seed(exp: str, arm: str, metric: str, variant: str = "main") -> dict[int, float]:
    out: dict[int, float] = {}
    for run in sorted((EXP / exp / "runs").glob(f"{exp}__{variant}__{arm}__s*")):
        vals = defaultdict(list)
        seed = None
        for line in gzip.open(run / "outcomes.jsonl.gz"):
            r = json.loads(line)
            if r["metric"] == metric:
                vals[r["agent_id"]].append(r["value"])
                seed = r["seed"]
        if seed is not None:
            out[int(seed)] = float(np.mean([np.mean(v) for v in vals.values()]))
    return out


def paired(t: dict[int, float], c: dict[int, float], exclude: set[int], n_boot: int = 2000, seed: int = 12345):
    seeds = sorted(set(t) & set(c) - exclude)
    d = np.array([t[s] - c[s] for s in seeds])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ctrl_mean = float(np.mean([c[s] for s in seeds]))
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "treatment_mean": round(float(np.mean([t[s] for s in seeds])), 4),
        "control_mean": round(ctrl_mean, 4),
        "estimate": round(float(d.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "relative": round(float(d.mean() / ctrl_mean), 4),
        "dz": round(float(d.mean() / d.std(ddof=1)), 3) if len(d) > 1 and d.std(ddof=1) > 0 else None,
    }


def main() -> int:
    res: dict[str, dict] = {}
    # H1 primary: developmental vs strongest baseline (chosen as the best-scoring baseline)
    m = "final_heldout_accuracy"
    arms = {a: per_seed("exp_baselines_heldout", a, m) for a in ("developmental", "plain_rl", "memory_only", "static_persona")}
    for label, excl in (("all_seeds", set()), ("excluding_seed_7", {7})):
        means = {a: np.mean([v for s, v in arms[a].items() if s not in excl]) for a in arms if a != "developmental"}
        strongest = max(means, key=means.get)
        r = paired(arms["developmental"], arms[strongest], excl)
        thr = 0.10 * r["control_mean"]
        r["strongest_baseline"] = strongest
        r["threshold_abs"] = round(thr, 4)
        r["verdict"] = (
            "passed" if r["ci_low"] > 0 and r["estimate"] >= thr
            else "failed" if r["ci_high"] < thr
            else "inconclusive"
        )
        res[f"H1/{label}"] = r
    # Exp6 primary: transfer_auc/shared/trained developmental vs developmental_specialists (min effect 0.03 abs)
    m6 = "transfer_auc/shared/trained"
    dev = per_seed("exp06_transfer", "developmental", m6)
    spec = per_seed("exp06_transfer", "developmental_specialists", m6)
    rl = per_seed("exp06_transfer", "plain_rl", m6)
    for label, excl in (("all_seeds", set()), ("excluding_seed_7", {7})):
        r = paired(dev, spec, excl)
        r["verdict"] = "passed" if r["ci_low"] > 0 and r["estimate"] >= 0.03 else ("failed" if r["ci_high"] < 0.03 else "inconclusive")
        res[f"Exp6/{label}"] = r
        res[f"Exp6_vs_plain_rl/{label}"] = paired(dev, rl, excl)
    # per-seed table for H1 (for the audit report)
    res["H1/per_seed_dev_minus_plain_rl"] = {
        s: round(arms["developmental"][s] - arms["plain_rl"][s], 4) for s in sorted(arms["developmental"])
    }
    txt = json.dumps(res, indent=2)
    print(txt)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
