#!/usr/bin/env python3
"""Critic (audit/critic-v3): calibration of the Phase III bootstrap test and power of the gate rules.

    PYTHONPATH=packages python reports/phase3/critic/stats_checks.py --out reports/phase3/critic/stats_checks.json

S0  Empirical paired SDs of per-participant NLL differences from the committed Phase II *tune* pilot
    (reports/phase2/human/p2_human_dev_pilot*/outcomes.jsonl.gz) to ground the SD assumptions.
S1  Type-I error of ``hpe.population.core.evaluate.bootstrap`` (one-sided percentile p, 2,000 resamples)
    under H0 (mean difference exactly 0) for normal, heavy-tailed and skewed differences, next to a
    one-sample t-test and a sign-flip permutation test.
S2  Monte-Carlo noise of the 2,000-resample p near the Holm thresholds (same data, different keys).
S3  Power of the frozen gate rules (thesis §5-§6: point estimate beyond the margin AND Holm p < 0.05;
    Gate C/B1 need the primary + >= 1 other counted family, Gate C also "not worse by > 0.02" anywhere,
    B2 needs >= 2 families) under a normal approximation, test n per family from the thesis (two_step
    minus the 3 exposed participants), homogeneous true effects, 1 or 2 frozen candidates.
No human data is read except the committed Phase II tune-partition outcome files in S0.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from hpe.core.rng import stream
from hpe.population.core.evaluate import bootstrap, holm

SEED = 20261104
TEST_N = {"restless4": 347, "garcia": 140, "described": 87, "two_step": 270}
PRIMARY = "restless4"


# ----------------------------------------------------------------------------- S0
def s0_phase2_sds() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in ("p2_human_dev_pilot", "p2_human_dev_pilot_hybrid"):
        f = Path("reports/phase2/human") / run / "outcomes.jsonl.gz"
        if not f.exists():
            continue
        d: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for ln in fh:
                r = json.loads(ln)
                if r["metric"] != "nll_per_choice":
                    continue
                fam, arm = r["arm"].split(":", 1)
                d[fam].setdefault(arm, {})[r["agent_id"]] = r["value"]
        for fam in ("garcia", "described", "bandit2"):
            arms = d.get(fam, {})
            sds = []
            for a, b in itertools.combinations(sorted(arms), 2):
                if "random" in (a, b) or "static_persona" in (a, b):
                    continue
                ks = sorted(set(arms[a]) & set(arms[b]))
                x = np.array([arms[a][k] - arms[b][k] for k in ks])
                sds.append(
                    {
                        "pair": f"{a}-{b}",
                        "n": len(ks),
                        "mean": round(float(x.mean()), 4),
                        "sd": round(float(x.std(ddof=1)), 4),
                    }
                )
            if sds:
                gru = [s for s in sds if "ref_gru" in s["pair"]]
                other = [s for s in sds if "ref_gru" not in s["pair"]]
                out.setdefault(fam, {})[run] = {
                    "sd_vs_ref_gru_range": [min(s["sd"] for s in gru), max(s["sd"] for s in gru)]
                    if gru
                    else None,
                    "sd_between_cognitive_models_range": [
                        min(s["sd"] for s in other),
                        max(s["sd"] for s in other),
                    ]
                    if other
                    else None,
                    "n": sds[0]["n"],
                }
    return out


# ----------------------------------------------------------------------------- S1
def _draw(dist: str, rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    if dist == "normal":
        x = rng.normal(0.0, 0.06, size)
    elif dist == "t3":
        x = rng.standard_t(3, size) * 0.06 / np.sqrt(3.0)
    elif dist == "skew_right":  # most participants slightly better, a few much worse (mean 0)
        g = rng.gamma(0.5, 0.06, size)
        x = g - 0.5 * 0.06
    elif dist == "skew_left":  # most participants slightly worse, a few much better (mean 0)
        g = rng.gamma(0.5, 0.06, size)
        x = -(g - 0.5 * 0.06)
    elif dist == "contaminated":  # 95 % N(0, .03), 5 % N(0, .3)
        x = np.where(rng.uniform(size=size) < 0.05, rng.normal(0, 0.3, size), rng.normal(0, 0.03, size))
    else:
        raise ValueError(dist)
    return x


def _signflip_p(x: np.ndarray, rng: np.random.Generator, n_perm: int = 2000) -> float:
    s = rng.choice([-1.0, 1.0], size=(n_perm, x.size))
    null = (s * np.abs(x)).mean(axis=1)
    return float((1 + (null <= x.mean()).sum()) / (n_perm + 1))


def s1_calibration(n_sims: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for dist in ("normal", "t3", "skew_right", "skew_left", "contaminated"):
        for n in (87, 347):
            rng = stream(SEED, "critic_s1", dist, n)
            xs = _draw(dist, rng, (n_sims, n))
            pb, pt, ps = [], [], []
            for i, x in enumerate(xs):
                pb.append(bootstrap(x, seed=SEED, key=("critic_s1", dist, str(n), str(i)))[3])
                pt.append(float(stats.ttest_1samp(x, 0.0, alternative="less").pvalue))
                ps.append(_signflip_p(x, rng))
            pb_a, pt_a, ps_a = np.array(pb), np.array(pt), np.array(ps)
            res[f"{dist}/n={n}"] = {
                a: {
                    "bootstrap": float((pb_a < a).mean()),
                    "t_test": float((pt_a < a).mean()),
                    "sign_flip": float((ps_a < a).mean()),
                }
                for a in (0.05, 0.0125)
            }
    res["n_sims_per_cell"] = n_sims
    res["mc_se_at_0.05"] = float(np.sqrt(0.05 * 0.95 / n_sims))
    return res


# ----------------------------------------------------------------------------- S2
def s2_resampling_noise() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target_z, label in ((2.24, "p~0.0125 (Holm rank 1 of 4)"), (1.96, "p~0.025 (Holm rank 3 of 4)")):
        rng = stream(SEED, "critic_s2", label)
        x = rng.normal(0, 0.06, 140)
        x = x - x.mean() - target_z * x.std(ddof=0) / np.sqrt(x.size)  # observed mean at the threshold
        ps = [bootstrap(x, seed=SEED, key=("critic_s2", str(k)))[3] for k in range(60)]
        out[label] = {"min": min(ps), "median": float(np.median(ps)), "max": max(ps), "sd": float(np.std(ps))}
    return out


# ----------------------------------------------------------------------------- S3
def _gate_pass(
    est: np.ndarray, se: np.ndarray, fams: list[str], margin: float, rule: str, n_cand: int
) -> np.ndarray:
    """est: [S, F] (negative favours the candidate). Holm over F families x n_cand candidates."""
    p = stats.norm.cdf(est / se)  # one-sided p for H0: mean >= 0
    S, F = est.shape
    # Holm with m = F * n_cand; the other candidate (if any) is assumed null (p ~ U(0,1)) -> conservative-ish
    if n_cand == 2:
        rng = stream(SEED, "critic_s3_other", rule, margin)
        p_all = np.concatenate([p, rng.uniform(size=(S, F))], axis=1)
    else:
        p_all = p
    m = p_all.shape[1]
    order = np.argsort(p_all, axis=1)
    sorted_p = np.take_along_axis(p_all, order, axis=1)
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * sorted_p), axis=1)
    adj = np.empty_like(adj_sorted)
    np.put_along_axis(adj, order, adj_sorted, axis=1)
    adj = adj[:, :F]
    fam_pass = (est <= -margin) & (adj < 0.05)
    prim = fam_pass[:, fams.index(PRIMARY)]
    n_pass = fam_pass.sum(axis=1)
    if rule == "C":
        return prim & (n_pass >= 2) & ~(est > 0.02).any(axis=1)
    if rule == "B1":
        return prim & (n_pass >= 2)
    return n_pass >= 2  # B2


def _holm_matches_production() -> bool:
    rng = stream(SEED, "critic_holm_check")
    p = rng.uniform(0, 0.08, size=(200, 4))
    est = np.full_like(p, -1.0)
    for row in p:
        ref = holm({str(i): float(v) for i, v in enumerate(row)})
        m = row.size
        order = np.argsort(row)
        adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * row[order]))
        mine = np.empty(m)
        mine[order] = adj
        if not np.allclose(mine, [ref[str(i)] for i in range(m)]):
            return False
    return bool(est.size)


def s3_power(n_sims: int) -> dict[str, Any]:
    out: dict[str, Any] = {"vectorised_holm_matches_evaluate.holm": _holm_matches_production()}
    gates = {"C": 0.02, "B1": 0.01, "B2": 0.005}
    for gate, margin in gates.items():
        for fam_set in (
            ["restless4", "garcia", "described"],
            ["restless4", "garcia", "described", "two_step"],
        ):
            for sd in (0.03, 0.06, 0.10):
                for mult in (1.0, 1.5, 2.0):
                    for n_cand in (1, 2):
                        delta = margin * mult
                        rng = stream(SEED, "critic_s3", gate, len(fam_set), sd, mult, n_cand)
                        se = np.array([sd / np.sqrt(TEST_N[f]) for f in fam_set])
                        est = -delta + rng.normal(size=(n_sims, len(fam_set))) * se
                        pw = _gate_pass(est, se, fam_set, margin, gate, n_cand)
                        fam_marg = {
                            f: float(
                                (
                                    (est[:, i] <= -margin)
                                    & (stats.norm.cdf(est[:, i] / se[i]) < 0.05 / (len(fam_set) * n_cand))
                                ).mean()
                            )
                            for i, f in enumerate(fam_set)
                        }
                        out[f"{gate}|m={len(fam_set)}|sd={sd}|delta={delta:.4f}|cands={n_cand}"] = {
                            "gate_pass_prob": round(float(pw.mean()), 3),
                            "per_family_pass_prob_bonferroni_worst_rank": {
                                k: round(v, 3) for k, v in fam_marg.items()
                            },
                        }
    # null: no true effect anywhere -> false-positive gate pass probability
    for gate, margin in gates.items():
        for sd in (0.03, 0.10):
            rng = stream(SEED, "critic_s3_null", gate, sd)
            fams = ["restless4", "garcia", "described", "two_step"]
            se = np.array([sd / np.sqrt(TEST_N[f]) for f in fams])
            est = rng.normal(size=(n_sims, len(fams))) * se
            out[f"NULL {gate}|m=4|sd={sd}"] = {
                "gate_pass_prob": round(float(_gate_pass(est, se, fams, margin, gate, 1).mean()), 5)
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/phase3/critic/stats_checks.json"))
    ap.add_argument("--sims", type=int, default=400, help="S1 simulations per cell")
    args = ap.parse_args()
    res = {
        "S0_phase2_tune_paired_sds": s0_phase2_sds(),
        "S1_type_I_error": s1_calibration(args.sims),
        "S2_bootstrap_p_resampling_noise": s2_resampling_noise(),
        "S3_power": s3_power(20000),
    }
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["S0_phase2_tune_paired_sds"], indent=1))
    print(json.dumps(res["S1_type_I_error"], indent=1))
    print(json.dumps(res["S2_bootstrap_p_resampling_noise"], indent=1))
    for k, v in res["S3_power"].items():
        if "cands=1" in k or k.startswith("NULL"):
            print(k, v["gate_pass_prob"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
