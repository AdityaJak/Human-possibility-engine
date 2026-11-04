#!/usr/bin/env python3
"""Critic (audit/critic-v3), results review of the exploratory (tune-partition) Phase III runs.

    PYTHONPATH=packages python reports/phase3/critic/results_review_v2.py \
        --out reports/phase3/critic/results_review_v2.json

Reads only committed result files (``reports/phase3/**/parts/*.json``: per-participant tune scores written by
the runner). No model is fitted and no data file is opened. It

1. recomputes the central paired contrasts across runs (participants matched by key), with the production
   bootstrap and a paired t-test, and checks that the participant sets are identical;
2. applies the tune stop rules in code (``core.gates.tune_stop_rule_c`` / ``tune_stop_rule_b1``) with the best
   comparator taken over *every* committed pool run for the family;
3. collects every committed single fit of one and the same pop_gru configuration (different seeds only) to
   measure seed-to-seed variation of a single network, and compares it with the margins;
4. sums the CPU seconds recorded in run logs (budget condition of the NO-GO rule).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from hpe.population.core import evaluate, gates

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "reports" / "phase3"
SEED = 20261104


def scores(run: str, part: str, variant: str = "main") -> dict[str, float]:
    r = json.loads((R / run / "parts" / f"{part}.json").read_text())
    return {s["key"]: s["nll"] for s in r["variants"][variant]}


def selected(run: str, part: str) -> dict[str, Any]:
    r = json.loads((R / run / "parts" / f"{part}.json").read_text())
    return {
        "select_nll": r["selected"]["select_nll"],
        "lapse": r["selected"]["lapse"],
        "violations": r["causality_violations"],
    }


def contrast(a: dict[str, float], b: dict[str, float], tag: str) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    sa = [evaluate.ParticipantScore(k, a[k], 1, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)) for k in keys]
    sb = [evaluate.ParticipantScore(k, b[k], 1, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)) for k in keys]
    c = evaluate.paired(sa, sb, seed=SEED, key=("critic_rr", tag), n_boot=20000)
    t = stats.ttest_1samp(d, 0.0)
    return {
        "estimate": round(c["estimate"], 5),
        "ci95": [round(c["ci_low"], 5), round(c["ci_high"], 5)],
        "sd": round(c["sd"], 4),
        "n": c["n"],
        "unpaired_keys": len(set(a) ^ set(b)),
        "p_two_sided_t": float(t.pvalue),
        "mean_first": round(float(np.mean([a[k] for k in keys])), 5),
        "mean_second": round(float(np.mean([b[k] for k in keys])), 5),
    }


RUNS = {
    "r4_pool": "baselines/x3_b03_pop_restless4",
    "r4_adapt": "baselines/x3_b04_adapt_restless4",
    "r4_cand": "candidates/x3_c01_restless4",
    "r4_ens": "director/x3_e01_ensemble_restless4",
    "ga_pool": "baselines/x3_b03_pop_garcia",
    "ga_cand": "candidates/x3_c03_garcia",
    "de_cand": "candidates/x3_c05_described",
}


def central_contrasts() -> dict[str, Any]:
    f3 = scores(RUNS["r4_cand"], "restless4__f3_arbiter")
    out: dict[str, Any] = {"restless4": {}, "garcia": {}, "described": {}}
    r4 = out["restless4"]
    for nm in ("pop_gru_ens", "pop_gru", "pop_tf", "pop_lstm", "ref_gru_p2"):
        r4[f"F3 - {nm} (B03)"] = contrast(f3, scores(RUNS["r4_pool"], f"restless4__{nm}"), f"r4 f3 {nm}")
    r4["F3 - pop_gru (same run, same size)"] = contrast(
        f3, scores(RUNS["r4_cand"], "restless4__pop_gru"), "r4 f3 popc"
    )
    r4["F3 - twin (B1-type)"] = contrast(
        f3, scores(RUNS["r4_cand"], "restless4__f3_arbiter", "twin"), "r4 f3 twin"
    )
    for nm in ("gru_hb_mix", "bayes_head", "ft_head"):
        r4[f"F3 - {nm} (B2-type, B04)"] = contrast(
            f3, scores(RUNS["r4_adapt"], f"restless4__{nm}"), f"r4 f3 {nm}"
        )
    ens = {
        nm: scores(RUNS["r4_ens"], f"restless4__{nm}") for nm in ("f3_ens", "pop_gru_ens", "bayes_head_ens")
    }
    r4["F3_ens - pop_gru_ens (E01)"] = contrast(ens["f3_ens"], ens["pop_gru_ens"], "r4 e01 f3 pop")
    r4["F3_ens - bayes_head_ens (E01)"] = contrast(ens["f3_ens"], ens["bayes_head_ens"], "r4 e01 f3 bh")
    r4["bayes_head_ens - pop_gru_ens (E01)"] = contrast(
        ens["bayes_head_ens"], ens["pop_gru_ens"], "r4 e01 bh pop"
    )
    r4["pop_gru_ens E01 - pop_gru_ens B03 (same config, other seeds)"] = contrast(
        ens["pop_gru_ens"], scores(RUNS["r4_pool"], "restless4__pop_gru_ens"), "r4 ens seeds"
    )
    g3 = scores(RUNS["ga_cand"], "garcia__f3_arbiter")
    ga = out["garcia"]
    for nm in ("pop_tf", "pop_gru_ens", "ref_gru_p2", "pop_gru"):
        ga[f"F3 - {nm} (B03)"] = contrast(g3, scores(RUNS["ga_pool"], f"garcia__{nm}"), f"ga f3 {nm}")
    ga["F3 - twin (B1-type)"] = contrast(
        g3, scores(RUNS["ga_cand"], "garcia__f3_arbiter", "twin"), "ga f3 twin"
    )
    ga["F3 - pop_gru (X3-001b, 8 configs)"] = contrast(
        g3, scores("director/x3_001b_core_sanity", "garcia__pop_gru"), "ga f3 pop001b"
    )
    ga["F3 - pop_gru (same run, 4 configs)"] = contrast(
        g3, scores(RUNS["ga_cand"], "garcia__pop_gru"), "ga f3 popc"
    )
    d3 = scores(RUNS["de_cand"], "described__f3_arbiter")
    de = out["described"]
    de["F3 - pop_gru (same run)"] = contrast(d3, scores(RUNS["de_cand"], "described__pop_gru"), "de f3 pop")
    de["F1 - pop_gru (same run)"] = contrast(
        scores(RUNS["de_cand"], "described__f1_latent"),
        scores(RUNS["de_cand"], "described__pop_gru"),
        "de f1 pop",
    )
    de["F3 - twin (B1-type)"] = contrast(
        d3, scores(RUNS["de_cand"], "described__f3_arbiter", "twin"), "de f3 twin"
    )
    return out


def stop_rules() -> dict[str, Any]:
    """Tune stop rules with the best comparator over all committed pool runs (lowest tune NLL)."""
    pools = {
        "restless4": [
            (RUNS["r4_pool"], f"restless4__{n}")
            for n in ("pop_gru_ens", "pop_gru", "pop_tf", "pop_lstm", "ref_gru_p2")
        ]
        + [
            (RUNS["r4_adapt"], f"restless4__{n}")
            for n in (
                "gru_hb_mix",
                "bayes_head",
                "ft_head",
                "ft_full",
                "reptile_gru",
                "cavia_gru",
                "exemplar_mix",
                "hb_cog",
            )
        ],
        "garcia": [
            (RUNS["ga_pool"], f"garcia__{n}")
            for n in ("pop_tf", "pop_gru_ens", "ref_gru_p2", "pop_gru", "pop_lstm")
        ],
        "described": [(RUNS["de_cand"], "described__pop_gru")],
    }
    cands = {
        "restless4": (RUNS["r4_cand"], "restless4"),
        "garcia": (RUNS["ga_cand"], "garcia"),
        "described": (RUNS["de_cand"], "described"),
    }
    best = {}
    for fam, members in pools.items():
        sel = sorted((selected(run, p)["select_nll"], run, p) for run, p in members)
        best[fam] = {"model": sel[0][2], "run": sel[0][1], "select_nll": round(sel[0][0], 5)}
    out: dict[str, Any] = {"best_comparator": best, "candidates": {}}
    counted = ["restless4", "garcia", "described"]
    for cand in ("f1_latent", "f2_world_policy", "f3_arbiter"):
        vs_best, vs_twin = {}, {}
        for fam in counted:
            run, prefix = cands[fam]
            a = scores(run, f"{prefix}__{cand}")
            vs_best[fam] = contrast(a, scores(best[fam]["run"], best[fam]["model"]), f"stop {cand} {fam}")
            vs_twin[fam] = contrast(a, scores(run, f"{prefix}__{cand}", "twin"), f"stop twin {cand} {fam}")
        out["candidates"][cand] = {
            "vs_best_estimates": {f: v["estimate"] for f, v in vs_best.items()},
            "vs_twin_estimates": {f: v["estimate"] for f, v in vs_twin.items()},
            "tune_stop_rule_C": gates.tune_stop_rule_c(vs_best, counted, "restless4"),
            "tune_stop_rule_B1_as_written": gates.tune_stop_rule_b1(vs_twin, counted),
            "tune_B1_with_primary": gates.tune_stop_rule_b1(vs_twin, counted)
            and vs_twin["restless4"]["estimate"] <= -gates.MARGIN_B1,
        }
    return out


def seed_variance() -> dict[str, Any]:
    """Every committed single fit of one pop_gru configuration; only the seed differs."""
    groups = {
        "restless4 hidden32 lr0.01 dropout0": ("restless4", {"hidden": 32, "lr": 0.01, "dropout": 0.0}),
        "garcia hidden12 lr0.01 dropout0": ("garcia", {"hidden": 12, "lr": 0.01, "dropout": 0.0}),
        "garcia hidden32 lr0.01 dropout0": ("garcia", {"hidden": 32, "lr": 0.01, "dropout": 0.0}),
    }
    trainer = {"max_epochs": 150, "patience": 10, "batch_size": 8}
    out: dict[str, Any] = {}
    for label, (fam, want) in groups.items():
        rows = []
        for p in sorted(glob.glob(str(R / "*" / "*" / "parts" / f"{fam}__pop_gru.json"))):
            r = json.loads(Path(p).read_text())
            for f in r["fits"]:
                c = f.get("config", {})
                if "skipped" in f or any(c.get(k) != v for k, v in want.items()):
                    continue
                if any(c.get(k, trainer[k]) != trainer[k] for k in trainer):
                    continue
                rows.append(
                    {
                        "source": str(Path(p).relative_to(R)),
                        "select_nll": f["select_nll"],
                        "lapse": f["lapse"],
                    }
                )
        for p in sorted(glob.glob(str(R / "candidates" / "*" / "parts" / f"{fam}__*.json"))):
            r = json.loads(Path(p).read_text())
            t = r.get("twin")
            if not t:
                continue
            c = t["config"]
            if all(c.get(k) == v for k, v in want.items()) and all(
                c.get(k, trainer[k]) == trainer[k] for k in trainer
            ):
                rows.append(
                    {
                        "source": str(Path(p).relative_to(R)) + " [twin]",
                        "select_nll": t["select_nll"],
                        "lapse": t["lapse"],
                    }
                )
        seen: set[float] = set()
        dedup = []
        for x in rows:  # identical values to all digits are the same fit re-used by another run (same seed)
            if x["select_nll"] in seen:
                x = {**x, "duplicate_of_earlier_row": True}
            else:
                seen.add(x["select_nll"])
            dedup.append(x)
        rows = dedup
        refit = ROOT / "reports" / "phase3" / "critic" / f"seed_refit_{fam}.json"
        if refit.exists():
            rf = json.loads(refit.read_text())
            if all(rf["config"].get(k) == v for k, v in want.items()):
                rows += [
                    {
                        "source": f"critic refit seed {r['seed']}",
                        "select_nll": r["select_nll"],
                        "lapse": r["lapse"],
                    }
                    for r in rf["fits"]
                ]
        v = np.array([x["select_nll"] for x in rows if not x.get("duplicate_of_earlier_row")])
        out[label] = {
            "fits": rows,
            "n": int(v.size),
            "mean": round(float(v.mean()), 5) if v.size else None,
            "sd": round(float(v.std(ddof=1)), 5) if v.size > 1 else None,
            "range": round(float(v.max() - v.min()), 5) if v.size else None,
        }
    return out


def cpu_spent() -> dict[str, Any]:
    tot = 0.0
    per: dict[str, float] = {}
    for p in sorted(glob.glob(str(R / "**" / "run_info.json"), recursive=True)):
        r = json.loads(Path(p).read_text())
        s = sum(float(i.get("cpu_s", 0.0)) for i in r.get("invocations", []))
        per[str(Path(p).relative_to(R))] = round(s / 3600.0, 2)
        tot += s
    return {
        "cpu_hours_recorded_in_run_info": round(tot / 3600.0, 1),
        "per_run_hours": per,
        "budget_hours": 60,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports/phase3/critic/results_review_v2.json")
    args = ap.parse_args()
    res = {
        "central_contrasts": central_contrasts(),
        "stop_rules": stop_rules(),
        "seed_variance": seed_variance(),
        "cpu": cpu_spent(),
    }
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1)[:15000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
