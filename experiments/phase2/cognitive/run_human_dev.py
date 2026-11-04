#!/usr/bin/env python3
"""Phase II EXPLORATORY human dev-partition comparison for Candidate A (hybrid_exemplar).

    HPE_PSYCH101_SUBSET=<jsonl> python experiments/phase2/cognitive/run_human_dev.py --paradigm mcpl
    ... --paradigm igt

Protocol (docs/phase2/EXPERIMENT_REGISTRY.md): hyper-parameters, lapse and IGT reward scale are
selected on TUNE participants only; every model is then scored once on DEV participants.
TEST participants are never loaded. Per-participant NLL/choice (lower is better) with participant
bootstrap CIs, paired contrasts, and the predictability-tercile breakdown of X-002.
Outputs: reports/phase2/cognitive/human_dev/<paradigm>/{results.json,report.md}. Exploratory only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import hpe.development  # noqa: F401  (registers developmental + hybrid_exemplar)
from hpe.evaluation.human.harness import _factory_for, lapse_nll, make_init, run_participant
from hpe.evaluation.human.models import IGT_ACTIONS, MCPL_ACTIONS, encode
from hpe.evaluation.human.psych101 import Participant, load_participants

ROOT = Path(__file__).resolve().parents[3]
PARTITIONS = ROOT / "reports/phase2/partitions/psych101.json"
GRIDS = Path(__file__).with_name("human_dev_grids.yaml")
FAMILY = {"mcpl": "mcpl", "igt": "igt"}


def partition_keys(paradigm: str) -> dict[str, set[str]]:
    parts = json.loads(PARTITIONS.read_text())
    out: dict[str, set[str]] = {"dev": set(), "tune": set(), "test": set()}
    for exp, v in parts["families"][FAMILY[paradigm]].items():
        for split in out:
            out[split].update(f"{exp}#{p}" for p in v[split])
    return out


def grid(spec: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not spec:
        return [{}]
    keys = sorted(spec)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(spec[k] for k in keys))]


def fit_and_score(
    name: str,
    spec: dict[str, Any],
    paradigm: str,
    tune: list[Participant],
    dev: list[Participant],
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    cls = spec.get("class", name)
    flags = dict(spec.get("flags", {}))
    k = MCPL_ACTIONS if paradigm == "mcpl" else IGT_ACTIONS
    scales = cfg["igt_reward_scale_grid"] if paradigm == "igt" else [0.0]
    enc: dict[tuple[str, float], list] = {}

    def steps(p: Participant, scale: float) -> list:
        key = (p.key, scale)
        if key not in enc:
            enc[key] = encode(p, scale=scale or 500.0, age_years=cfg["age_years"])
        return enc[key]

    t0 = time.process_time()
    best: tuple[float, dict[str, Any], float, float] | None = None
    n_cfg = 0
    for scale in scales:
        for params in grid(spec.get(paradigm, {})):
            n_cfg += 1
            factory = _factory_for(cls, {**flags, **params}, tune, paradigm)
            pcs = [run_participant(factory, steps(p, scale), make_init(p.key, seed))[0] for p in tune]
            for lapse in cfg["lapse_grid"]:
                obj = float(np.mean([lapse_nll(pc, lapse, k) for pc in pcs]))
                if best is None or obj < best[0] - 1e-12:
                    best = (obj, params, lapse, scale)
    assert best is not None
    obj, params, lapse, scale = best
    factory = _factory_for(cls, {**flags, **params}, tune, paradigm)
    per: dict[str, dict[str, float]] = {}
    for p in dev:
        pc, top = run_participant(factory, steps(p, scale), make_init(p.key, seed))
        per[p.key] = {"nll": lapse_nll(pc, lapse, k), "top1": float(top.mean()), "n": len(pc)}
    return {
        "model": name,
        "class": cls,
        "flags": flags,
        "params": params,
        "lapse": lapse,
        "igt_reward_scale": scale if paradigm == "igt" else None,
        "n_configs": n_cfg,
        "tune_objective": obj,
        "dev_mean_nll": float(np.mean([v["nll"] for v in per.values()])),
        "dev_mean_top1": float(np.mean([v["top1"] for v in per.values()])),
        "cpu_seconds": time.process_time() - t0,
        "per_participant": per,
    }


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", choices=["mcpl", "igt"], required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default="reports/phase2/cognitive/human_dev")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    src = os.environ.get("HPE_PSYCH101_SUBSET")
    if not src:
        print("set HPE_PSYCH101_SUBSET")
        return 2
    cfg = yaml.safe_load(GRIDS.read_text())
    keys = partition_keys(a.paradigm)
    parts = load_participants(Path(src), a.paradigm)
    tune = [p for p in parts if p.key in keys["tune"]]
    dev = [p for p in parts if p.key in keys["dev"]]
    used = {p.key for p in tune + dev}
    assert not (used & keys["test"]), "test participants must never be loaded"
    models = {m: s for m, s in cfg["models"].items() if a.paradigm in s or s.get(a.paradigm) is not None}
    models = {m: s for m, s in models.items() if a.paradigm in s}
    if a.models:
        models = {m: s for m, s in models.items() if m in a.models}
    print(
        f"{a.paradigm}: tune={len(tune)} dev={len(dev)} (test excluded: {len(keys['test'])}); models={list(models)}"
    )
    results = {}
    for name, spec in models.items():
        r = fit_and_score(name, spec, a.paradigm, tune, dev, cfg, a.seed)
        results[name] = r
        print(
            f"  {name:22s} dev NLL={r['dev_mean_nll']:.4f} top1={r['dev_mean_top1']:.3f} "
            f"params={r['params']} lapse={r['lapse']} cfgs={r['n_configs']} cpu={r['cpu_seconds']:.0f}s",
            flush=True,
        )
    rng = np.random.default_rng(a.seed)
    dev_keys = sorted(results[next(iter(results))]["per_participant"])
    summary: dict[str, Any] = {
        "paradigm": a.paradigm,
        "n_tune": len(tune),
        "n_dev": len(dev),
        "models": {},
        "contrasts": [],
        "terciles": {},
    }
    for name, r in results.items():
        x = np.array([r["per_participant"][k]["nll"] for k in dev_keys])
        lo, hi = boot_ci(x, rng, a.n_boot)
        summary["models"][name] = {
            "dev_nll": float(x.mean()),
            "ci": [lo, hi],
            "top1": r["dev_mean_top1"],
            "params": r["params"],
            "lapse": r["lapse"],
            "igt_reward_scale": r["igt_reward_scale"],
            "n_configs": r["n_configs"],
            "cpu_seconds": r["cpu_seconds"],
        }
    ref = "ref_instance_based" if "ref_instance_based" in results else None
    for treat in [m for m in results if m.startswith("hybrid")]:
        for ctrl in [
            c for c in (ref, "developmental", "memory_only", "plain_rl") if c and c in results and c != treat
        ]:
            d = np.array(
                [
                    results[treat]["per_participant"][k]["nll"] - results[ctrl]["per_participant"][k]["nll"]
                    for k in dev_keys
                ]
            )
            lo, hi = boot_ci(d, rng, a.n_boot)
            summary["contrasts"].append(
                {
                    "treatment": treat,
                    "control": ctrl,
                    "mean_diff_nll": float(d.mean()),
                    "ci": [lo, hi],
                    "share_treat_better": float((d < 0).mean()),
                    "n": len(d),
                }
            )
    if ref and "hybrid_exemplar" in results:
        top = np.array([results[ref]["per_participant"][k]["top1"] for k in dev_keys])
        q = np.quantile(top, [1 / 3, 2 / 3])
        for label, mask in (
            ("least_predictable", top <= q[0]),
            ("middle", (top > q[0]) & (top <= q[1])),
            ("most_predictable", top > q[1]),
        ):
            row = {}
            for m in results:
                row[m] = float(
                    np.mean(
                        [
                            results[m]["per_participant"][k]["nll"]
                            for k, keep in zip(dev_keys, mask, strict=True)
                            if keep
                        ]
                    )
                )
            summary["terciles"][label] = {"n": int(mask.sum()), **row}
    out = Path(a.out) / a.paradigm
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "models": results}, indent=1) + "\n")
    lines = [
        f"# Human DEV-partition comparison — {a.paradigm.upper()} (EXPLORATORY, Phase II)",
        "",
        f"tune={len(tune)} dev={len(dev)}; test participants never loaded. Lower NLL is better.",
        "",
        "| model | dev NLL [95% CI] | top-1 | selected params | lapse | configs | CPU s |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, v in sorted(summary["models"].items(), key=lambda kv: kv[1]["dev_nll"]):
        lines.append(
            f"| {m} | {v['dev_nll']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] | {v['top1']:.3f} | `{v['params']}` | {v['lapse']} | {v['n_configs']} | {v['cpu_seconds']:.0f} |"
        )
    lines += [
        "",
        "| contrast (treatment − control, NLL; negative favours treatment) | mean | 95% CI | share better | n |",
        "|---|---|---|---|---|",
    ]
    for c in summary["contrasts"]:
        lines.append(
            f"| {c['treatment']} − {c['control']} | {c['mean_diff_nll']:+.4f} | [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | {c['share_treat_better']:.2f} | {c['n']} |"
        )
    if summary["terciles"]:
        ms = list(results)
        lines += [
            "",
            "Terciles of participant predictability (ref_instance_based top-1 on dev): mean NLL per model",
            "",
            "| tercile | n | " + " | ".join(ms) + " |",
            "|---|---|" + "---|" * len(ms),
        ]
        for label, row in summary["terciles"].items():
            lines.append(f"| {label} | {row['n']} | " + " | ".join(f"{row[m]:.3f}" for m in ms) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
