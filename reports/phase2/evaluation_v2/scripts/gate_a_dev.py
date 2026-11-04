#!/usr/bin/env python3
"""Gate A controls on DEV seeds (EXPLORATORY): frozen-copy, permuted-feedback and age-script-only
gains for `developmental`, `hybrid_exemplar` (via the registry; its code is not read here) and
`plain_rl`, under `hpe.evaluation.learning_lab` with common random numbers.

    python reports/phase2/evaluation_v2/scripts/gate_a_dev.py [--workers 4] [--agents 20]

Protocol: Phase I v1 task world (families sort / compare / signal), learning-lab defaults (bandit
feedback, label noise 0.05, no demonstrations, 300 episodes per family in blocks of 10, ages 8 -> 18,
perceptual gate on), transfer and retention probes skipped (not needed for the gains). Every arm of a
seed sees the same items, the same feedback noise and the same exploration stream
(`hpe.core.rng` named streams keyed by seed and agent index). Dev seeds 1000-1009; the confirmatory
seeds 500-509 are refused by the shared driver.

Arms (policy class, policy params):
  developmental / developmental_frozen        developmental  {} / {use_learning: false}
  hybrid_exemplar / hybrid_exemplar_frozen    hybrid_exemplar {} / {use_learning: false}
  plain_rl / plain_rl_frozen                  plain_rl       {} / {lr: 0.0}
  age_script_only                             age_script_only {}   (H2 null: scripted gates only)
  static_persona                              static_persona {}    (reference)
Variant `permuted_feedback` (lab option feedback_permutation: true) is run for the three learners:
identical items and update calls, outcome information replaced by that of an independently sampled
item of the same family and split. Frozen copies are unaffected by the permutation (no update uses
the feedback), so they are run once. Before any run, the script verifies behaviourally that every
"frozen" arm and the age-script null leave their state hash unchanged after practice, and that
every learner changes it.

Per (variant, arm, seed): mean over the seed's agents of final_heldout_accuracy (then mean over
families) and of the tick-0 curve value. Gains are paired over seeds (paired percentile bootstrap,
4000 resamples, 95% CI; sign-flip permutation p; Cohen dz):
  frozen_copy_gain  = acc(learner) - acc(learner_frozen)                     expected >= 0.10
  permuted_gain     = acc(learner | permuted feedback) - acc(learner_frozen)  expected <= 0.02 (|CI| within margin)
  age_script_gain   = acc(learner) - acc(age_script_only)                     required >= 0.10 at Gate A
Outputs: reports/phase2/evaluation_v2/gate_a_dev/{results.json, report.md, run.log}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lab_runner import (  # noqa: E402
    DEV_SEEDS,
    Log,
    by,
    check_seeds,
    code_version,
    fmt,
    hardware,
    paired,
    run_jobs,
    summary,
)

from hpe.contracts import registry  # noqa: E402
from hpe.contracts.conformance import make_init  # noqa: E402
from hpe.core import stream  # noqa: E402
from hpe.tasks import SymbolWorld, policy_state_hash, practice  # noqa: E402

OUT_DIR = HERE.parent / "gate_a_dev"
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "developmental": ("developmental", {}),
    "developmental_frozen": ("developmental", {"use_learning": False}),
    "hybrid_exemplar": ("hybrid_exemplar", {}),
    "hybrid_exemplar_frozen": ("hybrid_exemplar", {"use_learning": False}),
    "plain_rl": ("plain_rl", {}),
    "plain_rl_frozen": ("plain_rl", {"lr": 0.0}),
    "age_script_only": ("age_script_only", {}),
    "static_persona": ("static_persona", {}),
}
LEARNERS = ["developmental", "hybrid_exemplar", "plain_rl"]
NON_LEARNING = [
    "developmental_frozen",
    "hybrid_exemplar_frozen",
    "plain_rl_frozen",
    "age_script_only",
    "static_persona",
]
PARAMS: dict[str, Any] = {
    "n_agents": 20,
    "episodes_per_family": 300,
    "block": 10,
    "checkpoint_every": 50,
    "skip_transfer": True,
    "skip_retention": True,
}
FROZEN_OF = {
    "developmental": "developmental_frozen",
    "hybrid_exemplar": "hybrid_exemplar_frozen",
    "plain_rl": "plain_rl_frozen",
}
MIN_GAIN, PERM_MARGIN = 0.10, 0.02


def freeze_check(log: Log) -> dict[str, dict[str, bool]]:
    """Behavioural check through the Policy protocol only. After 60 practice episodes with full
    feedback: do the held-out action probabilities change (the operational meaning of a frozen
    readout), and does the state hash change (stricter: bookkeeping counters such as n_updates also
    count)? A frozen arm must leave the probabilities unchanged; a learner must change them."""
    w = SymbolWorld(seed=1000)
    items = w.eval_set("compare")
    ctxs = np.stack([w.perceive(it.context, 12.0) for it in items])
    res: dict[str, dict[str, bool]] = {}
    for arm, (pc, params) in ARMS.items():
        pol = registry.get_factory(pc, params)(make_init(w))
        h = policy_state_hash(pol)
        p0 = pol.action_probabilities_batch("compare", "task", ctxs, items[0].n_actions, 12.0)
        practice(
            pol,
            w,
            "compare",
            60,
            stream(1, "freeze_check"),
            agent_id="a",
            tick=0,
            age_years=12.0,
            mode="full",
        )
        p1 = pol.action_probabilities_batch("compare", "task", ctxs, items[0].n_actions, 12.0)
        res[arm] = {
            "probabilities_changed": bool(np.abs(p1 - p0).max() > 0),
            "state_hash_changed": policy_state_hash(pol) != h,
        }
        log(
            f"freeze check {arm}: held-out probabilities "
            f"{'CHANGED' if res[arm]['probabilities_changed'] else 'unchanged'}, state hash "
            f"{'changed' if res[arm]['state_hash_changed'] else 'unchanged'} after 60 practice episodes"
        )
    bad = [a for a in NON_LEARNING if res[a]["probabilities_changed"]] + [
        a for a in LEARNERS if not res[a]["probabilities_changed"]
    ]
    if bad:
        raise SystemExit(f"freeze check failed for {bad}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--agents", type=int, default=PARAMS["n_agents"])
    ap.add_argument("--seeds", type=int, nargs="*", default=DEV_SEEDS)
    a = ap.parse_args()
    seeds = list(a.seeds)
    check_seeds(seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv = code_version()
    log = Log(OUT_DIR / "run.log")
    log(f"gate_a_dev: seeds {seeds}, agents {a.agents}, code {cv}, {hardware()}")
    params = {**PARAMS, "n_agents": a.agents}
    freeze = freeze_check(log)

    jobs: list[dict[str, Any]] = []
    for arm, (pc, pp) in ARMS.items():
        for s in seeds:
            jobs.append(
                {
                    "experiment": "gate_a_dev",
                    "arm": arm,
                    "policy_class": pc,
                    "policy_params": pp,
                    "variant": "main",
                    "seed": s,
                    "params": params,
                }
            )
    for arm in LEARNERS:
        pc, pp = ARMS[arm]
        for s in seeds:
            jobs.append(
                {
                    "experiment": "gate_a_dev",
                    "arm": arm,
                    "policy_class": pc,
                    "policy_params": pp,
                    "variant": "permuted_feedback",
                    "seed": s,
                    "params": {**params, "feedback_permutation": True},
                }
            )
    t_wall = time.perf_counter()
    results = run_jobs(jobs, a.workers, log)
    wall = time.perf_counter() - t_wall
    cpu = sum(r["cpu_s"] for r in results)
    fams = sorted(results[0]["final"])

    def vals(variant: str, arm: str, key: str = "final_mean", fam: str | None = None) -> list[float]:
        rs = by(results, variant, arm)
        return [r[key][fam] if fam else r[key] for r in rs]

    arms_summary: dict[str, Any] = {}
    for variant in ("main", "permuted_feedback"):
        for arm in ARMS:
            if not by(results, variant, arm):
                continue
            arms_summary[f"{variant}/{arm}"] = {
                "final": summary(vals(variant, arm), seed=11),
                "initial": summary(vals(variant, arm, "initial_mean"), seed=12),
                "per_family_final": {f: summary(vals(variant, arm, "final", f), seed=13) for f in fams},
                "decisions_per_seed": float(np.mean([r["decisions"] for r in by(results, variant, arm)])),
                "param_updates_per_seed": float(
                    np.mean([r["param_updates"] for r in by(results, variant, arm)])
                ),
            }

    gains: dict[str, Any] = {}
    for i, L in enumerate(LEARNERS):
        fr = FROZEN_OF[L]
        g = {
            "frozen_copy_gain": paired(vals("main", L), vals("main", fr), seed=100 + i),
            "permuted_gain": paired(vals("permuted_feedback", L), vals("main", fr), seed=200 + i),
            "learned_minus_permuted": paired(vals("main", L), vals("permuted_feedback", L), seed=300 + i),
            "age_script_gain": paired(vals("main", L), vals("main", "age_script_only"), seed=400 + i),
            "static_persona_gain": paired(vals("main", L), vals("main", "static_persona"), seed=500 + i),
            "per_family": {
                f: {
                    "frozen_copy_gain": paired(
                        vals("main", L, "final", f), vals("main", fr, "final", f), seed=600 + i
                    ),
                    "permuted_gain": paired(
                        vals("permuted_feedback", L, "final", f), vals("main", fr, "final", f), seed=700 + i
                    ),
                    "age_script_gain": paired(
                        vals("main", L, "final", f), vals("main", "age_script_only", "final", f), seed=800 + i
                    ),
                }
                for f in fams
            },
        }
        fc, pg, ag = g["frozen_copy_gain"], g["permuted_gain"], g["age_script_gain"]
        g["dev_pattern"] = {
            "frozen_copy_gain_ge_0.10_ci_excl_0": bool(fc["estimate"] >= MIN_GAIN and fc["ci_low"] > 0),
            "permuted_gain_within_±0.02": bool(-PERM_MARGIN < pg["ci_low"] and pg["ci_high"] < PERM_MARGIN),
            "permuted_gain_point_le_0.02": bool(pg["estimate"] <= PERM_MARGIN),
            "age_script_gain_ge_0.10_ci_excl_0": bool(ag["estimate"] >= MIN_GAIN and ag["ci_low"] > 0),
        }
        gains[L] = g

    out = {
        "label": "EXPLORATORY: dev seeds 1000-1009; nothing here is a confirmatory result",
        "what": __doc__.strip().splitlines()[0],
        "seeds": seeds,
        "params": params,
        "arms": {k: {"policy_class": v[0], "policy_params": v[1]} for k, v in ARMS.items()},
        "variants": {"main": {}, "permuted_feedback": {"feedback_permutation": True, "arms": LEARNERS}},
        "freeze_check": freeze,
        "definitions": {
            "frozen_copy_gain": "acc(learner) - acc(learner with learning disabled), paired by seed",
            "permuted_gain": "acc(learner under permuted feedback) - acc(frozen copy), paired by seed",
            "learned_minus_permuted": "acc(learner) - acc(learner under permuted feedback)",
            "age_script_gain": "acc(learner) - acc(age_script_only)",
            "acc": "final held-out accuracy, mean over families (sort, compare, signal) and over agents",
            "thresholds": {"min_gain": MIN_GAIN, "permuted_margin": PERM_MARGIN},
        },
        "arm_summaries": arms_summary,
        "gains": gains,
        "runs": results,
        "compute": {
            "cpu_s_total": round(cpu, 1),
            "wall_s": round(wall, 1),
            "n_jobs": len(jobs),
            "workers": a.workers,
            "hardware": hardware(),
        },
        "code_version": cv,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Gate A controls on DEV seeds (exploratory)",
        "",
        f"Seeds {seeds[0]}-{seeds[-1]} (n={len(seeds)}), {a.agents} agents per seed, v1 families {', '.join(fams)}; "
        "learning-lab defaults, transfer/retention skipped. Produced by `scripts/gate_a_dev.py`; "
        f"code `{out['code_version']}`; CPU {cpu / 60:.1f} min over {len(jobs)} runs. "
        "**Dev seeds only — not a confirmatory result.**",
        "",
        "## Held-out accuracy (mean over families; mean over seeds, 95% bootstrap CI)",
        "",
        "| variant / arm | initial (tick 0) | final | updates / seed |",
        "|---|---|---|---|",
    ]
    for k, s in arms_summary.items():
        lines.append(
            f"| {k} | {s['initial']['mean']:.3f} | {fmt(s['final'])} | {s['param_updates_per_seed']:.0f} |"
        )
    lines += [
        "",
        "## Gains (paired over seeds; estimate [95% CI], Cohen dz, sign-flip p)",
        "",
        "| learner | frozen-copy gain (≥0.10 expected) | permuted-feedback gain (≤0.02 expected) | learned − permuted | age-script-only gain (≥0.10 required) | static-persona gain |",
        "|---|---|---|---|---|---|",
    ]
    for L, g in gains.items():
        cells = []
        for key in (
            "frozen_copy_gain",
            "permuted_gain",
            "learned_minus_permuted",
            "age_script_gain",
            "static_persona_gain",
        ):
            d = g[key]
            cells.append(f"{fmt(d, 'estimate')} dz={d['cohen_dz']:.2f} p={d['p_signflip']:.3g}")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Dev pattern against the Gate A thresholds (exploratory, dev seeds)",
        "",
        "| learner | frozen-copy gain ≥ 0.10, CI > 0 | permuted gain CI within ±0.02 | permuted point ≤ 0.02 | age-script gain ≥ 0.10, CI > 0 |",
        "|---|---|---|---|---|",
    ]
    for L, g in gains.items():
        p = g["dev_pattern"]
        lines.append(
            f"| {L} | {p['frozen_copy_gain_ge_0.10_ci_excl_0']} | {p['permuted_gain_within_±0.02']} | "
            f"{p['permuted_gain_point_le_0.02']} | {p['age_script_gain_ge_0.10_ci_excl_0']} |"
        )
    lines += [
        "",
        "## Per-family gains",
        "",
        "| learner | family | frozen-copy gain | permuted gain | age-script gain |",
        "|---|---|---|---|---|",
    ]
    for L, g in gains.items():
        for f in fams:
            pf = g["per_family"][f]
            lines.append(
                f"| {L} | {f} | {fmt(pf['frozen_copy_gain'], 'estimate')} | {fmt(pf['permuted_gain'], 'estimate')} | {fmt(pf['age_script_gain'], 'estimate')} |"
            )
    lines += [
        "",
        "Freeze check (held-out action probabilities changed after 60 practice episodes with full feedback): "
        + ", ".join(f"{k}={'yes' if v['probabilities_changed'] else 'no'}" for k, v in freeze.items()),
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    log(f"wrote {OUT_DIR}/results.json and report.md; cpu {cpu:.0f}s wall {wall:.0f}s")
    for L, g in gains.items():
        log(
            f"{L}: frozen-copy {fmt(g['frozen_copy_gain'], 'estimate')} permuted {fmt(g['permuted_gain'], 'estimate')} "
            f"age-script {fmt(g['age_script_gain'], 'estimate')}"
        )


if __name__ == "__main__":
    main()
