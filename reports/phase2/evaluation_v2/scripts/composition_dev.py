#!/usr/bin/env python3
"""Gate B family 3 (`composition`) on DEV seeds (EXPLORATORY): plain_rl, memory_only, developmental
and hybrid_exemplar (via the registry) on the shared-structure world and the specialist control.

    python reports/phase2/evaluation_v2/scripts/composition_dev.py [--workers 4] [--agents 20]

Protocol: `hpe.evaluation.learning_lab` with task_world "composition", families primitive_a,
primitive_b, composition, composition_pair practised interleaved (blocks of 10, 300 episodes per
family), bandit feedback, label noise 0.05, no demonstrations, ages 8 -> 18, perceptual gate on,
transfer/retention skipped (v1-only probes). Common random numbers across arms within a seed.
Worlds: `shared` (composition uses the taught primitives) and `control` (shared_structure=false:
the composed primitives are drawn independently; the specialist-should-win control). Dev seeds
1000-1009 only.

Reported per (world, arm, family): final held-out accuracy (mean over agents; mean over seeds with
95% bootstrap CI), tick-0 accuracy, the mean learning curve, and for the composition families the
paired shared - control difference. The measured linear-probe ceiling from
composition_ceiling.json is shown next to the learners (spec §1: if no model exceeds the ceiling
the family is uninformative and must be reported as such).
Outputs: reports/phase2/evaluation_v2/composition_dev/{results.json, report.md, run.log}.
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

OUT_DIR = HERE.parent / "composition_dev"
CEILING = HERE.parent / "composition_ceiling.json"
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "plain_rl": ("plain_rl", {}),
    "memory_only": ("memory_only", {}),
    "developmental": ("developmental", {}),
    "hybrid_exemplar": ("hybrid_exemplar", {}),
}
WORLDS: dict[str, dict[str, Any]] = {"shared": {}, "control": {"shared_structure": False}}
FAMILIES = ["primitive_a", "primitive_b", "composition", "composition_pair"]
PARAMS: dict[str, Any] = {
    "task_world": "composition",
    "families": FAMILIES,
    "n_agents": 20,
    "episodes_per_family": 300,
    "block": 10,
    "checkpoint_every": 50,
    "skip_transfer": True,
    "skip_retention": True,
}


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
    log(f"composition_dev: seeds {seeds}, agents {a.agents}, code {cv}, {hardware()}")
    jobs: list[dict[str, Any]] = []
    for world, tc in WORLDS.items():
        for arm, (pc, pp) in ARMS.items():
            for s in seeds:
                jobs.append(
                    {
                        "experiment": "composition_dev",
                        "arm": arm,
                        "policy_class": pc,
                        "policy_params": pp,
                        "variant": world,
                        "seed": s,
                        "params": {**PARAMS, "n_agents": a.agents, "task_config": tc},
                    }
                )
    t_wall = time.perf_counter()
    results = run_jobs(jobs, a.workers, log)
    wall = time.perf_counter() - t_wall
    cpu = sum(r["cpu_s"] for r in results)

    def vals(world: str, arm: str, fam: str, key: str = "final") -> list[float]:
        return [r[key][fam] for r in by(results, world, arm)]

    table: dict[str, Any] = {}
    for world in WORLDS:
        for arm in ARMS:
            rs = by(results, world, arm)
            ticks = sorted(rs[0]["curve"][FAMILIES[0]], key=int)
            table[f"{world}/{arm}"] = {
                f: {
                    "final": summary(vals(world, arm, f), seed=21),
                    "initial": summary(vals(world, arm, f, "initial"), seed=22),
                    "curve_mean": {t: float(np.mean([r["curve"][f][t] for r in rs])) for t in ticks},
                }
                for f in FAMILIES
            }
            table[f"{world}/{arm}"]["leakage_holdout_symbols_in_practice"] = float(
                sum(r["leakage_holdout_symbols_in_practice"] or 0.0 for r in rs)
            )
    shared_minus_control = {
        arm: {
            f: paired(vals("shared", arm, f), vals("control", arm, f), seed=31 + i)
            for f in ("composition", "composition_pair")
        }
        for i, arm in enumerate(ARMS)
    }
    ceiling: dict[str, Any] | None = json.loads(CEILING.read_text()) if CEILING.exists() else None
    informative: dict[str, Any] = {}
    if ceiling:
        for world in WORLDS:
            lin = ceiling["ceiling_linear_raw_composition"][world]["mean_of_best_linear_probe"]
            informative[world] = {
                "linear_ceiling_mean": lin,
                "arms_whose_composition_ci_low_exceeds_linear_ceiling": [
                    arm for arm in ARMS if table[f"{world}/{arm}"]["composition"]["final"]["ci_low"] > lin
                ],
            }
    out = {
        "label": "EXPLORATORY: dev seeds 1000-1009; nothing here is a confirmatory result",
        "what": __doc__.strip().splitlines()[0],
        "seeds": seeds,
        "params": {**PARAMS, "n_agents": a.agents},
        "worlds": WORLDS,
        "arms": {k: {"policy_class": v[0], "policy_params": v[1]} for k, v in ARMS.items()},
        "table": table,
        "shared_minus_control": shared_minus_control,
        "informativeness_vs_linear_ceiling": informative,
        "ceiling_file": str(CEILING.name) if ceiling else None,
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
        "# `composition` family on DEV seeds (exploratory)",
        "",
        f"Seeds {seeds[0]}-{seeds[-1]} (n={len(seeds)}), {a.agents} agents per seed, 300 episodes per family, "
        "bandit feedback, label noise 0.05. Produced by `scripts/composition_dev.py`; "
        f"code `{out['code_version']}`; CPU {cpu / 60:.1f} min over {len(jobs)} runs. "
        "**Dev seeds only — not a confirmatory result.** Chance is 0.5 on every family (eval sets are exactly balanced).",
        "",
    ]
    if ceiling:
        lines += [
            "Measured probe ceilings for `composition` (from `composition_ceiling.json`, eval-set accuracy, mean over seeds):",
            "",
        ]
        for world in WORLDS:
            s = ceiling["summary"][world]
            lines.append(
                f"- {world}: "
                + ", ".join(
                    f"{pr} {s[pr]['composition']['mean']:.3f} [{s[pr]['composition']['ci_low']:.3f}, {s[pr]['composition']['ci_high']:.3f}]"
                    for pr in ("lstsq_raw", "logistic_raw", "logistic_quad", "oracle_prims")
                )
            )
        lines.append("")
    for world in WORLDS:
        lines += [
            f"## World `{world}`"
            + (
                " (specialist control: composed primitives independent of the taught ones)"
                if world == "control"
                else " (composition uses the taught primitives)"
            ),
            "",
            "| arm | " + " | ".join(FAMILIES) + " |",
            "|---|" + "---|" * len(FAMILIES),
        ]
        for arm in ARMS:
            t = table[f"{world}/{arm}"]
            lines.append(
                f"| {arm} | "
                + " | ".join(f"{fmt(t[f]['final'])} (t0 {t[f]['initial']['mean']:.2f})" for f in FAMILIES)
                + " |"
            )
        lines.append("")
    lines += [
        "## Shared − control (paired over seeds)",
        "",
        "| arm | composition | composition_pair |",
        "|---|---|---|",
    ]
    for arm, d in shared_minus_control.items():
        lines.append(
            f"| {arm} | {fmt(d['composition'], 'estimate')} dz={d['composition']['cohen_dz']:.2f} | "
            f"{fmt(d['composition_pair'], 'estimate')} dz={d['composition_pair']['cohen_dz']:.2f} |"
        )
    if informative:
        lines += ["", "## Informativeness against the measured linear ceiling", ""]
        for world, d in informative.items():
            arms_over = d["arms_whose_composition_ci_low_exceeds_linear_ceiling"]
            lines.append(
                f"- {world}: linear ceiling {d['linear_ceiling_mean']:.3f}; arms whose composition CI lies entirely above it: "
                + (
                    ", ".join(arms_over)
                    if arms_over
                    else "**none** (family uninformative for these learners at this budget)"
                )
            )
    lines.append("")
    (OUT_DIR / "report.md").write_text("\n".join(lines))
    log(f"wrote {OUT_DIR}/results.json and report.md; cpu {cpu:.0f}s wall {wall:.0f}s")
    for world in WORLDS:
        for arm in ARMS:
            t = table[f"{world}/{arm}"]
            log(f"{world}/{arm}: " + ", ".join(f"{f}={t[f]['final']['mean']:.3f}" for f in FAMILIES))


if __name__ == "__main__":
    main()
