#!/usr/bin/env python3
"""EXPLORATORY: how much knowledge do demonstrations actually transfer? (dev seeds, v1 vs v2)

    python experiments/phase2/social/transfer_audit.py [--seeds 1000 1001 1002] [--agents 100] [--years 15] [--out DIR]

Measurement-only hooks (they consume no randomness and change no state): every demonstration delivered to
a learner is recorded with the learner's probability of the *correct* action before and after the update,
its modal action before/after, whether the demonstrated action was correct, and the feedback mode. At the
end of the run each (agent, family) cell pairs the demonstrations it received with its own practice
episodes and its held-out measured accuracy.

Reported per arm: demonstrations per agent-year; share with instructional (full) feedback; share whose
demonstrated action was correct; mean information gain (nats on the correct action); flip rate (modal action
wrong -> correct because of the demonstration) and anti-flip rate (correct -> wrong); the share of
(agent, family) cells in which demonstrations outnumber practice episodes; and cells learned *only* through
demonstration (zero practice, >=1 demonstration) with their measured accuracy relative to chance.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

import hpe.development.social  # noqa: F401  (registers developmental_social)
from hpe.contracts.registry import get_factory
from hpe.contracts.runtime import Demonstration, LearningUpdateRecord, Policy
from hpe.contracts.schemas import LogConfig, SimulationConfig
from hpe.tasks import TRAINING_FAMILIES, SymbolWorld
from hpe.world import World

ARMS: dict[str, dict[str, Any]] = {"v1_connected": {}, "v2_connected": {"social_v2": {}}}


class _HookedSymbolWorld(SymbolWorld):
    """Same layout as SymbolWorld (no extra slots) so an instance can be re-classed in place; it only
    remembers the last item sampled, which is the item of the demonstration that follows."""

    __slots__ = ()
    last: ClassVar[dict[str, Any]] = {}

    def sample_item(self, *args: Any, **kwargs: Any) -> Any:
        item = super().sample_item(*args, **kwargs)
        _HookedSymbolWorld.last["item"] = item
        return item


def install_hooks(w: World, records: list[dict[str, Any]]) -> None:
    sw = w.symbol_world
    sw.__class__ = _HookedSymbolWorld
    last = _HookedSymbolWorld.last
    last.clear()

    def wrap(aid: str, pol: Policy) -> Policy:
        orig_learn = pol.learn_from_demonstration

        def learn_from_demonstration(demo: Demonstration) -> LearningUpdateRecord | None:
            item = last.get("item")
            correct = int(item.correct_action) if item is not None else -1
            before = np.asarray(pol.action_probabilities(demo.observation), dtype=float)
            out = orig_learn(demo)
            after = np.asarray(pol.action_probabilities(demo.observation), dtype=float)
            records.append(
                {
                    "learner": aid,
                    "tick": int(demo.observation.tick),
                    "family": demo.observation.family,
                    "mode": demo.feedback.mode,
                    "demonstrator": demo.demonstrator_id,
                    "trust": float(demo.trust),
                    "demo_correct": bool(demo.action == correct),
                    "p_before": float(before[correct]) if correct >= 0 else float("nan"),
                    "p_after": float(after[correct]) if correct >= 0 else float("nan"),
                    "modal_before_correct": bool(int(np.argmax(before)) == correct),
                    "modal_after_correct": bool(int(np.argmax(after)) == correct),
                }
            )
            return out

        pol.learn_from_demonstration = learn_from_demonstration  # type: ignore[method-assign]
        return pol

    orig_make = w._make_policy

    def make_policy(aid: str) -> Policy:
        return wrap(aid, orig_make(aid))

    w._make_policy = make_policy  # type: ignore[method-assign]
    for aid, pol in list(w._policies.items()):
        w._policies[aid] = wrap(aid, pol)


def run_one(arm: str, seed: int, n_agents: int, years: float) -> dict[str, Any]:
    params = ARMS[arm]
    cfg = SimulationConfig(
        seed=seed,
        n_agents=n_agents,
        years=years,
        policy_class="developmental",
        policy_params=dict(params),
        log=LogConfig(
            frames_every_ticks=0,
            decision_sample_rate=0.0,
            log_learning_episodes=False,
            assessment_interval_years=1.0,
            checkpoint_every_years=None,
        ),
        label=f"phase2 social transfer audit {arm}",
    )
    c0 = time.process_time()
    w = World(
        cfg, get_factory("developmental_social", dict(params)), run_id=f"social_transfer__{arm}__s{seed}"
    )
    records: list[dict[str, Any]] = []
    install_hooks(w, records)
    w.run()
    cpu = time.process_time() - c0
    agents = w.agents(include_dead=False)
    agent_years = max(1e-9, w._compute["agent_years"])
    n = len(records)
    meta = w.symbol_world.families_meta()
    chance = {f: 1.0 / meta[f]["n_actions"] for f in TRAINING_FAMILIES}
    gains = [
        math.log(max(1e-12, r["p_after"]) / max(1e-12, r["p_before"]))
        for r in records
        if not math.isnan(r["p_before"])
    ]
    by_cell: dict[tuple[str, str], int] = {}
    for r in records:
        by_cell[(r["learner"], r["family"])] = by_cell.get((r["learner"], r["family"]), 0) + 1
    cells: list[dict[str, Any]] = []
    for a in agents:
        if a.age_years < 10:
            continue
        for fam in TRAINING_FAMILIES:
            cap = a.capabilities.get(fam)
            if cap is None or cap.measured_accuracy is None:
                continue
            cells.append(
                {
                    "demos": by_cell.get((a.agent_id, fam), 0),
                    "practice": int(cap.practice_episodes),
                    "accuracy": float(cap.measured_accuracy),
                    "above_chance": float(cap.measured_accuracy) - chance[fam],
                }
            )
    demo_dominant = [c for c in cells if c["demos"] > c["practice"]]
    practice_dominant = [c for c in cells if c["demos"] <= c["practice"]]
    demo_only = [c for c in cells if c["practice"] == 0 and c["demos"] > 0]

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "arm": arm,
        "seed": seed,
        "run_id": w.run_id,
        "cpu_s": round(cpu, 2),
        "agent_years": round(agent_years, 2),
        "n_demos": n,
        "demos_per_agent_year": n / agent_years,
        "share_full_feedback": mean([1.0 if r["mode"] == "full" else 0.0 for r in records]),
        "share_demo_correct": mean([1.0 if r["demo_correct"] else 0.0 for r in records]),
        "mean_info_gain_nats": mean(gains),
        "mean_trust": mean([r["trust"] for r in records]),
        "flip_rate": mean(
            [1.0 if (not r["modal_before_correct"] and r["modal_after_correct"]) else 0.0 for r in records]
        ),
        "anti_flip_rate": mean(
            [1.0 if (r["modal_before_correct"] and not r["modal_after_correct"]) else 0.0 for r in records]
        ),
        "flips_per_agent_year": sum(
            1 for r in records if not r["modal_before_correct"] and r["modal_after_correct"]
        )
        / agent_years,
        "cells": len(cells),
        "share_cells_demo_dominant": len(demo_dominant) / max(1, len(cells)),
        "acc_above_chance_demo_dominant": mean([c["above_chance"] for c in demo_dominant]),
        "acc_above_chance_practice_dominant": mean([c["above_chance"] for c in practice_dominant]),
        "cells_demo_only": len(demo_only),
        "acc_above_chance_demo_only": mean([c["above_chance"] for c in demo_only]),
        "mean_demos_per_cell": mean([float(c["demos"]) for c in cells]),
        "mean_practice_per_cell": mean([float(c["practice"]) for c in cells]),
    }


KEYS = (
    "demos_per_agent_year",
    "share_full_feedback",
    "share_demo_correct",
    "mean_info_gain_nats",
    "mean_trust",
    "flip_rate",
    "anti_flip_rate",
    "flips_per_agent_year",
    "share_cells_demo_dominant",
    "acc_above_chance_demo_dominant",
    "acc_above_chance_practice_dominant",
    "cells_demo_only",
    "acc_above_chance_demo_only",
    "mean_demos_per_cell",
    "mean_practice_per_cell",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--years", type=float, default=15.0)
    ap.add_argument("--out", default="reports/phase2/social/transfer")
    a = ap.parse_args()
    for s in a.seeds:
        assert 1000 <= s <= 1099, f"dev seeds only (1000-1099): {s}"
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a", encoding="utf-8")
    results: dict[str, list[dict[str, Any]]] = {}
    total_cpu = 0.0
    for arm in ARMS:
        for seed in a.seeds:
            p = out / "runs" / f"{arm}__s{seed}.json"
            if p.exists():
                r = json.loads(p.read_text())
            else:
                r = run_one(arm, seed, a.agents, a.years)
                p.write_text(json.dumps(r, indent=1) + "\n")
            total_cpu += r["cpu_s"]
            results.setdefault(arm, []).append(r)
            line = (
                f"{arm:14s} s{seed} demos/ay={r['demos_per_agent_year']:.2f} full={r['share_full_feedback']:.2f} "
                f"correct={r['share_demo_correct']:.2f} gain={r['mean_info_gain_nats']:.3f} flip={r['flip_rate']:.3f} "
                f"anti={r['anti_flip_rate']:.3f} demo-dominant cells={r['share_cells_demo_dominant']:.3f} "
                f"demo-only cells={r['cells_demo_only']} cpu={r['cpu_s']:.1f}s"
            )
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    summary: dict[str, Any] = {
        "label": "EXPLORATORY — dev seeds only (Phase II registry); measurement hooks only, no mechanism change",
        "protocol": {"seeds": a.seeds, "n_agents": a.agents, "years": a.years, "arms": ARMS},
        "arms": {
            arm: {
                k: {"mean": float(np.nanmean([r[k] for r in rs])), "values": [r[k] for r in rs]} for k in KEYS
            }
            for arm, rs in results.items()
        },
        "cpu_seconds_total": round(total_cpu, 1),
        "cpu_hours_total": round(total_cpu / 3600, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# Knowledge transfer audit — Phase I vs social v2 (EXPLORATORY, dev seeds)",
        "",
        f"Seeds {a.seeds}; {a.agents} agents × {a.years:g} years; CPU {summary['cpu_hours_total']} h.",
        "",
        "| measure | " + " | ".join(results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for k in KEYS:
        lines.append(
            f"| {k} | " + " | ".join(f"{summary['arms'][arm][k]['mean']:.4f}" for arm in results) + " |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"CPU hours: {summary['cpu_hours_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
