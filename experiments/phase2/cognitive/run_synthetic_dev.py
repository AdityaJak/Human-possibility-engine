#!/usr/bin/env python3
"""Phase II EXPLORATORY synthetic dev experiment: Candidate A vs baselines on the v2 task families.

    python experiments/phase2/cognitive/run_synthetic_dev.py [--seeds 1000 1001 ...] [--agents 6]

Dev seeds only (1000–1099). One V2World per (seed, agent) so non-stationary regimes advance per agent.
Per family: 300 practice episodes (25% instructional, 75% outcome feedback, label noise 0.05),
held-out assessment every 50 episodes; ``delayed`` uses practice_delayed with k=5; ``nonstationary``
switches regime every 150 samples (assessments at 150/300 are on the new regime). Two worlds:
shared latent structure and the specialist control (independent structure per family).
Outputs: reports/phase2/cognitive/synthetic_dev/{results.json,report.md}. Exploratory only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import hpe.development  # noqa: F401  (registers developmental + hybrid_exemplar)
from hpe.contracts.conformance import make_init
from hpe.contracts.registry import get_factory
from hpe.core import stream
from hpe.tasks import assess, practice
from hpe.tasks.v2 import V2_FAMILIES, V2Config, V2World, practice_delayed

ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "hybrid_exemplar": ("hybrid_exemplar", {}),
    "hybrid_fixed_gate": ("hybrid_exemplar", {"use_learned_gate": False, "fixed_gate": 0.5}),
    "hybrid_exemplar_only": ("hybrid_exemplar", {"use_parametric": False}),
    "developmental": ("developmental", {}),
    "plain_rl": ("plain_rl", {}),
    "memory_only": ("memory_only", {}),
}
EPISODES = 300
BLOCK = 50
FULL_FRACTION = 0.25
NOISE = 0.05
DELAY_K = 5


def run_agent(arm: str, seed: int, agent: int, shared: bool) -> dict[str, list[float]]:
    cls, params = ARMS[arm]
    import hpe.evaluation.baselines  # noqa: F401  (registers static/memory/plain_rl)

    w = V2World(seed=seed, config=V2Config(shared_structure=shared))
    init = make_init(w, agent_id=f"a{agent:03d}", seed=seed * 100 + agent)  # type: ignore[arg-type]
    init.families = w.families_meta()
    pol = get_factory(cls, params)(init)
    curves: dict[str, list[float]] = {}
    for fam in V2_FAMILIES:
        rng = stream(seed, "dev_v2", arm, agent, fam)
        curve = [assess(pol, w, fam, 20.0).accuracy]
        for b in range(EPISODES // BLOCK):
            for e in range(BLOCK):
                mode = "full" if rng.random() < FULL_FRACTION else "bandit"
                if fam == "delayed":
                    practice_delayed(
                        pol,
                        w,
                        fam,
                        1,
                        rng,
                        k=DELAY_K,
                        agent_id=init.agent_id,
                        tick=b * BLOCK + e,
                        age_years=20.0,
                        mode=mode,
                        noise=NOISE,
                    )  # type: ignore[arg-type]
                else:
                    practice(
                        pol,
                        w,
                        fam,
                        1,
                        rng,
                        agent_id=init.agent_id,
                        tick=b * BLOCK + e,
                        age_years=20.0,
                        mode=mode,
                        noise=NOISE,
                    )  # type: ignore[arg-type]
            curve.append(assess(pol, w, fam, 20.0).accuracy)
        curves[fam] = curve
    return curves


def summarize(raw: dict, seeds: list[int], n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for world in raw:
        out[world] = {"final": {}, "contrasts": [], "curves": {}}
        per_seed: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARMS:
            per_seed[arm] = {}
            for fam in V2_FAMILIES:
                # mean over agents per seed, final assessment
                vals = np.array(
                    [
                        np.mean([raw[world][arm][str(s)][str(a)][fam][-1] for a in raw[world][arm][str(s)]])
                        for s in seeds
                    ]
                )
                per_seed[arm][fam] = vals
                curve = np.mean(
                    [[raw[world][arm][str(s)][str(a)][fam] for a in raw[world][arm][str(s)]] for s in seeds],
                    axis=(0, 1),
                )
                out[world]["curves"].setdefault(arm, {})[fam] = [round(float(v), 4) for v in curve]
            out[world]["final"][arm] = {
                fam: {"mean": float(per_seed[arm][fam].mean()), "sd": float(per_seed[arm][fam].std(ddof=1))}
                for fam in V2_FAMILIES
            }
        for fam in V2_FAMILIES:
            for arm in (
                "hybrid_fixed_gate",
                "hybrid_exemplar_only",
                "developmental",
                "plain_rl",
                "memory_only",
            ):
                d = per_seed["hybrid_exemplar"][fam] - per_seed[arm][fam]
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                m = d[idx].mean(axis=1)
                out[world]["contrasts"].append(
                    {
                        "family": fam,
                        "treatment": "hybrid_exemplar",
                        "control": arm,
                        "mean": float(d.mean()),
                        "ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                        "n_seeds": len(d),
                    }
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--out", default="reports/phase2/cognitive/synthetic_dev")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    assert all(1000 <= s <= 1099 for s in a.seeds), "dev seeds only"
    t0 = time.process_time()
    raw: dict[str, Any] = {}
    for world, shared in (("shared", True), ("specialist", False)):
        raw[world] = {}
        for arm in ARMS:
            raw[world][arm] = {}
            for s in a.seeds:
                raw[world][arm][str(s)] = {str(g): run_agent(arm, s, g, shared) for g in range(a.agents)}
            fin = {
                fam: np.mean(
                    [raw[world][arm][str(s)][str(g)][fam][-1] for s in a.seeds for g in range(a.agents)]
                )
                for fam in V2_FAMILIES
            }
            print(f"{world:10s} {arm:22s} " + " ".join(f"{f}={v:.3f}" for f, v in fin.items()), flush=True)
    cpu = time.process_time() - t0
    summary = summarize(raw, a.seeds, a.n_boot, np.random.default_rng(20261105))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "episodes": EPISODES,
                    "block": BLOCK,
                    "full_fraction": FULL_FRACTION,
                    "noise": NOISE,
                    "delay_k": DELAY_K,
                    "seeds": a.seeds,
                    "agents": a.agents,
                },
                "cpu_seconds": cpu,
                "summary": summary,
                "raw": raw,
            },
            indent=1,
        )
        + "\n"
    )
    lines = [
        "# Synthetic DEV experiment — Candidate A on v2 families (EXPLORATORY, Phase II)",
        "",
        f"seeds {a.seeds} (dev), {a.agents} agents/arm, {EPISODES} episodes/family, CPU {cpu:.0f} s. Held-out accuracy after the full budget (mean over seeds ± sd over seeds).",
    ]
    for world in summary:
        lines += [
            "",
            f"## {world} structure",
            "",
            "| arm | " + " | ".join(V2_FAMILIES) + " |",
            "|---|" + "---|" * len(V2_FAMILIES),
        ]
        for arm in ARMS:
            lines.append(
                f"| {arm} | "
                + " | ".join(
                    f"{summary[world]['final'][arm][f]['mean']:.3f} ± {summary[world]['final'][arm][f]['sd']:.3f}"
                    for f in V2_FAMILIES
                )
                + " |"
            )
        lines += ["", "| family | hybrid − control | mean | 95% CI (paired by seed) |", "|---|---|---|---|"]
        for c in summary[world]["contrasts"]:
            lines.append(
                f"| {c['family']} | {c['control']} | {c['mean']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
