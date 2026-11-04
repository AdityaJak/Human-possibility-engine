#!/usr/bin/env python3
"""EXPLORATORY: misinformation damage and recovery under social mechanism v2 (dev seeds).

    python experiments/phase2/social/misinfo_recovery.py [--seeds 1000 ...] [--agents 100] [--years 15] [--out DIR]

Arms (policy ``developmental_social``; misinformation = 10% of adults assert wrong labels as instruction):

  v2_clean          social v2, no misinformation
  v2_misinfo_0_5    social v2, misinformation active in years [0, 5) then switched off
  v1_misinfo_0_5    Phase I social mechanism, same misinformation window

Per year we record cohort held-out accuracy (ages 10-60), and for v2 the share of demonstrations that came
from misinformers, their correctness, and mean trust toward honest vs misinforming demonstrators.
Damage = (clean − misinfo) accuracy at the end of the window (year 5); recovery = the same gap at year 15.
Paired bootstrap over seeds. Outputs: per-run JSON, summary.json, report.md, run.log.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import hpe.development.social  # noqa: F401  (registers developmental_social)
from hpe.contracts.protocols import ListSink
from hpe.contracts.registry import get_factory
from hpe.contracts.schemas import (
    Intervention,
    InterventionKind,
    InterventionTarget,
    LogConfig,
    SimulationConfig,
)
from hpe.evaluation.society import cohort_mean_accuracy
from hpe.world import World

MISINFO_WINDOW = Intervention(
    name="rumours",
    kind=InterventionKind.MISINFORMATION,
    target=InterventionTarget(age_min=18, age_max=80, fraction=0.1),
    start_year=0.0,
    end_year=5.0,
)
ARMS: dict[str, tuple[dict[str, Any], list[Intervention]]] = {
    "v2_clean": ({"social_v2": {}}, []),
    "v2_misinfo_0_5": ({"social_v2": {}}, [MISINFO_WINDOW]),
    "v1_misinfo_0_5": ({}, [MISINFO_WINDOW]),
}


def paired_bootstrap(
    t: list[float], c: list[float], n_boot: int = 4000, seed: int = 20261105
) -> dict[str, float]:
    d = np.asarray(t, dtype=float) - np.asarray(c, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "estimate": float(d.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(len(d)),
    }


def _f(x: Any) -> float:
    return float("nan") if x is None else float(x)


def run_one(arm: str, seed: int, n_agents: int, years: int) -> dict[str, Any]:
    params, interventions = ARMS[arm]
    cfg = SimulationConfig(
        seed=seed,
        n_agents=n_agents,
        years=float(years),
        policy_class="developmental",
        policy_params=dict(params),
        interventions=list(interventions),
        log=LogConfig(
            frames_every_ticks=0,
            decision_sample_rate=0.0,
            log_learning_episodes=False,
            assessment_interval_years=1.0,
            checkpoint_every_years=None,
        ),
        label=f"phase2 social misinfo {arm}",
    )
    sink = ListSink()
    c0 = time.process_time()
    w = World(
        cfg,
        get_factory("developmental_social", dict(params)),
        sink=sink,
        run_id=f"social_misinfo__{arm}__s{seed}",
    )
    yearly: list[dict[str, Any]] = []
    for y in range(1, years + 1):
        w.run(until_year=float(y))
        agents = w.agents(include_dead=False)
        row: dict[str, Any] = {
            "year": y,
            "accuracy_10_60": cohort_mean_accuracy(agents, age_min=10, age_max=60),
        }
        stats = [e.payload for e in sink.events if e.provenance.mechanism == "world.social.v2.stats"]
        if stats:
            ys = stats[-1]["year_stats"]
            row["demos"] = ys["demos"]
            row["demos_from_misinformers"] = ys["demos_from_misinformers"]
            row["misinformer_demo_share"] = ys["demos_from_misinformers"] / max(1.0, ys["demos"])
            row["trust_honest"] = _f(stats[-1]["mean_trust_honest"])
            row["trust_misinformer"] = _f(stats[-1]["mean_trust_misinformer"])
        yearly.append(row)
    return {
        "arm": arm,
        "seed": seed,
        "run_id": w.run_id,
        "event_stream_hash": w.event_stream_hash(),
        "cpu_s": round(time.process_time() - c0, 2),
        "yearly": yearly,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--out", default="reports/phase2/social/misinfo")
    a = ap.parse_args()
    for s in a.seeds:
        assert 1000 <= s <= 1099, f"dev seeds only (1000-1099): {s}"
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a", encoding="utf-8")
    results: dict[str, dict[int, dict[str, Any]]] = {}
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
            results.setdefault(arm, {})[seed] = r
            y5, y15 = r["yearly"][4], r["yearly"][-1]
            line = f"{arm:16s} s{seed} acc@5={y5['accuracy_10_60']:.4f} acc@{a.years}={y15['accuracy_10_60']:.4f} cpu={r['cpu_s']:.1f}s"
            if "misinformer_demo_share" in y5:
                line += f" misinf-share@5={y5['misinformer_demo_share']:.3f} trust(h/m)@5={y5['trust_honest']:.2f}/{y5['trust_misinformer']:.2f}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    seeds = sorted(a.seeds)

    def series(arm: str, key: str) -> list[list[float]]:
        return [
            [r["yearly"][y].get(key, float("nan")) for y in range(a.years)]
            for r in (results[arm][s] for s in seeds)
        ]

    curves = {
        arm: {"accuracy_10_60_mean": [float(x) for x in np.mean(series(arm, "accuracy_10_60"), axis=0)]}
        for arm in ARMS
    }
    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN years (no misinformers) are expected
    for arm in ("v2_clean", "v2_misinfo_0_5"):
        for key in ("misinformer_demo_share", "trust_honest", "trust_misinformer"):
            curves[arm][key + "_mean"] = [
                float(x) for x in np.nanmean(np.asarray(series(arm, key), dtype=float), axis=0)
            ]
    contrasts = []
    for name, t_arm, c_arm, year in (
        ("damage_v2_year5", "v2_clean", "v2_misinfo_0_5", 5),
        ("residual_v2_year15", "v2_clean", "v2_misinfo_0_5", a.years),
        ("damage_v1_year5", "v2_clean", "v1_misinfo_0_5", 5),
    ):
        t = [results[t_arm][s]["yearly"][year - 1]["accuracy_10_60"] for s in seeds]
        c = [results[c_arm][s]["yearly"][year - 1]["accuracy_10_60"] for s in seeds]
        contrasts.append(
            {"name": name, "treatment": t_arm, "control": c_arm, "year": year, **paired_bootstrap(t, c)}
        )
    # recovery: did the (clean - misinfo) gap shrink between year 5 and the final year? positive = recovery
    gap5 = [
        results["v2_clean"][s]["yearly"][4]["accuracy_10_60"]
        - results["v2_misinfo_0_5"][s]["yearly"][4]["accuracy_10_60"]
        for s in seeds
    ]
    gapN = [
        results["v2_clean"][s]["yearly"][-1]["accuracy_10_60"]
        - results["v2_misinfo_0_5"][s]["yearly"][-1]["accuracy_10_60"]
        for s in seeds
    ]
    contrasts.append(
        {
            "name": "recovery_v2_gap5_minus_gapFinal",
            "treatment": "gap@5",
            "control": f"gap@{a.years}",
            **paired_bootstrap(gap5, gapN),
        }
    )
    summary = {
        "label": "EXPLORATORY — dev seeds only (Phase II registry); not a confirmatory result",
        "protocol": {
            "seeds": seeds,
            "n_agents": a.agents,
            "years": a.years,
            "misinformation": MISINFO_WINDOW.model_dump(mode="json"),
        },
        "curves": curves,
        "contrasts": contrasts,
        "cpu_seconds_total": round(total_cpu, 1),
        "cpu_hours_total": round(total_cpu / 3600, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# Misinformation damage and recovery under social v2 (EXPLORATORY, dev seeds)",
        "",
        f"Seeds {seeds}; {a.agents} agents × {a.years} years; misinformation 10% of adults, years 0–5 only; CPU {summary['cpu_hours_total']} h.",
        "",
        "| year | v2 clean acc | v2 misinfo acc | v1 misinfo acc | v2 misinf. demo share | v2 trust honest | v2 trust misinformer |",
        "|---|---|---|---|---|---|---|",
    ]
    for y in range(a.years):
        m = curves["v2_misinfo_0_5"]
        tm = m["trust_misinformer_mean"][y]
        lines.append(
            f"| {y + 1} | {curves['v2_clean']['accuracy_10_60_mean'][y]:.4f} | {m['accuracy_10_60_mean'][y]:.4f} | {curves['v1_misinfo_0_5']['accuracy_10_60_mean'][y]:.4f} | {m['misinformer_demo_share_mean'][y]:.3f} | {m['trust_honest_mean'][y]:.3f} | {'—' if np.isnan(tm) else f'{tm:.3f}'} |"
        )
    lines += ["", "| contrast | estimate | 95% CI (paired, seeds) |", "|---|---|---|"]
    for ct in contrasts:
        lines.append(
            f"| {ct['name']} ({ct['treatment']} − {ct['control']}) | {ct['estimate']:+.4f} | [{ct['ci_low']:+.4f}, {ct['ci_high']:+.4f}] (n={ct['n']}) |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(contrasts) - 3 :]))
    print(f"CPU hours: {summary['cpu_hours_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
