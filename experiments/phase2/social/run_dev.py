#!/usr/bin/env python3
"""EXPLORATORY dev-partition experiment for social mechanism v2 (Candidate E).

    python experiments/phase2/social/run_dev.py [--seeds 1000 1001 ...] [--agents 100] [--years 15] [--out DIR]

Arms (all use the ``developmental_social`` policy, which equals the Phase I developmental learner
unless the world flag ``social_v2`` is set):

  v1_connected       Phase I social mechanism (flag absent)
  v1_isolated        Phase I + full social isolation intervention
  v2_connected       social mechanism v2
  v2_isolated        v2 + full social isolation
  v2_reward_only     v2 with information=False  (social-reward-only control: meetings, ties, trust
                     game, but no demonstrations or cooperation items)
  v2_no_learn_signal v2 with learn_signal=False (socialising reports no learning-progress term)
  v1_misinfo         Phase I + misinformation (10% of adults)
  v2_misinfo         v2 + misinformation (10% of adults)

Seeds must be dev seeds (1000-1099). Every run is deterministic. Outputs: per-run JSON, summary.json,
report.md, run.log (CPU seconds). Twin arms share seed -> population_seed -> task_seed, so contrasts are
paired by seed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
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
from hpe.evaluation.society import cohort_mean_accuracy, mean_friends
from hpe.world import World

ISOLATION = Intervention(
    name="isolation",
    kind=InterventionKind.SOCIAL_ISOLATION,
    target=InterventionTarget(fraction=1.0),
    start_year=0.0,
)
MISINFO = Intervention(
    name="rumours",
    kind=InterventionKind.MISINFORMATION,
    target=InterventionTarget(age_min=18, age_max=80, fraction=0.1),
    start_year=0.0,
)

ARMS: dict[str, tuple[dict[str, Any], list[Intervention]]] = {
    "v1_connected": ({}, []),
    "v1_isolated": ({}, [ISOLATION]),
    "v2_connected": ({"social_v2": {}}, []),
    "v2_isolated": ({"social_v2": {}}, [ISOLATION]),
    "v2_reward_only": ({"social_v2": {"information": False}}, []),
    "v2_no_learn_signal": ({"social_v2": {"learn_signal": False}}, []),
    "v1_misinfo": ({}, [MISINFO]),
    "v2_misinfo": ({"social_v2": {}}, [MISINFO]),
}

CONTRASTS: list[tuple[str, str, str, str]] = [
    ("connection_effect_v2", "v2_connected", "v2_isolated", "accuracy_10_60"),
    ("connection_effect_v1", "v1_connected", "v1_isolated", "accuracy_10_60"),
    ("information_effect_v2", "v2_connected", "v2_reward_only", "accuracy_10_60"),
    ("v2_vs_v1_connected", "v2_connected", "v1_connected", "accuracy_10_60"),
    ("learn_signal_effect", "v2_connected", "v2_no_learn_signal", "accuracy_10_60"),
    ("youth_connection_effect_v2", "v2_connected", "v2_isolated", "youth_accuracy_10_20"),
    ("youth_information_effect_v2", "v2_connected", "v2_reward_only", "youth_accuracy_10_20"),
    ("misinformation_harm_v2", "v2_connected", "v2_misinfo", "accuracy_10_60"),
    ("misinformation_harm_v1", "v1_connected", "v1_misinfo", "accuracy_10_60"),
    ("friends_v2_vs_v1", "v2_connected", "v1_connected", "mean_friends"),
    ("friends_v2_vs_reward_only", "v2_connected", "v2_reward_only", "mean_friends"),
    ("socialize_share_v2_vs_v1", "v2_connected", "v1_connected", "socialize_share"),
    ("socialize_share_learn_signal", "v2_connected", "v2_no_learn_signal", "socialize_share"),
]


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


def run_one(arm: str, seed: int, n_agents: int, years: float) -> dict[str, Any]:
    params, interventions = ARMS[arm]
    cfg = SimulationConfig(
        seed=seed,
        n_agents=n_agents,
        years=years,
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
        label=f"phase2 social dev {arm}",
    )
    sink = ListSink()
    t0, c0 = time.perf_counter(), time.process_time()
    w = World(
        cfg,
        get_factory("developmental_social", dict(params)),
        sink=sink,
        run_id=f"social_dev__{arm}__s{seed}",
    )
    w.run()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    agents = w.agents(include_dead=False)
    alive_adults = [a for a in agents if a.age_years >= 10]
    social_share = (
        float(np.mean([1.0 if a.activity == "socialize" else 0.0 for a in alive_adults]))
        if alive_adults
        else float("nan")
    )
    v2_stats = [e.payload for e in sink.events if e.provenance.mechanism == "world.social.v2.stats"]
    cum = v2_stats[-1]["cumulative"] if v2_stats else None
    agent_years = max(1e-9, w._compute["agent_years"])
    out: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "run_id": w.run_id,
        "event_stream_hash": w.event_stream_hash(),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "agent_years": round(agent_years, 2),
        "metrics": {
            "accuracy_10_60": cohort_mean_accuracy(agents, age_min=10, age_max=60),
            "youth_accuracy_10_20": cohort_mean_accuracy(agents, age_min=10, age_max=20),
            "mean_friends": mean_friends(agents, age_min=10, age_max=80),
            "socialize_share": social_share,
            "population": float(len(agents)),
        },
        "v2": None,
    }
    if cum is not None:
        last = v2_stats[-1]
        out["v2"] = {
            "cumulative": cum,
            "demos_per_agent_year": cum["demos"] / agent_years,
            "meetings_per_agent_year": cum["meetings"] / agent_years,
            "repeat_meeting_share": cum["repeat_meetings"] / max(1.0, cum["meetings"]),
            "instructional_share": cum["demos_full"] / max(1.0, cum["demos"]),
            "demo_correct_share": cum["demos_correct"] / max(1.0, cum["demos"]),
            "mean_info_gain": cum["info_gain_sum"] / max(1.0, cum["info_gain_n"]),
            "cooperation_per_agent_year": cum["cooperation_items"] / agent_years,
            "friend_ties_final": last["friend_ties"],
            "acquaintance_ties_final": last["acquaintance_ties"],
            "trust_honest_series": w._sv2_state.trust_honest,
            "trust_misinformer_series": w._sv2_state.trust_misinformer,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--years", type=float, default=15.0)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="reports/phase2/social/dev")
    a = ap.parse_args()
    for s in a.seeds:
        assert 1000 <= s <= 1099, f"dev seeds only (1000-1099): {s}"
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a", encoding="utf-8")
    results: dict[str, dict[int, dict[str, Any]]] = {}
    total_cpu = 0.0
    for arm in a.arms:
        for seed in a.seeds:
            p = out / "runs" / f"{arm}__s{seed}.json"
            if p.exists():
                r = json.loads(p.read_text())
            else:
                r = run_one(arm, seed, a.agents, a.years)
                p.write_text(json.dumps(r, indent=1) + "\n")
            total_cpu += r["cpu_s"]
            results.setdefault(arm, {})[seed] = r
            line = f"{arm:20s} s{seed} acc10_60={r['metrics']['accuracy_10_60']:.4f} friends={r['metrics']['mean_friends']:.3f} soc={r['metrics']['socialize_share']:.3f} cpu={r['cpu_s']:.1f}s"
            if r["v2"]:
                line += f" demos/ay={r['v2']['demos_per_agent_year']:.2f} gain={r['v2']['mean_info_gain']:.3f} repeat={r['v2']['repeat_meeting_share']:.2f}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    arms_summary: dict[str, Any] = {}
    for arm, by_seed in results.items():
        seeds = sorted(by_seed)
        m: dict[str, Any] = {}
        for key in (
            "accuracy_10_60",
            "youth_accuracy_10_20",
            "mean_friends",
            "socialize_share",
            "population",
        ):
            vals = [by_seed[s]["metrics"][key] for s in seeds]
            m[key] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        if all(by_seed[s]["v2"] for s in seeds):
            for key in (
                "demos_per_agent_year",
                "meetings_per_agent_year",
                "repeat_meeting_share",
                "instructional_share",
                "demo_correct_share",
                "mean_info_gain",
                "cooperation_per_agent_year",
                "friend_ties_final",
            ):
                vals = [by_seed[s]["v2"][key] for s in seeds]
                m[key] = {"mean": float(np.mean(vals)), "values": vals}
            th = [by_seed[s]["v2"]["trust_honest_series"][-1] for s in seeds]
            tm = [by_seed[s]["v2"]["trust_misinformer_series"][-1] for s in seeds]
            m["trust_honest_final"] = {"mean": float(np.nanmean(th)), "values": th}
            m["trust_misinformer_final"] = {
                "mean": float(np.nanmean(tm)) if not all(math.isnan(x) for x in tm) else None,
                "values": tm,
            }
        arms_summary[arm] = {"n_seeds": len(seeds), "metrics": m}
    contrasts = []
    for name, t_arm, c_arm, metric in CONTRASTS:
        if t_arm not in results or c_arm not in results:
            continue
        seeds = sorted(set(results[t_arm]) & set(results[c_arm]))
        t = [results[t_arm][s]["metrics"][metric] for s in seeds]
        c = [results[c_arm][s]["metrics"][metric] for s in seeds]
        if any(math.isnan(x) for x in t + c):
            continue
        contrasts.append(
            {"name": name, "treatment": t_arm, "control": c_arm, "metric": metric, **paired_bootstrap(t, c)}
        )
    summary = {
        "label": "EXPLORATORY — dev seeds only (Phase II registry); not a confirmatory result",
        "protocol": {
            "seeds": a.seeds,
            "n_agents": a.agents,
            "years": a.years,
            "policy": "developmental_social",
            "arms": {
                k: {"policy_params": v[0], "interventions": [i.model_dump(mode="json") for i in v[1]]}
                for k, v in ARMS.items()
                if k in a.arms
            },
        },
        "arms": arms_summary,
        "contrasts": contrasts,
        "cpu_seconds_total": round(total_cpu, 1),
        "cpu_hours_total": round(total_cpu / 3600, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# Social mechanism v2 — dev-partition results (EXPLORATORY)",
        "",
        f"Seeds {a.seeds}; {a.agents} agents × {a.years:g} years; policy `developmental_social`; CPU {summary['cpu_hours_total']} h.",
        "",
        "| arm | n | accuracy 10–60 | youth 10–20 | mean friends | socialise share | demos/agent-yr | repeat share | instructional | info gain | trust honest | trust misinf. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, s in arms_summary.items():
        m = s["metrics"]

        def g(k: str, m: dict[str, Any] = m) -> str:
            return f"{m[k]['mean']:.3f}" if k in m and m[k]["mean"] is not None else "—"

        lines.append(
            f"| {arm} | {s['n_seeds']} | {g('accuracy_10_60')} | {g('youth_accuracy_10_20')} | {g('mean_friends')} | {g('socialize_share')} | {g('demos_per_agent_year')} | {g('repeat_meeting_share')} | {g('instructional_share')} | {g('mean_info_gain')} | {g('trust_honest_final')} | {g('trust_misinformer_final')} |"
        )
    lines += [
        "",
        "| contrast | treatment − control | metric | estimate | 95% CI (paired, seeds) |",
        "|---|---|---|---|---|",
    ]
    for ct in contrasts:
        lines.append(
            f"| {ct['name']} | {ct['treatment']} − {ct['control']} | {ct['metric']} | {ct['estimate']:+.4f} | [{ct['ci_low']:+.4f}, {ct['ci_high']:+.4f}] (n={ct['n']}) |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(contrasts) - 3 :]))
    print(f"CPU hours: {summary['cpu_hours_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
