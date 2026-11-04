"""Parser validation for Phase III family adapters — dev/tune only, through the locked loader.

    python -m hpe.population.families.validate two_step --partition dev --out reports/phase3/baselines/validation

Writes ``<out>/<family>_<partition>.json`` with parse rate, exclusions, trials per participant, feedback modes,
choice and outcome descriptives and (two_step) the stay-probability signature by previous outcome × transition.
The test partition is refused here even if a freeze record exists: validation never needs it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import GenericTrial, validate_generic
from hpe.population.core.data import load_participants
from hpe.population.core.families import get_family


def _q(xs: list[float] | list[int]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "min": float(a.min()),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def _stay_signature(parts: list[Participant]) -> dict[str, Any]:
    """Two-step stay probabilities (stage-1 repeat of the previous day's spaceship).

    Two-stage designs: by previous reward (1/0) × previous transition (common/rare, with "common" = the
    participant's most frequent planet for that spaceship). One-stage designs: by previous outcome sign and by
    whether the start pair repeated. Participant-level means, then averaged.
    """
    cells: dict[str, list[float]] = defaultdict(list)
    for p in parts:
        design = p.meta.get("design")
        common = {
            ship: max(sorted(cnt), key=lambda pl: cnt[pl])
            for ship, cnt in sorted(p.meta["common_planet"].items())
        }
        pairs = p.meta["labels"]["pairs"]
        planets = p.meta["labels"]["planets"]
        per: dict[str, list[float]] = defaultdict(list)
        prev_ship: str | None = None
        prev_pair: int | None = None
        prev_rew: float | None = None
        prev_rare: bool | None = None
        prev_day = -10
        trials = [t for t in p.trials if isinstance(t, GenericTrial)]
        for i, t in enumerate(trials):
            if t.phase != "stage1":
                continue
            day = int(t.meta["day"])
            pair_idx = 0 if t.features[2] <= 0 else 1
            ship = pairs[pair_idx][t.choice] if len(pairs) > 1 else pairs[0][t.choice]
            planet = planets[int(t.meta["planet"])]
            rew: float | None
            if design == "one_stage":
                rew = float(t.meta["points"])
            else:
                nxt = trials[i + 1] if i + 1 < len(trials) else None
                rew = (
                    float(nxt.meta["points"])
                    if nxt is not None and nxt.phase == "stage2" and int(nxt.meta["day"]) == day
                    else None
                )
            rare = common.get(ship) != planet
            if prev_ship is not None and day == prev_day + 1 and prev_rew is not None:
                stay = 1.0 if ship == prev_ship else 0.0
                if design == "one_stage":
                    if prev_pair == pair_idx:
                        sign = "pos" if prev_rew > 0 else ("zero" if prev_rew == 0 else "neg")
                        per[f"same_pair|prev_{sign}"].append(stay)
                else:
                    per[f"prev_r{int(prev_rew > 0)}|{'rare' if prev_rare else 'common'}"].append(stay)
            prev_ship, prev_pair, prev_rew, prev_rare, prev_day = ship, pair_idx, rew, rare, day
        for k, v in per.items():
            if v:
                cells[f"{design}:{k}"].append(float(np.mean(v)))
    return {k: {"participants": len(v), "mean_stay": float(np.mean(v))} for k, v in sorted(cells.items())}


def validate(family: str, partition: str) -> dict[str, Any]:
    if partition not in ("dev", "tune"):
        raise ValueError("parser validation uses dev or tune only")
    spec = get_family(family)
    parts, excluded = load_participants(family, partition, experiment_id=f"validation_{family}")
    out: dict[str, Any] = {
        "family": family,
        "role": spec.role,
        "partition": partition,
        "participants_parsed": len(parts),
        "excluded": len(excluded),
        "exclusion_notes": excluded,
        "per_experiment": validate_generic(parts) if parts else {},
    }
    by_exp: dict[str, list[Participant]] = defaultdict(list)
    for p in parts:
        by_exp[p.experiment].append(p)
    details: dict[str, Any] = {}
    for exp, ps in sorted(by_exp.items()):
        steps = [len(p.trials) for p in ps]
        free = [sum(1 for t in p.trials if not t.forced) for p in ps]
        choices = [t.choice for p in ps for t in p.trials if not t.forced]
        rewards = [t.reward for p in ps for t in p.trials if t.mode != "none" and t.reward is not None]
        d: dict[str, Any] = {
            "participants": len(ps),
            "steps_per_participant": _q(steps),
            "free_steps_per_participant": _q(free),
            "p_action0": float(np.mean(np.asarray(choices) == 0)) if choices else None,
            "normalised_reward": _q([float(r) for r in rewards]),
        }
        if family == "two_step":
            d["design"] = sorted({p.meta["design"] for p in ps})
            d["days_seen"] = _q([p.meta["days_seen"] for p in ps])
            d["missed_stage1_per_participant"] = _q([p.meta["missed_stage1"] for p in ps])
            d["missed_stage2_per_participant"] = _q([p.meta["missed_stage2"] for p in ps])
            fr = []
            for p in ps:
                for _ship, cnt in sorted(p.meta["common_planet"].items()):
                    tot = sum(cnt.values())
                    if tot >= 5:
                        fr.append(max(cnt.values()) / tot)
            d["common_transition_fraction"] = _q(fr)
            d["stay_signature"] = _stay_signature(ps)
        details[exp] = d
    out["details"] = details
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.population.families.validate")
    ap.add_argument("family")
    ap.add_argument("--partition", default="dev", choices=["dev", "tune"])
    ap.add_argument("--out", type=Path, default=Path("reports/phase3/baselines/validation"))
    args = ap.parse_args(argv)
    res = validate(args.family, args.partition)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.family}_{args.partition}.json"
    path.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(
        f"{args.family}/{args.partition}: parsed {res['participants_parsed']}, excluded {res['excluded']} -> {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
