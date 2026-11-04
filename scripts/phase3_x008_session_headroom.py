#!/usr/bin/env python3
"""X3-008 (exploratory, director): how much does a person's *earlier-session* history improve prediction when the
in-context state would otherwise be lost between sessions? This bounds what a persistent person-level state could
add in deployments where each session starts fresh (the Future Self setting), unlike the single-session
benchmarks where an in-context recurrent model already sees the whole history.

Data: the external GBE risky-choice family (``families/gbe_risk.py``; CC0; seeded 2,000 dev / 1,000 tune
subsample; first 4 plays of 30 choices per person). Two population GRUs of the same size and trainer:
  * ``full``   trained and evaluated on each person's concatenated plays (the hidden state carries across plays);
  * ``session`` trained and evaluated on each play as a separate sequence (state reset at every play start).
Score: per participant, mean NLL over the free choices of plays 2-4 (play 1 has no earlier play), lapse chosen
on tune per model from the standard grid; paired bootstrap ``session − full`` (positive = value of earlier
sessions). Also ``full`` weights evaluated with the state reset (same weights, history removed).

    HPE_GBE_MAT=... PYTHONPATH=packages python scripts/phase3_x008_session_headroom.py --out reports/phase3/director/x3_008_session_headroom
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hpe.population.core import evaluate
from hpe.population.core.data import load_family
from hpe.population.core.encoding import EncodedSeq, Layout, truncate
from hpe.population.core.interface import get_model

EXP = "x3_008_session_headroom"
SEED = 20261111


def split_plays(seq: EncodedSeq) -> list[EncodedSeq]:
    """Cut a concatenated GBE sequence at block starts (each play starts a block); keys get a play suffix."""
    L = Layout(seq.k_max)
    starts = [t for t in range(seq.T) if seq.x[t, L.block_start] > 0.5] or [0]
    if starts[0] != 0:
        starts = [0, *starts]
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else seq.T
        part = EncodedSeq(
            f"{seq.key}/play{i}",
            seq.family,
            seq.k_max,
            seq.n_actions,
            seq.x[s:e].copy(),
            seq.actions[s:e].copy(),
            seq.avail[s:e].copy(),
            seq.scored[s:e].copy(),
        )
        # the first row of a fresh session has no previous trial: clear the previous-trial slots
        part.x[0, L.prev_action] = 0
        part.x[0, L.prev_outcomes] = 0
        part.x[0, L.prev_observed] = 0
        part.x[0, L.prev_reward] = 0
        part.x[0, L.prev_feedback] = 0
        part.x[0, L.prev_feat] = 0
        out.append(part)
    return out


def later_play_logp(model, seq: EncodedSeq, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Log-probs and a mask selecting rows of plays 2+ (in full-sequence row order)."""
    L = Layout(seq.k_max)
    starts = [t for t in range(seq.T) if seq.x[t, L.block_start] > 0.5]
    first_end = starts[1] if len(starts) > 1 else seq.T
    mask = np.zeros(seq.T, dtype=bool)
    mask[first_end:] = True
    if mode == "full":
        lp = model.predict_logprobs(seq)
    else:
        lp = np.concatenate([model.predict_logprobs(p) for p in split_plays(seq)], axis=0)
    return lp, mask


def scores(model, seqs: list[EncodedSeq], mode: str, lam: float) -> dict[str, float]:
    out = {}
    for s in seqs:
        lp, mask = later_play_logp(model, s, mode)
        sub = truncate(s, s.T)
        sub.scored = s.scored & mask
        if sub.scored.sum() == 0:
            continue
        out[s.key] = evaluate.score(sub, lp, lam).nll
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev, _ = load_family("gbe_risk", "dev", experiment_id=EXP, k_max=2)
    tune, _ = load_family("gbe_risk", "tune", experiment_id=EXP, k_max=2)
    spec = get_model("pop_gru")
    cfg = {"hidden": args.hidden, "lr": 0.01, "dropout": 0.0}
    res: dict = {"config": cfg, "n_dev": len(dev), "n_tune": len(tune)}
    models = {}
    for name, train in (("full", dev), ("session", [p for s in dev for p in split_plays(s)])):
        m = spec.factory(cfg, 2, dev[0].x.shape[1])
        t0 = time.process_time()
        log = m.fit(train, seed=SEED, threads=2, budget_s=2700)
        res[f"{name}_fit"] = {
            "cpu_s": round(time.process_time() - t0, 1),
            "epochs": log["epochs"],
            "stopped": log["stopped"],
        }
        models[name] = m
    table = {}
    for label, (mname, mode) in {
        "full_history": ("full", "full"),
        "session_only": ("session", "session"),
        "full_weights_state_reset": ("full", "session"),
    }.items():
        by_lapse = {lam: scores(models[mname], tune, mode, lam) for lam in evaluate.LAPSE_GRID}
        lam = min(by_lapse, key=lambda v: (np.mean(list(by_lapse[v].values())), v))
        table[label] = {
            "lapse": lam,
            "scores": by_lapse[lam],
            "mean": float(np.mean(list(by_lapse[lam].values()))),
        }
    res["means"] = {k: v["mean"] for k, v in table.items()}
    contrasts = {}
    for a, b in (("session_only", "full_history"), ("full_weights_state_reset", "full_history")):
        keys = sorted(set(table[a]["scores"]) & set(table[b]["scores"]))
        d = np.array([table[a]["scores"][k] - table[b]["scores"][k] for k in keys])
        est, lo, hi, _p = evaluate.bootstrap(d, seed=SEED, key=(a, b))
        contrasts[f"{a} − {b}"] = {"estimate": est, "ci": [lo, hi], "n": len(keys)}
    res["contrasts"] = contrasts
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
