#!/usr/bin/env python3
"""Critic (audit/critic-v3): independent refit of one pop_gru configuration with new seeds (seed variance).

    HPE_PSYCH101_P3=<subset> PYTHONPATH=packages python reports/phase3/critic/seed_refit.py \
        --family garcia --hidden 32 --dropout 0.0 --seeds 3 --out reports/phase3/critic/seed_refit_garcia.json

Dev and tune partitions only (``load_family``; the test partition is never requested). One thread. The
configuration and trainer settings are those of the committed X3-B03 / X3-C03 runs; only the seed differs.
Tune NLL is computed exactly as the runner does (fresh copy and opaque key per participant, lapse selected
on tune from the standard grid), so the numbers are comparable with the committed ``select_nll`` values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hpe.core.rng import derive_seed
from hpe.population.core import evaluate, runner
from hpe.population.core.data import load_family
from hpe.population.core.encoding import input_dim
from hpe.population.core.interface import get_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="garcia")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev, _ = load_family(args.family, "dev", experiment_id="critic_seed_refit")
    tune, _ = load_family(args.family, "tune", experiment_id="critic_seed_refit")
    cfg = {
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "max_epochs": 150,
        "patience": 10,
        "batch_size": 8,
    }
    rows = []
    for i in range(args.seeds):
        seed = derive_seed(20261104, "critic_seed_refit", args.family, i) % (2**31 - 1)
        model = get_model("pop_gru").factory(dict(cfg), 4, input_dim(4))
        t0 = time.process_time()
        log = model.fit(dev, seed=seed, threads=1, budget_s=2700.0)
        cpu = time.process_time() - t0
        lps = [runner.predict_isolated(model, s, f"critic-{i}-{j}") for j, s in enumerate(tune)]
        by = {
            lam: evaluate.mean_nll([evaluate.score(s, lp, lam) for s, lp in zip(tune, lps, strict=True)])
            for lam in evaluate.LAPSE_GRID
        }
        lam = min(by, key=lambda v: (by[v], v))
        rows.append(
            {
                "seed": seed,
                "select_nll": by[lam],
                "lapse": lam,
                "nll_lapse0": by[0.0],
                "fit_cpu_s": round(cpu, 1),
                "epochs": log["epochs"],
                "stopped": log["stopped"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    v = np.array([r["select_nll"] for r in rows])
    res = {
        "family": args.family,
        "config": cfg,
        "n_dev": len(dev),
        "n_tune": len(tune),
        "fits": rows,
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
        "range": float(v.max() - v.min()),
        "threads": 1,
    }
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
