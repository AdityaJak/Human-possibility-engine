"""Toy verification of the design-team candidates (synthetic data only; no human data).

Question per candidate: does individual adaptation beat the candidate's own frozen variant when
individuals genuinely differ (``hetero``), and does it *not* when they do not (``homo``)? Also reports
the core ``pop_gru`` at the same hidden size and the generating-model oracle (bandit toy) for scale.

    PYTHONPATH=packages python experiments/phase3/candidates/toy_verification.py \
        --out reports/phase3/candidates/toy [--models f1_latent,f3_arbiter] [--kinds cue,bandit]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hpe.population.candidates import common
from hpe.population.candidates.toy import make_population
from hpe.population.core import evaluate
from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import input_dim
from hpe.population.core.interface import get_model

MODELS = [
    "pop_gru",
    "f1_latent",
    "f1_trait",
    "f1_cavia",
    "f1_bolt",
    "f3_arbiter",
    "f3_fixedshare",
    "f2_world_policy",
    "f2_delta",
]


def run_one(name: str, kind: str, hetero: bool, n_train: int, T: int, seed: int, cfg: dict) -> dict:
    train = make_population(n_train, T, hetero=hetero, seed=seed, kind=kind)
    test = make_population(80, T, hetero=hetero, seed=seed, offset=10_000, kind=kind)
    model = get_model(name).factory(dict(cfg), 4, input_dim(4))
    t0 = time.process_time()
    log = model.fit(train, seed=seed, threads=2, budget_s=1800)
    fit_cpu = time.process_time() - t0
    res: dict[str, np.ndarray] = {}
    for mode in ("adapt", "frozen", "donor"):
        sc = []
        for i, s in enumerate(test):
            if mode == "donor":
                lp = model.predict_logprobs(s, donor=test[(i + 1) % len(test)])
            else:
                lp = model.predict_logprobs(s, mode=mode)
            sc.append(evaluate.score(s, lp, 0.0).nll)
        res[mode] = np.array(sc)
        if not getattr(model, "adaptive", False):
            break
    out = {
        "model": name,
        "kind": kind,
        "hetero": hetero,
        "n_train": n_train,
        "T": T,
        "adapt_nll": float(res["adapt"].mean()),
        "params": model.param_count(),
        "fit_cpu_s": round(fit_cpu, 1),
        "causal": check_causal(model, test[0]) or "ok",
    }
    if "frozen" in res:
        for v in ("frozen", "donor"):
            d = res["adapt"] - res[v]
            est, lo, hi, p = evaluate.bootstrap(d, seed=seed, key=("toy", name, kind, str(hetero), v))
            out[f"{v}_nll"] = float(res[v].mean())
            out[f"adapt_minus_{v}"] = {"estimate": est, "ci": [lo, hi], "p_one_sided": p}
    meta = log.get("meta_train")
    if isinstance(meta, dict):
        out["meta_val_history"] = meta.get("val_history")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--kinds", default="cue,bandit")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "toy_verification.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["kind"], r["hetero"]))
    cfg = {"hidden": 16, "lr": 0.01, "m": 10}
    for kind in args.kinds.split(","):
        for hetero in (True, False):
            common.clear_backbone_cache()
            for name in args.models.split(","):
                if (name, kind, hetero) in done:
                    continue
                r = run_one(name, kind, hetero, args.n_train, args.T, args.seed, cfg)
                with path.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                fz = r.get("adapt_minus_frozen", {})
                print(
                    f"{kind:6s} hetero={hetero!s:5s} {name:16s} adapt={r['adapt_nll']:.4f} "
                    + (
                        f"frozen={r['frozen_nll']:.4f} Δ={fz['estimate']:+.4f} [{fz['ci'][0]:+.4f}, {fz['ci'][1]:+.4f}] "
                        if fz
                        else ""
                    )
                    + f"params={r['params']} cpu={r['fit_cpu_s']}s causal={r['causal']}",
                    flush=True,
                )
    render(path, args.out / "toy_verification.md", args)


def render(path: Path, out: Path, args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    lines = [
        "# Toy verification of the design-team candidates (synthetic data only)",
        "",
        f"Train {args.n_train} simulated people, test 80 held-out people, T = {args.T}, seed {args.seed}; "
        "hidden 16, lr 0.01, m = 10. Δ = adapt − frozen (negative = adaptation helps), paired bootstrap 95 % CI. "
        "Pass rule: Δ < −0.01 with the CI below 0 when individuals differ (hetero); Δ > −0.005 when they "
        "do not (homo). `pop_gru` (same hidden size) is shown for scale.",
        "",
        "| toy | individuals differ | model | adapt NLL | frozen NLL | Δ adapt − frozen [95% CI] | Δ adapt − donor | params | fit CPU s | causal |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fz, dn = r.get("adapt_minus_frozen"), r.get("adapt_minus_donor")
        lines.append(
            f"| {r['kind']} | {r['hetero']} | {r['model']} | {r['adapt_nll']:.4f} | "
            + (
                f"{r['frozen_nll']:.4f} | {fz['estimate']:+.4f} [{fz['ci'][0]:+.4f}, {fz['ci'][1]:+.4f}] | "
                if fz
                else "— | — | "
            )
            + (f"{dn['estimate']:+.4f} | " if dn else "— | ")
            + f"{r['params']} | {r['fit_cpu_s']} | {r['causal']} |"
        )
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
