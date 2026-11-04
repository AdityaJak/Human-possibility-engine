#!/usr/bin/env python3
"""Measured probe ceilings for the `composition` family (spec §1: "linear probe ceiling measured and
recorded"). DEV seeds only; no learner is involved.

    python reports/phase2/evaluation_v2/scripts/composition_ceiling.py

For every dev seed and both world variants (shared primitives / specialist control) each probe is
fitted on the TRAIN symbol pool (64 symbols) and scored on the fixed, label-balanced holdout eval
sets that `hpe.tasks.assess` uses (60 items per family; pair items are scored by the higher probe
score of the two slots), so a probe's number is exactly what a frozen policy with that readout would
measure in the learning lab (chance 0.5):

  lstsq_raw        least-squares linear readout on the raw +-1 features (+ bias)
  logistic_raw     logistic regression on the raw features (full-batch gradient descent, L2 1e-3)
  logistic_quad    logistic regression on raw features + all pairwise products (quadratic probe)
  oracle_prims     logistic regression on [r1(x), r2(x), r1*r2] -- what a learner that has abstracted
                   the two taught primitives sees; this is the reference for "composes abstracted
                   primitives" (it is only meaningful in the shared variant; in the control variant
                   the taught primitives are not the composed ones, which is the point of that control)

Output: reports/phase2/evaluation_v2/composition_ceiling.json and composition_ceiling.log.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lab_runner import DEV_SEEDS, Log, check_seeds, code_version, summary  # noqa: E402

from hpe.tasks.v2.composition import COMPOSITION_FAMILIES, CompositionConfig, CompositionWorld  # noqa: E402

OUT = HERE.parent / "composition_ceiling.json"
WORLDS = {"shared": CompositionConfig(), "control": CompositionConfig(shared_structure=False)}


def _bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def _quad(x: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(x.shape[1], k=1)
    return np.concatenate([x, (x[:, :, None] * x[:, None, :])[:, iu[0], iu[1]]], axis=1)


def _prims(w: CompositionWorld, x: np.ndarray) -> np.ndarray:
    a, b = w.taught.r1(x).astype(np.float64) * 2 - 1, w.taught.r2(x).astype(np.float64) * 2 - 1
    return np.stack([a, b, a * b], axis=1)


def logistic_fit(
    x: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.1, l2: float = 1e-3
) -> np.ndarray:
    xb = _bias(x)
    w = np.zeros(xb.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
        g = xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def lstsq_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(_bias(x), y * 2.0 - 1.0, rcond=None)
    return w


PROBES: dict[str, tuple[str, Any]] = {
    "lstsq_raw": ("raw", lstsq_fit),
    "logistic_raw": ("raw", logistic_fit),
    "logistic_quad": ("quad", logistic_fit),
    "oracle_prims": ("prims", logistic_fit),
}


def features(kind: str, w: CompositionWorld, x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return {"raw": lambda: x, "quad": lambda: _quad(x), "prims": lambda: _prims(w, x)}[kind]()


def eval_accuracy(w: CompositionWorld, family: str, kind: str, weights: np.ndarray) -> float:
    items = w.eval_set(family)
    correct = 0
    for it in items:
        ctx = it.context if it.context.ndim == 2 else it.context[None, :]
        scores = _bias(features(kind, w, ctx)) @ weights
        pred = int(scores[0] > 0) if len(scores) == 1 else int(np.argmax(scores))
        correct += int(pred == it.correct_action)
    return correct / len(items)


def main() -> None:
    cv = code_version()
    log = Log(HERE.parent / "composition_ceiling.log")
    check_seeds(DEV_SEEDS)
    t0 = time.process_time()
    base_family = {
        "primitive_a": "primitive_a",
        "primitive_b": "primitive_b",
        "composition": "composition",
        "composition_pair": "composition",
    }
    per_seed: dict[str, dict[str, dict[str, list[float]]]] = {
        v: {pr: {f: [] for f in COMPOSITION_FAMILIES} for pr in PROBES} for v in WORLDS
    }
    meta: dict[str, list[dict[str, Any]]] = {v: [] for v in WORLDS}
    for variant, cfg in WORLDS.items():
        for seed in DEV_SEEDS:
            w = CompositionWorld(seed, cfg)
            xtr = w.symbols[w.train_ids]
            meta[variant].append(
                {
                    "seed": seed,
                    "op": w.op,
                    "taught": w.taught.describe(),
                    "composed": w.composed.describe(),
                    "class_shares_holdout": w.class_shares(),
                }
            )
            for pr, (kind, fit) in PROBES.items():
                for f in COMPOSITION_FAMILIES:
                    y = w.labels[base_family[f]][w.train_ids].astype(np.float64)
                    weights = fit(features(kind, w, xtr), y)
                    per_seed[variant][pr][f].append(eval_accuracy(w, f, kind, weights))
            log(
                f"{variant} seed {seed} op={w.op}: "
                + ", ".join(
                    f"{pr}/{f}={per_seed[variant][pr][f][-1]:.3f}" for pr in PROBES for f in ("composition",)
                )
            )
    out: dict[str, Any] = {
        "label": "EXPLORATORY dev-seed measurement; not a result claim",
        "what": __doc__.strip().splitlines()[0],
        "seeds": DEV_SEEDS,
        "config": {
            v: {
                k: getattr(cfg, k)
                for k in (
                    "n_features",
                    "n_train_symbols",
                    "n_holdout_symbols",
                    "shared_structure",
                    "n_literals",
                    "parity_size",
                    "eval_items_per_family",
                    "min_per_label",
                )
            }
            for v, cfg in WORLDS.items()
        },
        "probes": {pr: {"features": kind, "fit": fit.__name__} for pr, (kind, fit) in PROBES.items()},
        "chance": 0.5,
        "summary": {
            v: {pr: {f: summary(per_seed[v][pr][f], seed=7) for f in COMPOSITION_FAMILIES} for pr in PROBES}
            for v in WORLDS
        },
        "per_seed": per_seed,
        "worlds": meta,
        # the operator matters: with a conjunction r1 (25% positive) AND / IMPLIES leave a linear shortcut
        # on balanced eval sets (predict r1 / not-r1), XOR leaves none; so the ceiling is reported per op too
        "by_op": {
            v: {
                pr: {
                    f: {
                        op: {
                            "mean": float(
                                np.mean(
                                    [
                                        x
                                        for x, m in zip(per_seed[v][pr][f], meta[v], strict=True)
                                        if m["op"] == op
                                    ]
                                )
                            ),
                            "n_seeds": sum(m["op"] == op for m in meta[v]),
                        }
                        for op in sorted({m["op"] for m in meta[v]})
                    }
                    for f in COMPOSITION_FAMILIES
                }
                for pr in PROBES
            }
            for v in WORLDS
        },
        "ceiling_linear_raw_composition": {
            v: {
                "mean_of_best_linear_probe": max(
                    np.mean(per_seed[v]["lstsq_raw"]["composition"]),
                    np.mean(per_seed[v]["logistic_raw"]["composition"]),
                ),
                "max_over_seeds_best_linear_probe": max(
                    max(per_seed[v]["lstsq_raw"]["composition"]),
                    max(per_seed[v]["logistic_raw"]["composition"]),
                ),
            }
            for v in WORLDS
        },
        "code_version": cv,
        "cpu_s": round(time.process_time() - t0, 2),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for v in WORLDS:
        log(f"== {v}")
        for pr in PROBES:
            log(
                "  "
                + pr
                + ": "
                + ", ".join(
                    f"{f}={out['summary'][v][pr][f]['mean']:.3f} [{out['summary'][v][pr][f]['ci_low']:.3f},"
                    f" {out['summary'][v][pr][f]['ci_high']:.3f}]"
                    for f in COMPOSITION_FAMILIES
                )
            )
        log(
            "  composition by op (logistic_raw): "
            + ", ".join(
                f"{op}={d['mean']:.3f} (n={d['n_seeds']})"
                for op, d in out["by_op"][v]["logistic_raw"]["composition"].items()
            )
        )
    log(f"wrote {OUT} (cpu {out['cpu_s']} s)")


if __name__ == "__main__":
    main()
