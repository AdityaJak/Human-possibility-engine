"""Exploratory Gate F run (GBE affect or Najar social data): fit on dev, select (config, lapse) on tune,
report on tune.

Never touches the test partition (the loaders refuse it without a freeze record). Same metric and
statistics as the Phase III runner (``core.evaluate``): per-participant mean NLL per free choice after
lapse mixing, lapse and configuration chosen by tune mean NLL, paired percentile bootstrap.

    python -m hpe.population.affect.experiment --dataset gbe|najar --out reports/phase3/affect/<run> [--resume]
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from hpe.core.rng import derive_seed
from hpe.population.affect import gbe, najar
from hpe.population.core import evaluate
from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import SequencePredictor, get_model
from hpe.population.core.runner import code_version

DEFAULT_MODELS = (
    "affect_gru_base",
    "affect_gru_obs",
    "affect_gru_shuf",
    "affect_gru_aux",
    "affect_gru_aux_shuf",
)
CONTRASTS = (
    ("affect_gru_obs", "affect_gru_shuf", "Gate F contrast: observed ratings vs matched-complexity control"),
    ("affect_gru_obs", "affect_gru_base", "observed ratings vs no affect input"),
    ("affect_gru_shuf", "affect_gru_base", "control vs no affect input (person-level rating information)"),
    ("affect_gru_aux", "affect_gru_aux_shuf", "latent-mood readout: observed vs permuted rating targets"),
    ("affect_gru_aux", "affect_gru_base", "latent-mood readout vs plain GRU"),
)
SOCIAL_MODELS = ("social_gru_base", "social_gru_obs", "social_gru_shuf")
SOCIAL_CONTRASTS = (
    (
        "social_gru_obs",
        "social_gru_shuf",
        "Gate F contrast: observed demonstrations vs matched-complexity control",
    ),
    ("social_gru_obs", "social_gru_base", "observed demonstrations vs no social input"),
    ("social_gru_shuf", "social_gru_base", "control vs no social input"),
)


def subset_mask(dataset: str, s: EncodedSeq) -> np.ndarray:
    """GBE: choices right after a new rating. Najar: choices on a pair that has been demonstrated."""
    if dataset == "gbe":
        return np.asarray(s.x[:, gbe.D_BASE + 1] > 0.5)
    return np.asarray(s.x[:, najar.D_BASE + 0] > 0.5)


def configs_of(grid: dict[str, list[Any]], fixed: dict[str, Any], cap: int = 24) -> list[dict[str, Any]]:
    keys = sorted(grid)
    combos = [dict(zip(keys, v, strict=True)) for v in itertools.product(*(grid[k] for k in keys))] or [{}]
    if len(combos) > cap:
        raise ValueError(f"{len(combos)} configurations > cap {cap}")
    return [{**fixed, **c} for c in combos]


def score_all(
    model: SequencePredictor, seqs: list[EncodedSeq], lapses: list[float]
) -> tuple[dict[float, list[evaluate.ParticipantScore]], dict[str, np.ndarray]]:
    by_lapse: dict[float, list[evaluate.ParticipantScore]] = {lam: [] for lam in lapses}
    logps: dict[str, np.ndarray] = {}
    for s in seqs:
        lp = model.predict_logprobs(s)
        logps[s.key] = lp
        for lam in lapses:
            by_lapse[lam].append(evaluate.score(s, lp, lam))
    return by_lapse, logps


def subset_nll(
    seqs: list[EncodedSeq], logps: dict[str, np.ndarray], lam: float, fresh: bool, dataset: str = "gbe"
) -> dict[str, float]:
    """Per-participant mean NLL on rows inside (``fresh``) or outside :func:`subset_mask`."""
    out: dict[str, float] = {}
    for s in seqs:
        lp = evaluate.apply_lapse(logps[s.key], s.avail, lam)
        m = s.scored & (subset_mask(dataset, s) == fresh)
        idx = np.flatnonzero(m)
        if idx.size:
            out[s.key] = float(-lp[idx, s.actions[idx]].mean())
    return out


def paired_dict(a: dict[str, float], b: dict[str, float], seed: int, key: tuple[str, ...]) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    est, lo, hi, p = evaluate.bootstrap(d, seed=seed, key=key)
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "p_one_sided": p, "n": len(keys)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    t_load = time.process_time()
    if args.dataset == "gbe":
        data, notes = gbe.load_gbe(["dev", "tune"], experiment_id=args.run_id, max_plays=args.max_plays)
        mod: Any = gbe
        contrasts_spec: tuple[tuple[str, str, str], ...] = CONTRASTS
    else:
        data, notes = najar.load_najar(["dev", "tune"], experiment_id=args.run_id)
        mod = najar
        contrasts_spec = SOCIAL_CONTRASTS
    fit = gbe.subsample(data["dev"], args.n_fit, args.seed, "fit")
    tune = gbe.subsample(data["tune"], args.n_tune, args.seed, "tune")
    load_s = time.process_time() - t_load
    lapses = list(evaluate.LAPSE_GRID)
    data_info = {
        "dataset": mod.DATASET,
        "source": mod.SOURCE,
        "partition_file": str(mod.PARTITION_FILE.relative_to(gbe.REPO_ROOT)),
        "dev_available": len(data["dev"]),
        "tune_available": len(data["tune"]),
        "n_fit": len(fit),
        "n_tune_scored": len(tune),
        "fit_choices": int(sum(s.n_scored for s in fit)),
        "tune_choices": int(sum(s.n_scored for s in tune)),
        "max_plays": args.max_plays,
        "exclusion_notes": len(notes),
        "load_cpu_s": round(load_s, 1),
    }
    print(json.dumps(data_info), flush=True)
    results: dict[str, Any] = {}
    best_scores: dict[str, list[evaluate.ParticipantScore]] = {}
    fresh_nll: dict[str, dict[str, dict[str, float]]] = {}
    grid = json.loads(args.grid)
    fixed = json.loads(args.fixed)
    for name in args.models:
        part = out / "parts" / f"{name}.json.gz"
        if args.resume and part.exists():
            rec = json.loads(gzip.decompress(part.read_bytes()))
            results[name] = rec["result"]
            best_scores[name] = [
                evaluate.ParticipantScore(
                    r["key"], r["nll"], r["n"], r["top1"], r["ece"], tuple(r["quarters"])
                )  # type: ignore[arg-type]
                for r in rec["scores"]
            ]
            fresh_nll[name] = rec["fresh"]
            print(f"[resume] {name}", flush=True)
            continue
        spec = get_model(name)
        cfgs = configs_of(grid or spec.default_grid, fixed)
        budget_left = float(args.budget_s)
        rows: list[dict[str, Any]] = []
        best: (
            tuple[float, int, float, SequencePredictor, dict[float, list[evaluate.ParticipantScore]]] | None
        ) = None
        best_logps: dict[str, np.ndarray] = {}
        t_model = time.process_time()
        for ci, cfg in enumerate(cfgs):
            if budget_left <= 0:
                rows.append({"config": cfg, "skipped": "budget exhausted"})
                continue
            model = spec.factory(cfg, mod.K, mod.D_BASE)
            seed = derive_seed(args.seed, "affect_fit", name, repr(sorted(cfg.items()))) % (2**31 - 1)
            t0 = time.process_time()
            log = model.fit(fit, seed=seed, threads=args.threads, budget_s=budget_left)
            fit_s = time.process_time() - t0
            budget_left -= fit_s
            t1 = time.process_time()
            by_lapse, logps = score_all(model, tune, lapses)
            score_s = time.process_time() - t1
            tune_nll = {lam: evaluate.mean_nll(v) for lam, v in by_lapse.items()}
            lam_best = min(lapses, key=lambda lam: (tune_nll[lam], lam))
            rows.append(
                {
                    "config": cfg,
                    "params": model.param_count(),
                    "fit_log": {k: v for k, v in log.items() if k != "val_history"},
                    "val_history": log.get("val_history"),
                    "fit_cpu_s": round(fit_s, 1),
                    "score_cpu_s": round(score_s, 1),
                    "tune_nll_by_lapse": {str(k): round(v, 6) for k, v in tune_nll.items()},
                    "best_lapse": lam_best,
                }
            )
            print(
                f"{name} cfg{ci} {cfg} params={model.param_count()} fit={fit_s:.0f}s epochs={log.get('epochs')} "
                f"stop={log.get('stopped')} tune_nll={tune_nll[lam_best]:.5f} (λ={lam_best})",
                flush=True,
            )
            if best is None or tune_nll[lam_best] < best[0]:
                best = (tune_nll[lam_best], ci, lam_best, model, by_lapse)
                best_logps = logps
        if best is None:
            raise RuntimeError(f"{name}: no configuration fitted")
        nll, ci, lam, model, by_lapse = best
        probe = sorted(tune, key=lambda s: (-s.T, s.key))[:2] + tune[:1]
        violations = [v for s in probe for v in check_causal(model, s, seed=args.seed)]
        fresh_nll[name] = {
            "fresh": subset_nll(tune, best_logps, lam, True, args.dataset),
            "other": subset_nll(tune, best_logps, lam, False, args.dataset),
        }
        scores = by_lapse[lam]
        result = {
            "selected_config": rows[ci]["config"],
            "selected_lapse": lam,
            "tune_mean_nll": nll,
            "tune_top1": float(np.mean([s.top1 for s in scores])),
            "tune_ece": float(np.mean([s.ece for s in scores])),
            "tune_nll_by_quarter": [float(np.nanmean([s.quarters[q] for s in scores])) for q in range(4)],
            "params": model.param_count(),
            "describe": model.describe(),
            "causal_violations": violations,
            "model_cpu_s": round(time.process_time() - t_model, 1),
            "configs": rows,
        }
        results[name] = result
        best_scores[name] = scores
        rec = {
            "result": result,
            "scores": [
                {
                    "key": s.key,
                    "nll": s.nll,
                    "n": s.n,
                    "top1": s.top1,
                    "ece": s.ece,
                    "quarters": list(s.quarters),
                }
                for s in scores
            ],
            "fresh": fresh_nll[name],
        }
        part.write_bytes(gzip.compress(json.dumps(rec).encode()))
        print(f"== {name}: tune NLL {nll:.5f} λ={lam} causal_violations={len(violations)}", flush=True)
    contrasts: list[dict[str, Any]] = []
    for a, b, label in contrasts_spec:
        if a not in best_scores or b not in best_scores:
            continue
        c = evaluate.paired(best_scores[a], best_scores[b], seed=args.seed, key=("affect", a, b))
        c_f = paired_dict(fresh_nll[a]["fresh"], fresh_nll[b]["fresh"], args.seed, ("affect_fresh", a, b))
        c_o = paired_dict(fresh_nll[a]["other"], fresh_nll[b]["other"], args.seed, ("affect_other", a, b))
        contrasts.append(
            {
                "a": a,
                "b": b,
                "label": label,
                "all_choices": c,
                "choices_in_subset": c_f,
                "other_choices": c_o,
                "meets_gate_f_margin_if_confirmatory": bool(
                    float(c["estimate"]) <= -0.02 and float(c["ci_high"]) < 0
                ),
            }
        )
    summary = {
        "run_id": args.run_id,
        "status": "EXPLORATORY (dev fit, tune selection and report; test never accessed)",
        "code_version": code_version(),
        "python": platform.python_version(),
        "seed": args.seed,
        "threads": args.threads,
        "budget_s_per_model": args.budget_s,
        "lapse_grid": lapses,
        "data": data_info,
        "subset": "choices right after a new rating"
        if args.dataset == "gbe"
        else "choices on a demonstrated pair",
        "models": results,
        "contrasts": contrasts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", choices=("gbe", "najar"), default="gbe")
    ap.add_argument("--run-id", default="affect_gbe_explore")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-fit", type=int, default=0, help="dev participants used for fitting (0 = all)")
    ap.add_argument("--n-tune", type=int, default=0, help="tune participants scored (0 = all)")
    ap.add_argument("--max-plays", type=int, default=gbe.MAX_PLAYS)
    ap.add_argument("--grid", default="{}", help="JSON grid overriding each model's default grid")
    ap.add_argument("--fixed", default="{}", help="JSON fixed hyper-parameters added to every configuration")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=2700.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if args.models is None:
        args.models = list(DEFAULT_MODELS if args.dataset == "gbe" else SOCIAL_MODELS)
    run(args)


if __name__ == "__main__":
    main()
