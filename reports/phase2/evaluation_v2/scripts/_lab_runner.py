"""Shared driver for the science/evaluation-v2 DEV scripts: run `hpe.evaluation.learning_lab` jobs in
worker processes with per-job CPU accounting, and summarise per-seed values with bootstrap CIs.

Dev seeds (1000-1099) and tune seeds (2000-2049) only; the confirmatory seeds 500-509 are refused.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.evaluation import stats
from hpe.evaluation.learning_lab import LabContext, resolve_params, run_learning_lab

DEV_SEEDS = list(range(1000, 1010))
N_BOOT = 4000


def check_seeds(seeds: list[int]) -> None:
    bad = [s for s in seeds if not (1000 <= s <= 1099 or 2000 <= s <= 2049)]
    if bad:
        raise SystemExit(f"refusing non-dev/tune seeds {bad} (confirmatory seeds 500-509 are never run here)")


def code_version() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return sha.strip() + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def hardware() -> str:
    return f"{platform.machine()} {os.cpu_count()} cpu, {platform.system()} {platform.release()}, python {platform.python_version()}"


class Log:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fh = path.open("w")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """One (variant, arm, seed) learning-lab run. Returns per-family means over agents of the
    initial (tick 0) and final held-out accuracy, the mean learning curve, compute counters and the
    CPU seconds this job consumed in its worker process."""
    t0, c0 = time.perf_counter(), time.process_time()
    p = resolve_params(job["params"])
    factory = registry.get_factory(job["policy_class"], job["policy_params"])
    run_id = f"{job['variant']}__{job['arm']}__{job['seed']}"
    ctx = LabContext(job["experiment"], job["arm"], int(job["seed"]), run_id, p)
    out = run_learning_lab(factory, ctx)
    fams = list(p["families"])
    final: dict[str, float] = {}
    initial: dict[str, float] = {}
    curve: dict[str, dict[str, float]] = {}
    for f in fams:
        fin = [o.value for o in out if o.metric == f"final_heldout_accuracy/{f}"]
        cur = [o for o in out if o.metric == f"curve/heldout_accuracy/{f}"]
        final[f] = float(np.mean(fin))
        initial[f] = float(np.mean([o.value for o in cur if o.tick == 0]))
        ticks = sorted({o.tick for o in cur})
        curve[f] = {str(t): float(np.mean([o.value for o in cur if o.tick == t])) for t in ticks}
    dec = float(sum(o.value for o in out if o.metric == "compute/decisions"))
    upd = float(sum(o.value for o in out if o.metric == "compute/param_updates"))
    leak = [o.value for o in out if o.metric == "leakage/holdout_symbols_in_practice"]
    return {
        "variant": job["variant"],
        "arm": job["arm"],
        "seed": int(job["seed"]),
        "n_agents": int(p["n_agents"]),
        "final": final,
        "initial": initial,
        "final_mean": float(np.mean([final[f] for f in fams])),
        "initial_mean": float(np.mean([initial[f] for f in fams])),
        "curve": curve,
        "decisions": dec,
        "param_updates": upd,
        "leakage_holdout_symbols_in_practice": float(sum(leak)) if leak else None,
        "cpu_s": time.process_time() - c0,
        "wall_s": time.perf_counter() - t0,
    }


def run_jobs(jobs: list[dict[str, Any]], workers: int, log: Log) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    log(f"{len(jobs)} jobs on {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(run_job, jobs, chunksize=1), 1):
            results.append(r)
            log(
                f"{i}/{len(jobs)} {r['variant']}/{r['arm']} seed {r['seed']}: "
                f"final {r['final_mean']:.3f} (init {r['initial_mean']:.3f}) cpu {r['cpu_s']:.1f}s"
            )
    return results


def summary(values: list[float], seed: int) -> dict[str, float]:
    b = stats.bootstrap_ci(values, n_boot=N_BOOT, seed=seed)
    return {
        "mean": b.estimate,
        "ci_low": b.ci_low,
        "ci_high": b.ci_high,
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "n": len(values),
    }


def paired(treatment: list[float], control: list[float], seed: int) -> dict[str, float]:
    b = stats.paired_bootstrap(treatment, control, n_boot=N_BOOT, seed=seed)
    return {
        "estimate": b.estimate,
        "ci_low": b.ci_low,
        "ci_high": b.ci_high,
        "cohen_dz": stats.cohen_dz(treatment, control),
        "p_signflip": stats.permutation_test_paired(treatment, control, seed=seed),
        "n": b.n,
    }


def fmt(d: dict[str, float], key: str = "mean") -> str:
    return f"{d[key]:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def by(results: list[dict[str, Any]], variant: str, arm: str) -> list[dict[str, Any]]:
    return sorted(
        (r for r in results if r["variant"] == variant and r["arm"] == arm), key=lambda r: r["seed"]
    )
