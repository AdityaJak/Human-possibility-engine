"""Audit 1b: re-run selected (arm, seed) jobs of a committed learning-lab experiment from scratch
at the current code version and compare the raw outcome logs with the committed ones.

Usage: python reports/audit/scripts/rerun_from_scratch.py <tmp_dir> [experiment_id] [seed] [arm ...]
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

from hpe.evaluation.runner import execute_job, expand_jobs
from hpe.evaluation.spec import load_spec

ROOT = Path(__file__).resolve().parents[3]


def _canon(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in gzip.open(path)]
    return rows


def main() -> int:
    tmp = Path(sys.argv[1])
    exp = sys.argv[2] if len(sys.argv) > 2 else "exp_baselines_heldout"
    seed_arg = sys.argv[3] if len(sys.argv) > 3 else "3"
    seeds = None if seed_arg == "all" else {int(seed_arg)}
    arms = sys.argv[4:] or ["developmental", "plain_rl"]
    all_arms = arms == ["all"]
    spec = load_spec(ROOT / "experiments" / f"{exp}.yaml")
    exp_dir = tmp / exp
    (exp_dir / "runs").mkdir(parents=True, exist_ok=True)
    jobs = [
        j
        for j in expand_jobs(spec.frozen(), exp_dir)
        if j.variant == "main" and (seeds is None or j.seed in seeds) and (all_arms or j.arm in arms)
    ]
    report = {}
    for job in jobs:
        (exp_dir / "runs" / job.run_id).mkdir(parents=True, exist_ok=True)
        info = execute_job(job)
        new = exp_dir / "runs" / job.run_id / "outcomes.jsonl.gz"
        old = ROOT / "reports" / "experiments" / exp / "runs" / job.run_id / "outcomes.jsonl.gz"
        a, b = _canon(old), _canon(new)
        same_rows = a == b
        sha_old = hashlib.sha256(gzip.open(old).read()).hexdigest()[:16]
        sha_new = hashlib.sha256(gzip.open(new).read()).hexdigest()[:16]
        report[job.run_id] = {
            "n_rows_committed": len(a),
            "n_rows_rerun": len(b),
            "rows_identical": same_rows,
            "uncompressed_sha256_committed": sha_old,
            "uncompressed_sha256_rerun": sha_new,
            "wall_time_s": round(info["wall_time_s"], 2),
        }
        print(job.run_id, report[job.run_id], flush=True)
    (tmp / f"rerun_{exp}_s{seed_arg}.json").write_text(json.dumps(report, indent=2))
    n_same = sum(v["rows_identical"] for v in report.values())
    print(f"SUMMARY {exp}: {n_same}/{len(report)} main-variant jobs bit-identical to committed raw logs")
    return 0 if all(v["rows_identical"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
