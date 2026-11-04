#!/usr/bin/env python3
"""Descriptive lifespan table for a stored society run, computed from its raw outcome log.

    python scripts/describe_run.py reports/runs/<run_id> [--markdown]

Prints mean held-out accuracy per task family by simulated age band. Descriptive only (no
inference); used in RESEARCH_RESULTS.md and regenerable by anyone from the committed log.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

BANDS = [(0, 3), (3, 6), (6, 10), (10, 14), (14, 18), (18, 30), (30, 50), (50, 65), (65, 80), (80, 120)]
FAMILIES = ("sort", "compare", "signal", "barter")


def describe(run_dir: Path) -> list[dict[str, float | str | int]]:
    sums: dict[tuple[int, str], list[float]] = defaultdict(list)
    with gzip.open(run_dir / "outcomes.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if not r["metric"].startswith("holdout_accuracy/"):
                continue
            fam = r["metric"].split("/", 1)[1]
            age = r["age_years"]
            for i, (lo, hi) in enumerate(BANDS):
                if lo <= age < hi:
                    sums[(i, fam)].append(float(r["value"]))
                    break
    rows: list[dict[str, float | str | int]] = []
    for i, (lo, hi) in enumerate(BANDS):
        row: dict[str, float | str | int] = {"age_band": f"{lo}-{hi}"}
        n = 0
        for fam in FAMILIES:
            v = sums.get((i, fam), [])
            row[fam] = round(sum(v) / len(v), 3) if v else float("nan")
            n = max(n, len(v))
        row["n_assessments"] = n
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rows = describe(Path(a.run_dir))
    if a.markdown:
        print("| age band | " + " | ".join(FAMILIES) + " | assessments |")
        print("|---|" + "---|" * (len(FAMILIES) + 1))
        for r in rows:
            print(
                f"| {r['age_band']} | "
                + " | ".join(f"{r[f]:.3f}" for f in FAMILIES)
                + f" | {r['n_assessments']} |"
            )
    else:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
