"""Paired contrasts between any two arms of a human-validation v2 run, from its raw outcomes.

The runner reports only `developmental − …` contrasts. This recomputes any pair (e.g. Candidate A vs
the population-fitted GRU) from `outcomes.jsonl.gz` with the same per-participant pairing and the
same paired bootstrap (`hpe.evaluation.human.report.bootstrap_mean`), so numbers are directly
comparable with the run's own report.

Usage: python scripts/phase2_contrasts.py <run_dir> <paradigm> <first_model> <second_model> [...]
       python scripts/phase2_contrasts.py reports/phase2/human/p2_human_dev_pilot garcia developmental ref_gru
Several pairs may be given as first:second tokens. Negative estimates favour the first model.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

from hpe.evaluation.human.report import bootstrap_mean

METRIC = "nll_per_choice"


def per_participant(run_dir: Path, paradigm: str) -> dict[str, dict[str, float]]:
    """{model: {participant: nll}} for one paradigm."""
    out: dict[str, dict[str, float]] = {}
    with gzip.open(run_dir / "outcomes.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("metric") != METRIC:
                continue
            arm = str(r.get("arm", ""))
            fam, _, model = arm.partition(":")
            if fam != paradigm:
                continue
            out.setdefault(model, {})[str(r["agent_id"])] = float(r["value"])
    return out


def contrast(
    table: dict[str, dict[str, float]], first: str, second: str, n_boot: int = 2000, seed: int = 20261105
) -> dict[str, float | int | str]:
    a, b = table.get(first, {}), table.get(second, {})
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit(f"no paired participants for {first} vs {second}")
    d = np.array([a[i] - b[i] for i in ids])
    est, lo, hi, p = bootstrap_mean(d, n_boot, seed, first, second)
    dz = float(est / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
    return {
        "first": first,
        "second": second,
        "estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "p": p,
        "dz": dz,
        "n": len(ids),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    run_dir, paradigm, *pairs = argv
    table = per_participant(Path(run_dir), paradigm)
    if not table:
        raise SystemExit(f"no {METRIC} rows for paradigm {paradigm!r} in {run_dir}")
    if len(pairs) == 2 and ":" not in pairs[0]:
        pairs = [f"{pairs[0]}:{pairs[1]}"]
    print(f"{paradigm}: models {sorted(table)}")
    print("| contrast | estimate | 95% CI | p | dz | n |\n|---|---|---|---|---|---|")
    for pair in pairs:
        first, _, second = pair.partition(":")
        c = contrast(table, first, second)
        print(
            f"| {first} − {second} | {c['estimate']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
            f"| {c['p']:.4f} | {c['dz']:.2f} | {c['n']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
