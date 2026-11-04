#!/usr/bin/env python3
"""Phase III participant partitions for Psych-101 studies NEVER used in Phases I-II.

    HPE_PSYCH101_FULL=<path to prompts_training.jsonl> python scripts/phase3_partitions.py

Frozen before any Phase III model is fitted (docs/phase3/PHASE3_RESEARCH_THESIS.md). Same scheme as
Phase II (scripts/phase2_partitions.py): dev 40 % / tune 20 % / test 40 % per experiment file, a
permutation seeded by (split seed, SHA-256 of the experiment name), and a SHA-256 over the test keys.
Only participant identifiers are written (no transcripts). The Phase II partition file stays authoritative
for the families it covers (garcia2023experiential, plonsky2018when, gershman*, wilson2014, ...).

Families and their role (fixed here, before fitting):
    restless4         bahrami2020four/                    4-armed restless bandit (primary, counted)
    two_step          kool2016when/ kool2017cost/ zorowitz2023data/   two-step tasks (counted if parser validated pre-freeze)
    spatial_bandit    schulz2020finding/                  8-option bandit rounds (exploratory only)
    horizon_external  sadeghiyeh2020temporal/ somerville2017charting/ feng2021dynamics/
                      independent replications of the horizon task; for the external-replication analysis only
                      their TEST participants are scored and their dev/tune participants are used for parser
                      validation only (never fitted on)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLIT_SEED = 20261106
FRACTIONS = {"dev": 0.4, "tune": 0.2, "test": 0.4}
FAMILIES = {
    "restless4": ("bahrami2020four/",),
    "two_step": ("kool2016when/", "kool2017cost/", "zorowitz2023data/"),
    "spatial_bandit": ("schulz2020finding/",),
    "horizon_external": ("sadeghiyeh2020temporal/", "somerville2017charting/", "feng2021dynamics/"),
}
EXPECTED_FULL_SHA256 = "7a370a94084acd6f73f20d6559214683cfadcab8a1f69e369df285ba75ccef01"


def main() -> int:
    src = os.environ.get("HPE_PSYCH101_FULL")
    if not src or not Path(src).exists():
        print("set HPE_PSYCH101_FULL to Psych-101 prompts_training.jsonl", file=sys.stderr)
        return 2
    sha = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    if sha != EXPECTED_FULL_SHA256:
        print(f"unexpected source sha256 {sha}", file=sys.stderr)
        return 2
    prefixes = tuple(p for ps in FAMILIES.values() for p in ps)
    by_exp: dict[str, set[str]] = defaultdict(set)
    with open(src, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["experiment"].startswith(prefixes):
                by_exp[d["experiment"]].add(str(d["participant"]))
    out: dict[str, object] = {
        "split_seed": SPLIT_SEED,
        "fractions": FRACTIONS,
        "source_sha256": sha,
        "families": {},
    }
    test_concat = []
    for fam, fam_prefixes in FAMILIES.items():
        fam_out: dict[str, object] = {}
        for exp in sorted(e for e in by_exp if e.startswith(fam_prefixes)):
            ids = sorted(by_exp[exp], key=lambda s: (len(s), s))
            digest = int.from_bytes(hashlib.sha256(exp.encode()).digest()[:4], "little")
            rng = np.random.default_rng([SPLIT_SEED, digest])
            perm = [ids[i] for i in rng.permutation(len(ids))]
            n = len(perm)
            n_dev, n_tune = int(round(n * FRACTIONS["dev"])), int(round(n * FRACTIONS["tune"]))
            parts = {
                "dev": sorted(perm[:n_dev]),
                "tune": sorted(perm[n_dev : n_dev + n_tune]),
                "test": sorted(perm[n_dev + n_tune :]),
            }
            fam_out[exp] = {"n": n, **parts, "counts": {k: len(v) for k, v in parts.items()}}
            test_concat += [f"{exp}#{p}" for p in parts["test"]]
        out["families"][fam] = fam_out  # type: ignore[index]
    out["test_partition_sha256"] = hashlib.sha256("\n".join(sorted(test_concat)).encode()).hexdigest()
    dest = Path("reports/phase3/partitions/psych101_p3.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for fam, exps in out["families"].items():  # type: ignore[union-attr]
        for exp, v in exps.items():
            c = v["counts"]
            print(
                f"{fam:18s} {exp:38s} n={v['n']:4d} dev={c['dev']:4d} tune={c['tune']:3d} test={c['test']:4d}"
            )
    print("test partition sha256:", out["test_partition_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
