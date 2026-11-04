#!/usr/bin/env python3
"""Phase II human participant partitions (dev 40% / tune 20% / test 40%), frozen before model development.

    HPE_PSYCH101_SUBSET=<path to psych101 learning subset jsonl> python scripts/phase2_partitions.py

Writes reports/phase2/partitions/psych101.json: per experiment file, sorted participant keys per
partition, counts, the split seed, and a SHA-256 over the test partition so later tampering is detectable.
Only participant identifiers are stored (no transcripts).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLIT_SEED = 20261105
FRACTIONS = {"dev": 0.4, "tune": 0.2, "test": 0.4}
FAMILIES = {
    "mcpl": ("collsiöö2023MCPL/",),
    "igt": ("steingroever2015data/",),
    "two_armed_bandit": ("gershman2018deconstructing/", "gershman2020reward/"),
    "horizon_task": ("wilson2014humans/",),
    "probability_learning": ("speekenbrink2008learning/",),
    "reversal_learning": ("lefebvre2017behavioural/", "waltz2020differential/"),
    "experience_based_choice": ("garcia2023experiential/", "plonsky2018when/"),
}


def main() -> int:
    src = os.environ.get("HPE_PSYCH101_SUBSET")
    if not src or not Path(src).exists():
        print(
            "set HPE_PSYCH101_SUBSET to the Psych-101 learning subset (see datasets/adapters/psych101.py)",
            file=sys.stderr,
        )
        return 2
    by_exp: dict[str, set[str]] = defaultdict(set)
    with open(src, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            by_exp[d["experiment"]].add(str(d["participant"]))
    out: dict[str, object] = {
        "split_seed": SPLIT_SEED,
        "fractions": FRACTIONS,
        "source_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest(),
        "families": {},
    }
    test_concat = []
    for fam, prefixes in FAMILIES.items():
        fam_out: dict[str, object] = {}
        for exp in sorted(e for e in by_exp if e.startswith(prefixes)):
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
            fam_out[exp] = {
                "n": n,
                **{k: v for k, v in parts.items()},
                "counts": {k: len(v) for k, v in parts.items()},
            }
            test_concat += [f"{exp}#{p}" for p in parts["test"]]
        out["families"][fam] = fam_out
    out["test_partition_sha256"] = hashlib.sha256("\n".join(sorted(test_concat)).encode()).hexdigest()
    dest = Path("reports/phase2/partitions/psych101.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for fam, exps in out["families"].items():  # type: ignore[union-attr]
        for exp, v in exps.items():
            print(
                f"{fam:24s} {exp:40s} n={v['n']:4d} dev={v['counts']['dev']:4d} tune={v['counts']['tune']:3d} test={v['counts']['test']:4d}"
            )
    print("test partition sha256:", out["test_partition_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
