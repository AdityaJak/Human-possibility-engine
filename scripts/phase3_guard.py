#!/usr/bin/env python3
"""Phase III leakage guard: no test-partition participant key may appear in any committed artifact
outside the confirmatory directory, for either partition file used in Phase III.

    python scripts/phase3_guard.py [PATH ...]      (default: reports docs experiments packages scripts tests)

Scans with the Phase II scanner (``scripts/phase2_guard.py``) against the union of
``reports/phase2/partitions/psych101.json`` and ``reports/phase3/partitions/psych101_p3.json`` test keys.
Allowed, and reported as disclosed rather than ignored: the partition files themselves, the exposed-
participant disclosure (``reports/phase3/partitions/exposed_participants.json``), Phase I human-validation
artifacts (pre-freeze contact, Phase II registry §1) and ``reports/phase3/confirmatory/`` (post-freeze).
Exit 1 on any violation. No model is fitted; only files are read.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "reports/phase2/partitions/psych101.json"
P3 = ROOT / "reports/phase3/partitions/psych101_p3.json"
ALLOWED = (
    "reports/phase2/partitions/",
    "reports/phase3/partitions/",
    "reports/human_validation/",
    "reports/phase3/confirmatory/",
    "reports/phase2/critic/guard_run",
)
DEFAULT_PATHS = ("reports", "docs", "experiments", "packages", "scripts", "tests")
# Broader than the Phase II pattern: Phase III study files include names without a digit (bahrami2020four/exp.csv).
KEY_RE = re.compile(r"([A-Za-z0-9_.\-À-ɏ]+/exp\d*\.csv)#([A-Za-z0-9_\-]+)")


def _phase2_guard():
    spec = importlib.util.spec_from_file_location("phase2_guard", ROOT / "scripts" / "phase2_guard.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules["phase2_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    paths = [ROOT / p for p in (args or DEFAULT_PATHS)]
    g = _phase2_guard()
    keys = g.load_test_keys(P2) | g.load_test_keys(P3)
    violations, disclosed, scanned = [], [], 0
    for f in g.iter_files(paths):
        rel = g._rel(f, ROOT)
        scanned += 1
        hits: dict[str, int] = {}
        for line in g.read_lines(f):
            for m in KEY_RE.finditer(line):
                k = f"{m.group(1)}#{m.group(2)}"
                if k in keys:
                    hits[k] = hits.get(k, 0) + 1
        if hits:
            rec = {"file": rel, "distinct_test_keys": len(hits), "sample": sorted(hits)[:3]}
            (disclosed if rel.startswith(ALLOWED) else violations).append(rec)
    # Critic finding T-3: lane code must not read raw Psych-101 records (only hpe.population.core.data may).
    raw_patterns = ("iter_records(", "prompts_training", "psych101_phase3_subset", "psych101_learning_subset")
    allowed_code = (
        "packages/hpe/population/core/data.py",
        "packages/hpe/evaluation/human/",
        "scripts/phase2_",
        "scripts/phase3_partitions.py",
        "scripts/phase3_guard.py",
        "datasets/",
    )
    raw_access = []
    for base in ("packages/hpe/population", "scripts", "experiments/phase3"):
        for f in sorted((ROOT / base).rglob("*.py")):
            rel = str(f.relative_to(ROOT))
            if rel.startswith(allowed_code):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            hits = [pat for pat in raw_patterns if pat in text]
            if hits:
                raw_access.append({"file": rel, "patterns": hits})
    violations += [{"kind": "raw_record_access", **r} for r in raw_access]
    out = {
        "check": "phase3 check-test-access",
        "n_test_keys": len(keys),
        "files_scanned": scanned,
        "violations": violations,
        "disclosed": disclosed,
        "clean": not violations,
    }
    print(json.dumps(out, indent=1))
    return 0 if out["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
