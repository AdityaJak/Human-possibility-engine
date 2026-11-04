#!/usr/bin/env python3
"""Download and verify Psych-101 (Apache-2.0), then extract the learning-experiment subset.

    python datasets/adapters/psych101.py            # -> datasets/external/psych101_learning_subset.jsonl
    python datasets/adapters/psych101.py --check    # only verify an existing local copy

The full file is ~859 MB. Nothing is committed; see docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from hpe.evaluation.human import data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify an existing copy only")
    args = ap.parse_args()
    subset = data.default_path(data.PSYCH101_LEARNING_SUBSET)
    if args.check:
        path = data.locate(data.PSYCH101_LEARNING_SUBSET)
        print(f"ok {path} sha256={data.sha256_file(path)}")
        return 0
    full = data.default_path(data.PSYCH101_FULL)
    if not full.exists():
        print(f"downloading {data.PSYCH101_FULL.url} -> {full}")
        data.download(data.PSYCH101_FULL.url, full, data.PSYCH101_FULL.sha256)
    n = data.filter_jsonl(full, subset, data.LEARNING_EXPERIMENT_PREFIXES)
    data.verify(subset, data.PSYCH101_LEARNING_SUBSET.sha256)
    print(f"wrote {subset} ({n} participant transcripts, sha256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
