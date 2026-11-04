#!/usr/bin/env python3
"""Download and verify HeartSteps V1 suggestions.csv (CC-BY-4.0).

    python datasets/adapters/heartsteps.py           # -> datasets/external/heartsteps_v1_suggestions.csv
    python datasets/adapters/heartsteps.py --check   # verify + print data checks

Nothing is committed; see docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from hpe.evaluation.human import data, heartsteps  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = data.HEARTSTEPS_SUGGESTIONS
    if args.check:
        path = data.locate(src)
    else:
        path = data.default_path(src)
        if not path.exists():
            print(f"downloading {src.url} -> {path}")
            data.download(src.url, path, src.sha256)
        data.verify(path, src.sha256)
    print(json.dumps(heartsteps.validate(heartsteps.load_decision_points(path)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
