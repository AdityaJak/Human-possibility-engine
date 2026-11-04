"""Audit 1c: every committed RunManifest must carry code_version, seed, config_hash and a log sha256.

Usage: python reports/audit/scripts/manifest_check.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from hpe.contracts.schemas import RunManifest

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    paths = sorted((ROOT / "reports").rglob("manifest.json"))
    problems: list[str] = []
    versions: collections.Counter[str] = collections.Counter()
    dirty = 0
    for p in paths:
        try:
            m = RunManifest.model_validate_json(p.read_text())
        except Exception as e:  # noqa: BLE001
            problems.append(f"{p.relative_to(ROOT)}: not a valid RunManifest ({type(e).__name__})")
            continue
        missing = [
            k
            for k, v in {
                "code_version": m.code_version,
                "config_hash": m.config_hash,
                "event_log_sha256": m.event_log_sha256,
                "compute": m.compute,
            }.items()
            if not v
        ]
        if missing:
            problems.append(f"{p.relative_to(ROOT)}: missing {missing}")
        versions[str(m.code_version)] += 1
        if m.code_version and "dirty" in m.code_version:
            dirty += 1
        if m.config.seed is None:
            problems.append(f"{p.relative_to(ROOT)}: no seed")
    out = {
        "n_manifests": len(paths),
        "problems": problems,
        "code_versions": dict(versions.most_common()),
        "dirty_code_versions": dirty,
    }
    print(json.dumps(out, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
