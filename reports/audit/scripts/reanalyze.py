"""Audit 1a: copy each committed experiment dir to a temp dir, re-run analysis from raw logs,
and compare result.json / report.md byte-for-byte with the committed files.

Usage: python reports/audit/scripts/reanalyze.py <tmp_dir>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPS = ["exp_baselines_heldout", "exp01_acquired_learning", "exp06_transfer", "exp_ablations"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for e in EXPS:
        src = ROOT / "reports" / "experiments" / e
        dst = tmp / e
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        proc = subprocess.run(
            [sys.executable, "-m", "hpe.evaluation", "analyze", str(dst)],
            cwd=ROOT, capture_output=True, text=True,
        )
        res = {
            "rc": proc.returncode,
            "result_identical": sha(src / "result.json") == sha(dst / "result.json"),
            "report_identical": sha(src / "report.md") == sha(dst / "report.md"),
            "result_sha256": sha(src / "result.json")[:16],
        }
        out[e] = res
        print(e, res, flush=True)
    (tmp / "reanalyze.json").write_text(json.dumps(out, indent=2))
    return 0 if all(v["result_identical"] and v["rc"] == 0 for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
