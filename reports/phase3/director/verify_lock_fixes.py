"""Director's re-run of the critic's lock checks (``reports/phase3/critic/lock_checks.py``, unchanged), with each
negative check wrapped so that a refusal (exception) is recorded as the outcome. Output:
reports/phase3/director/verify_lock_fixes.json. Not the critic's sign-off."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("critic_lock", ROOT / "reports/phase3/critic/lock_checks.py")
lc = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["critic_lock"] = lc
spec.loader.exec_module(lc)  # type: ignore[union-attr]


def attempt(fn):  # type: ignore[no-untyped-def]
    import hpe.population.core.data as d

    saved = d.EXPOSED_FILE
    try:
        return {"returned": fn()}
    except Exception as e:  # a refusal is the expected outcome after the fixes
        return {"raised": type(e).__name__, "message": str(e)[:300], "where": traceback.format_exc().splitlines()[-3][:200]}
    finally:
        d.EXPOSED_FILE = saved
        d.exposed_keys.cache_clear()


def main() -> int:
    res = {name: attempt(fn) for name, fn in (
        ("L1_forged_freeze_record", lc.forged_freeze),
        ("L3_tampered_partitions", lc.tampered_partitions),
        ("L4_exposed_file_fail_open", lc.exposed_fail_open),
        ("L7_guard_blind_to_phase3_keys", lc.guard_regex),
    )}
    out = ROOT / "reports/phase3/director/verify_lock_fixes.json"
    out.write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, default=str, ensure_ascii=False)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
