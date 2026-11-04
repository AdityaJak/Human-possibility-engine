#!/usr/bin/env python3
"""Export the API's OpenAPI document to docs/openapi.json.

python apps/api/export_openapi.py          # write
python apps/api/export_openapi.py --check  # exit 1 if stale (CI)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "apps" / "api"))
OUT = ROOT / "docs" / "openapi.json"


def build() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.setdefault("HPE_DATA_DIR", tmp)
        os.environ.setdefault("HPE_REPORTS_DIR", str(Path(tmp) / "reports"))
        from hpe_api.main import create_app
        from hpe_api.settings import Settings

        app = create_app(Settings(data_dir=Path(tmp), reports_dir=Path(tmp) / "reports"))
        doc = app.openapi()
        app.state.ctx.close()
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("docs/openapi.json is stale; run: python apps/api/export_openapi.py")
            return 1
        print("openapi up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(json.loads(text)['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
