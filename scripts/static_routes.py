"""Materialise the SPA's client-side routes as real directories for static hosts.

GitHub Pages (and any plain file server) answers a deep link such as ``/lab/benchmarks`` with its
``404.html`` — the app still renders from that fallback, but the document is served with HTTP 404,
which browsers, crawlers and our end-to-end check all treat as a broken page. This script copies the
built ``index.html`` to ``<route>/index.html`` for every route the app can serve statically:

* the fixed routes of ``apps/web/src/App.tsx``,
* ``future/<step>`` for every step id declared in ``apps/web/src/pages/future/model.ts``,
* ``lab/runs/<run>/agents/<agent>`` for every agent page pre-rendered under ``<dist>/api/runs``.

``404.html`` stays the fallback for anything else (an unknown route *should* be a 404).

Usage: python scripts/static_routes.py [--dist apps/web/dist] [--web apps/web]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FIXED_ROUTES = (
    "lab",
    "lab/society",
    "lab/learning",
    "lab/experiment",
    "lab/evidence",
    "lab/benchmarks",
    "lab/phase3",
    "future",
    "demo",
)

_STEP_ID = re.compile(r'\{\s*id:\s*"([a-z0-9-]+)"\s*,\s*label:')


def future_steps(model_ts: Path) -> list[str]:
    """Step ids of the Future Self flow, read from the TypeScript source of truth."""
    text = model_ts.read_text(encoding="utf-8")
    start = text.find("export const STEPS")
    if start < 0:
        raise SystemExit(f"STEPS not found in {model_ts}")
    end = text.find("] as const", start)
    steps = _STEP_ID.findall(text[start : end if end > 0 else None])
    if "welcome" not in steps:
        raise SystemExit(f"could not parse Future Self steps from {model_ts}")
    return steps


def agent_routes(api_dir: Path) -> list[str]:
    """``lab/runs/<run>/agents/<agent>`` for every pre-rendered agent detail page."""
    routes: list[str] = []
    runs = api_dir / "runs"
    if not runs.is_dir():
        return routes
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        agents = run_dir / "agents"
        if not agents.is_dir():
            continue
        for agent in sorted(agents.glob("*.json")):
            routes.append(f"lab/runs/{run_dir.name}/agents/{agent.stem}")
    return routes


def all_routes(dist: Path, web: Path) -> list[str]:
    steps = future_steps(web / "src" / "pages" / "future" / "model.ts")
    return [*FIXED_ROUTES, *(f"future/{s}" for s in steps), *agent_routes(dist / "api")]


def materialise(dist: Path, web: Path) -> int:
    index = dist / "index.html"
    if not index.is_file():
        raise SystemExit(f"{index} missing — build the web app first")
    written = 0
    for route in all_routes(dist, web):
        target_dir = dist / route
        if target_dir.exists() and not target_dir.is_dir():
            raise SystemExit(f"route {route!r} collides with a file in {dist}")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(index, target_dir / "index.html")
        written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", default="apps/web/dist")
    ap.add_argument("--web", default="apps/web")
    args = ap.parse_args(argv)
    n = materialise(Path(args.dist), Path(args.web))
    print(f"materialised {n} routes under {args.dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
