#!/usr/bin/env python3
"""Pre-render the read-only API surface to static JSON for the GitHub Pages deployment.

    python scripts/export_static_api.py --out apps/web/public/api [--reports reports] [--no-wheel]

Starts the REAL FastAPI application in-process (hpe_api.main.create_app + TestClient) against the
committed stored real outputs under ``--reports`` and writes, for every GET the web app uses,
``<out>/<path>.json`` holding exactly the bytes the live API returns. Parameterised endpoints are
written once in their fullest form; the static transport (apps/web/src/api/static/get.ts) applies
query filters client-side:

    /runs/{id}/agents       every agent (alive and dead), one AgentPage
    /runs/{id}/agents/{aid} one AgentDetail per agent
    /runs/{id}/events       every event of the significant types the dashboard feed shows
    /runs/{id}/frames       every stored frame tick (every=1)
    /runs/{id}/timeseries   every WorldTick
    /runs/{id}/download     -> runs/{id}/download.jsonl.gz (same gzip JSONL the API streams)

Nothing here is authored by hand: if the API has no data for a path, no file is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "apps" / "api"))

# Must match SIGNIFICANT_EVENT_TYPES in apps/web/src/lib/events.ts (the Live Society feed).
SIGNIFICANT_EVENT_TYPES = [
    "birth",
    "death",
    "school_enroll",
    "school_graduate",
    "job_start",
    "job_end",
    "partnership_formed",
    "shock",
    "intervention",
]

EVENTS_PAGE = 5000  # API maximum for /events limit
AGENTS_PAGE = 20000  # API maximum for /agents limit
FRAMES_PAGE = 5000  # API maximum for /frames limit
FRAMES_EVERY = 3  # native stride of the pre-rendered frames; the client applies `every` relative to it


class Exporter:
    def __init__(self, client: Any, out: Path) -> None:
        self.client = client
        self.out = out
        self.files: dict[str, int] = {}
        self.skipped: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any | None:
        r = self.client.get(path, params=params)
        if r.status_code == 404:
            self.skipped.append(path)
            return None
        r.raise_for_status()
        return r.json()

    def _write(self, path: str, payload: Any) -> None:
        target = self.out / (path.lstrip("/") + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        target.write_bytes(data)
        self.files[str(target.relative_to(self.out))] = len(data)

    def _write_bytes(self, rel: str, data: bytes) -> None:
        target = self.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files[rel] = len(data)

    def dump(self, path: str, params: dict[str, Any] | None = None) -> Any | None:
        payload = self._get(path, params)
        if payload is not None:
            self._write(path, payload)
        return payload

    # ------------------------------------------------------------------ runs
    def export_run(self, run_id: str, *, downloads: bool) -> None:
        base = f"/runs/{run_id}"
        self.dump(base)
        # agents: one page holding every agent, alive and dead
        first = self._get(f"{base}/agents", {"limit": AGENTS_PAGE, "offset": 0})
        if first is None:
            return
        items = list(first["items"])
        total = int(first["total"])
        offset = len(items)
        while offset < total:
            page = self._get(f"{base}/agents", {"limit": AGENTS_PAGE, "offset": offset})
            if page is None or not page["items"]:
                break
            items.extend(page["items"])
            offset += len(page["items"])
        self._write(f"{base}/agents", {"items": items, "total": total, "offset": 0, "limit": max(total, 1)})
        for a in items:
            self.dump(f"{base}/agents/{a['agent_id']}")
        # events of the significant types, fully paged
        ev_items: list[Any] = []
        ev_total = 0
        offset = 0
        while True:
            page = self._get(
                f"{base}/events",
                {"types": SIGNIFICANT_EVENT_TYPES, "limit": EVENTS_PAGE, "offset": offset},
            )
            if page is None:
                break
            ev_total = int(page["total"])
            ev_items.extend(page["items"])
            offset += len(page["items"])
            if not page["items"] or offset >= ev_total:
                break
        self._write(
            f"{base}/events",
            {"items": ev_items, "total": ev_total, "offset": 0, "limit": max(len(ev_items), 1)},
        )
        # frames: every stored tick (the dashboard subsamples client-side)
        frames = self._get(f"{base}/frames", {"every": FRAMES_EVERY, "limit": FRAMES_PAGE})
        if frames is not None:
            self._write(f"{base}/frames", frames)
        self.dump(f"{base}/timeseries", {"every": 1})
        if downloads:
            self.export_download(run_id)

    def export_download(self, run_id: str) -> None:
        r = self.client.get(f"/runs/{run_id}/download")
        if r.status_code != 200:
            self.skipped.append(f"/runs/{run_id}/download")
            return
        self._write_bytes(f"runs/{run_id}/download.jsonl.gz", r.content)

    # ------------------------------------------------------------------ everything
    def export_all(self, *, downloads: bool = True, experiment_downloads: bool = True) -> None:
        self.dump("/health")
        runs = self.dump("/runs") or []
        for run in runs:
            self.export_run(run["run_id"], downloads=downloads)
        experiments = self._get("/experiments") or []
        # The list is only used for definitions/status; per-experiment outcome samples are served by
        # /experiments/{id}/results, so they are not duplicated into the list file.
        self._write("/experiments", [{**e, "outcomes_sample": []} for e in experiments])
        exp_run_ids: list[str] = []
        for exp in experiments:
            info = self.dump(f"/experiments/{exp['experiment_id']}/results")
            if info and info.get("result"):
                exp_run_ids.extend(info["result"].get("run_ids", []))
        self.dump("/research/claims")
        self.dump("/research/metrics")
        self.dump("/research/phase3")
        self.dump("/benchmarks")
        evidence = self.dump("/evidence") or []
        for rec in evidence:
            self.dump(f"/evidence/{rec['evidence_id']}")
        if downloads and experiment_downloads:
            seen = {r["run_id"] for r in runs}
            for rid in exp_run_ids:
                if rid in seen:
                    continue
                seen.add(rid)
                self.export_download(rid)

    # ------------------------------------------------------------------ report
    def manifest(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for rel, size in self.files.items():
            key = rel.split("/")[0] if "/" in rel else rel
            if rel.startswith("runs/") and rel.endswith(".jsonl.gz"):
                key = "downloads"
            elif rel.startswith("runs/") and "/agents/" in rel:
                key = "agent_details"
            groups[key] = groups.get(key, 0) + size
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": _git_sha(),
            "files": len(self.files),
            "bytes": sum(self.files.values()),
            "groups_bytes": dict(sorted(groups.items())),
            "skipped_404": self.skipped,
            "note": "Pre-rendered by scripts/export_static_api.py from the real API over stored real outputs.",
        }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def build_wheel(dest: Path) -> Path | None:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.whl"):
        old.unlink()
    subprocess.run([uv, "build", "--wheel", "-o", str(dest)], cwd=ROOT, check=True)
    wheels = sorted(dest.glob("*.whl"))
    return wheels[0] if wheels else None


def export(
    out: Path, reports: Path, *, downloads: bool = True, experiment_downloads: bool = True
) -> dict[str, Any]:
    out = out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="hpe-static-") as tmp:
        os.environ["HPE_REPORTS_DIR"] = str(reports.resolve())
        os.environ["HPE_DATA_DIR"] = tmp
        os.environ.setdefault("HPE_LLM_PROVIDER", "none")
        from fastapi.testclient import TestClient

        from hpe_api.main import create_app

        with TestClient(create_app()) as client:
            ex = Exporter(client, out)
            ex.export_all(downloads=downloads, experiment_downloads=experiment_downloads)
            manifest = ex.manifest()
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="apps/web/public/api")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--wheels", default="apps/web/public/wheels", help="where to build the hpe wheel")
    ap.add_argument("--no-wheel", action="store_true")
    ap.add_argument("--no-downloads", action="store_true")
    ap.add_argument("--no-experiment-downloads", action="store_true")
    ap.add_argument("--budget-mb", type=float, default=40.0, help="warn above this total size")
    a = ap.parse_args()
    manifest = export(
        Path(a.out),
        Path(a.reports),
        downloads=not a.no_downloads,
        experiment_downloads=not a.no_experiment_downloads,
    )
    print(f"static API: {manifest['files']} files, {manifest['bytes'] / 1e6:.1f} MB -> {a.out}")
    for k, v in manifest["groups_bytes"].items():
        print(f"  {k:16s} {v / 1e6:7.2f} MB")
    if manifest["skipped_404"]:
        print(f"  skipped (404): {len(manifest['skipped_404'])} paths")
    if manifest["bytes"] > a.budget_mb * 1e6:
        print(f"  WARNING: total exceeds the {a.budget_mb:.0f} MB budget", file=sys.stderr)
    if not a.no_wheel:
        wheel = build_wheel(Path(a.wheels))
        print(f"wheel: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
