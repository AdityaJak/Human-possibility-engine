"""Collect committed Phase III results for the dashboard (no torch, no data access; files only).

Reads every ``reports/phase3/<lane>/<experiment_id>/summary.json`` (+ ``definition.json``) and the
tables of ``docs/phase3/EXPERIMENT_REGISTRY_V3.md`` (exploratory, confirmatory, abandoned, deviations),
and ``reports/phase3/verdict.json`` when the director has written one. Nothing is computed here that is
not already in a committed artifact; the dashboard shows exactly what the reports say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DISCLAIMER = (
    "Phase III research results on real human choice data (Psych-101, Apache-2.0). Exploratory rows use "
    "development/tuning participants and are selection-optimistic; only confirmatory rows can pass a gate. "
    "Predicting choices is not evidence that any model improves decisions, and nothing here is a forecast "
    "about any individual."
)

_KEEP_MODEL_FIELDS = (
    "report_nll",
    "ci",
    "n",
    "top1",
    "ece",
    "quarters",
    "select_nll",
    "params",
    "role",
    "lapse",
    "grid_edges",
    "fit_cpu_s",
    "causality_violations",
)


def _table_rows(md: str, header_start: str) -> list[list[str]]:
    """Rows of the first markdown table whose header line starts with ``header_start``."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            rows = []
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip().strip("|").split("|")])
            return rows
    return []


def registry(docs_dir: Path) -> dict[str, list[dict[str, str]]]:
    p = Path(docs_dir) / "phase3" / "EXPERIMENT_REGISTRY_V3.md"
    if not p.exists():
        return {"exploratory": [], "confirmatory": [], "abandoned": [], "amendments": []}
    md = p.read_text(encoding="utf-8")
    spec = {
        "exploratory": (
            "| ID | Date | Lane | Question",
            ["id", "date", "lane", "question", "partition", "result", "artifacts"],
        ),
        "confirmatory": (
            "| ID | Date | Gate |",
            ["id", "date", "gate", "preregistration", "freeze", "result", "verdict", "artifacts"],
        ),
        "abandoned": ("| Date | Lane | What |", ["date", "lane", "what", "why", "artifacts"]),
        "amendments": ("| Date | Change |", ["date", "change", "reason", "seen_confirmatory"]),
    }
    out: dict[str, list[dict[str, str]]] = {}
    for key, (hdr, cols) in spec.items():
        out[key] = [dict(zip(cols, r, strict=False)) for r in _table_rows(md, hdr) if any(r)]
    return out


def experiments(reports_dir: Path) -> list[dict[str, Any]]:
    root = Path(reports_dir) / "phase3"
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for summ in sorted(root.glob("*/*/summary.json")):
        d = summ.parent
        try:
            summary = json.loads(summ.read_text(encoding="utf-8"))
            defn = json.loads((d / "definition.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fams: dict[str, Any] = {}
        for fam, f in summary.get("families", {}).items():
            fams[fam] = {
                "role": f.get("role"),
                "best_comparator": f.get("best_comparator"),
                "models": {
                    name: {k: m.get(k) for k in _KEEP_MODEL_FIELDS} for name, m in f.get("models", {}).items()
                },
                "contrasts": f.get("contrasts", []),
            }
        out.append(
            {
                "experiment_id": defn.get("experiment_id", d.name),
                "lane": defn.get("lane", d.parent.name),
                "title": defn.get("title", ""),
                "hypothesis": defn.get("hypothesis", ""),
                "report_partition": summary.get("report_partition", defn.get("report_partition")),
                "path": str(d.relative_to(Path(reports_dir).parent))
                if Path(reports_dir).is_absolute()
                else str(d),
                "preregistration_hash": defn.get("preregistration_hash"),
                "families": fams,
                "candidate_vs_best_comparator": summary.get("candidate_vs_best_comparator", {}),
            }
        )
    return out


def collect(reports_dir: Path, docs_dir: Path) -> dict[str, Any]:
    verdict_path = Path(reports_dir) / "phase3" / "verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8")) if verdict_path.exists() else None
    return {
        "disclaimer": DISCLAIMER,
        "experiments": experiments(reports_dir),
        "registry": registry(docs_dir),
        "verdict": verdict,
    }
