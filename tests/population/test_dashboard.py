"""The Phase III dashboard collector reports committed artifacts verbatim and parses the registry tables."""

from __future__ import annotations

import json
from pathlib import Path

from hpe.population.core.dashboard import collect


def test_collect_reads_summaries_and_registry(tmp_path: Path):
    exp = tmp_path / "reports" / "phase3" / "lane" / "x_demo"
    exp.mkdir(parents=True)
    (exp / "definition.json").write_text(
        json.dumps({"experiment_id": "x_demo", "lane": "lane", "title": "t"})
    )
    summary = {
        "report_partition": "tune",
        "families": {
            "restless4": {
                "role": "counted_primary",
                "best_comparator": "pop_gru",
                "models": {
                    "pop_gru": {"report_nll": 0.7, "ci": [0.69, 0.71], "n": 10, "params": 5, "extra": 1}
                },
                "contrasts": [],
            }
        },
    }
    (exp / "summary.json").write_text(json.dumps(summary))
    docs = tmp_path / "docs" / "phase3"
    docs.mkdir(parents=True)
    (docs / "EXPERIMENT_REGISTRY_V3.md").write_text(
        "## 3\n\n| ID | Date | Lane | Question | Partition | Result | Artifacts |\n|---|---|---|---|---|---|---|\n"
        "| X3-001 | 2026-11-04 | director | q | tune | r | `a/` |\n\n"
        "## 5\n\n| Date | Lane | What | Why abandoned | Artifacts |\n|---|---|---|---|---|\n"
        "| 2026-11-04 | director | w | y | `b/` |\n"
    )
    out = collect(tmp_path / "reports", tmp_path / "docs")
    assert out["verdict"] is None and "not evidence" in out["disclaimer"]
    [e] = out["experiments"]
    m = e["families"]["restless4"]["models"]["pop_gru"]
    assert m["report_nll"] == 0.7 and m["ci"] == [0.69, 0.71] and "extra" not in m
    assert out["registry"]["exploratory"][0]["id"] == "X3-001"
    assert out["registry"]["abandoned"][0]["why"] == "y"
    assert out["registry"]["confirmatory"] == []
