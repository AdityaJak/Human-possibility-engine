"""The static exporter pre-renders exactly what the real API serves for a stored run."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hpe.cli import main as hpe_main

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_static_api", ROOT / "scripts" / "export_static_api.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["export_static_api"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_exporter_renders_every_page_endpoint_for_a_tiny_real_run(tmp_path: Path, monkeypatch):
    reports = tmp_path / "reports"
    rc = hpe_main(
        [
            "run",
            "--agents",
            "10",
            "--years",
            "1",
            "--seed",
            "7",
            "--policy",
            "developmental",
            "--out",
            str(reports / "runs"),
            "--run-id",
            "tiny",
            "--frames-every",
            "2",
        ]
    )
    assert rc == 0
    mod = _load_exporter()
    out = tmp_path / "api"
    manifest = mod.export(out, reports, downloads=True, experiment_downloads=False)

    assert (out / "health.json").exists() and (out / "research" / "claims.json").exists()
    assert (out / "benchmarks.json").exists() and (out / "evidence.json").exists()
    runs = json.loads((out / "runs.json").read_text())
    assert [r["run_id"] for r in runs] == ["tiny"] and runs[0]["source"] == "stored_real_output"
    assert json.loads((out / "runs" / "tiny.json").read_text())["run_id"] == "tiny"

    agents = json.loads((out / "runs" / "tiny" / "agents.json").read_text())
    assert agents["total"] == len(agents["items"]) >= 10
    for a in agents["items"]:
        detail = json.loads((out / "runs" / "tiny" / "agents" / f"{a['agent_id']}.json").read_text())
        assert detail["state"]["agent_id"] == a["agent_id"]

    events = json.loads((out / "runs" / "tiny" / "events.json").read_text())
    assert events["total"] == len(events["items"]) > 0
    assert {e["type"] for e in events["items"]} <= set(mod.SIGNIFICANT_EVENT_TYPES)

    frames = json.loads((out / "runs" / "tiny" / "frames.json").read_text())
    assert frames["ticks"] == sorted(frames["ticks"]) and len(frames["frames"]) == len(frames["ticks"]) > 0
    series = json.loads((out / "runs" / "tiny" / "timeseries.json").read_text())
    assert len(series["ticks"]) >= len(frames["ticks"])

    with gzip.open(out / "runs" / "tiny" / "download.jsonl.gz", "rt") as f:
        first = json.loads(f.readline())
    assert isinstance(first, dict) and "tiny" in json.dumps(first)  # same gzip JSONL the API streams

    m = json.loads((out / "_manifest.json").read_text())
    assert m["files"] == manifest["files"] > 20 and m["bytes"] > 0
    assert not any(p.endswith("/events") for p in m["skipped_404"])
