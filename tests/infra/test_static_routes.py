"""Static-host route materialisation: every client-side route gets a real index.html."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("static_routes", ROOT / "scripts" / "static_routes.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["static_routes"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_future_steps_are_read_from_the_app_source():
    mod = _load()
    steps = mod.future_steps(ROOT / "apps/web/src/pages/future/model.ts")
    assert steps[0] == "welcome" and "conversation" in steps and "privacy" in steps
    assert len(steps) == len(set(steps)) >= 10


def test_materialise_writes_every_route_and_keeps_the_fallback(tmp_path: Path):
    mod = _load()
    dist = tmp_path / "dist"
    (dist / "api" / "runs" / "run_a" / "agents").mkdir(parents=True)
    (dist / "api" / "runs" / "run_b").mkdir()  # no agent pages -> no agent routes
    (dist / "api" / "runs" / "run_a" / "agents" / "a00001.json").write_text("{}")
    (dist / "api" / "runs" / "run_a" / "agents" / "a00000.json").write_text("{}")
    (dist / "index.html").write_text("<!doctype html><title>x</title>")
    (dist / "404.html").write_text("<!doctype html><title>x</title>")

    n = mod.materialise(dist, ROOT / "apps/web")

    routes = mod.all_routes(dist, ROOT / "apps/web")
    assert n == len(routes)
    assert "lab/runs/run_a/agents/a00000" in routes and "lab/runs/run_a/agents/a00001" in routes
    assert not any(r.startswith("lab/runs/run_b") for r in routes)
    for route in ("lab/benchmarks", "future/welcome", "demo", "lab/runs/run_a/agents/a00001"):
        assert (dist / route / "index.html").read_text() == (dist / "index.html").read_text()
    assert (dist / "404.html").exists()
    assert (dist / "api" / "runs" / "run_a" / "agents" / "a00000.json").read_text() == "{}"
    # idempotent
    assert mod.materialise(dist, ROOT / "apps/web") == n
