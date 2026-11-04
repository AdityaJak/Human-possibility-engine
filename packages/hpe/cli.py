"""``hpe`` command-line interface (no API keys required).

hpe info                         versions and registered policy classes
hpe smoke [--out DIR]            seeded smoke test (CI): 25 agents x 3 years, replay-determinism check
hpe run --agents N --years Y     run one society and store it in the §5 run layout
hpe demo-run                     the stored 100-agent demo run used by the offline demo mode
"""

from __future__ import annotations

import argparse
import importlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

from hpe.contracts.schemas import LogConfig, PolicyClass, SimulationConfig


def _world_available() -> bool:
    try:
        importlib.import_module("hpe.world").World  # noqa: B018
        return True
    except (ImportError, AttributeError):
        return False


def run_and_store(config: SimulationConfig, out: Path, run_id: str, *, quiet: bool = False) -> dict[str, Any]:
    """Run one world to completion and write the INTERFACES §5 stored-run layout to ``out``."""
    from hpe.contracts.registry import get_factory
    from hpe.storage.run_store import RunDirectory
    from hpe.storage.sinks import JsonlGzSink

    world_cls = importlib.import_module("hpe.world").World
    factory = get_factory(config.policy_class.value, config.policy_params)
    out.mkdir(parents=True, exist_ok=True)
    sink = JsonlGzSink(out)
    world = world_cls(config, factory, sink=sink, run_id=run_id)
    t0, c0 = time.perf_counter(), time.process_time()
    last = [-1.0]

    def progress(p: float) -> None:
        if not quiet and p - last[0] >= 0.1:
            last[0] = p
            print(f"  {run_id}: {p:5.0%}  year {world.year:5.1f}", flush=True)

    final_tick = world.run(progress=progress)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    sink.close()
    rd = RunDirectory(out)
    rd.write_agents(world.agents())
    rd.write_edges(world.edges())
    rd.write_map(world.world_map())
    rd.write_outcomes(world.outcomes())
    manifest = world.manifest()
    updates: dict[str, Any] = {
        "event_log_path": "events.jsonl.gz",
        "event_log_sha256": getattr(sink, "sha256", None) or manifest.event_log_sha256,
        "n_events": getattr(sink, "n_events", manifest.n_events),
    }
    manifest = manifest.model_copy(update=updates)
    rd.write_manifest(manifest)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "run_id": run_id,
        "event_stream_hash": world.event_stream_hash(),
        "population": final_tick.population,
        "year": world.year,
        "events": updates["n_events"],
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "peak_rss_mb": round(peak, 1),
    }


def _learning_smoke() -> int:
    from hpe.contracts.conformance import make_init
    from hpe.contracts.registry import get_factory
    from hpe.core import stream
    from hpe.tasks import SymbolWorld, assess, practice

    w = SymbolWorld(seed=0)
    pol = get_factory("developmental")(make_init(w))
    before = assess(pol, w, "compare", 20.0).accuracy
    practice(
        pol, w, "compare", 200, stream(0, "smoke"), agent_id="a0000", tick=0, age_years=20.0, mode="full"
    )
    after = assess(pol, w, "compare", 20.0).accuracy
    print(f"learning smoke: held-out compare accuracy {before:.3f} -> {after:.3f}")
    return 0 if after > before else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    if not _world_available():
        print("hpe.world not integrated in this build; running learning smoke only")
        return _learning_smoke()
    cfg = SimulationConfig(
        seed=args.seed,
        n_agents=args.agents,
        years=args.years,
        policy_class=PolicyClass(args.policy),
        log=LogConfig(frames_every_ticks=6, checkpoint_every_years=None),
        label="smoke",
    )
    out = Path(args.out)
    a = run_and_store(cfg, out / "run_a", "smoke_a", quiet=True)
    b = run_and_store(cfg, out / "run_b", "smoke_b", quiet=True)
    ok = a["event_stream_hash"] == b["event_stream_hash"]
    print(json.dumps({"run_a": a, "run_b": b, "replay_identical": ok}, indent=2))
    return 0 if ok and a["events"] > 0 else 1


def cmd_run(args: argparse.Namespace) -> int:
    if not _world_available():
        print("hpe.world is not available in this build", file=sys.stderr)
        return 2
    cfg = SimulationConfig(
        seed=args.seed,
        n_agents=args.agents,
        years=args.years,
        dt_days=args.dt_days,
        policy_class=PolicyClass(args.policy),
        log=LogConfig(frames_every_ticks=args.frames_every),
        label=args.label,
    )
    run_id = args.run_id or f"run_{args.policy}_{args.agents}a_{int(args.years)}y_s{args.seed}"
    res = run_and_store(cfg, Path(args.out) / run_id, run_id)
    print(json.dumps(res, indent=2))
    return 0


def cmd_demo_run(args: argparse.Namespace) -> int:
    args.agents, args.years, args.dt_days = 100, 80.0, 30
    args.run_id = args.run_id or f"demo_developmental_100a_80y_s{args.seed}"
    args.out = args.out or "reports/runs"
    args.label = "investor demo: 100 agents, 80 simulated years"
    return cmd_run(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hpe", description="Human Possibility Engine CLI (no API keys required)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="print versions and registered policy classes")
    sp = sub.add_parser("smoke", help="seeded smoke simulation with replay-determinism check")
    sp.add_argument("--out", default="data/smoke")
    sp.add_argument("--agents", type=int, default=25)
    sp.add_argument("--years", type=float, default=3.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--policy", default="developmental")
    for name in ("run", "demo-run"):
        rp = sub.add_parser(name)
        rp.add_argument("--agents", type=int, default=100)
        rp.add_argument("--years", type=float, default=80.0)
        rp.add_argument("--dt-days", type=int, default=30)
        rp.add_argument("--seed", type=int, default=42)
        rp.add_argument("--policy", default="developmental")
        rp.add_argument("--frames-every", type=int, default=1)
        rp.add_argument("--out", default=None if name == "demo-run" else "data/runs")
        rp.add_argument("--run-id", default=None)
        rp.add_argument("--label", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "info":
        from hpe import __version__
        from hpe.contracts import CONTRACT_VERSION
        from hpe.contracts.registry import registered

        print(f"hpe {__version__} contract {CONTRACT_VERSION} policies={registered()}")
        return 0
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "demo-run":
        return cmd_demo_run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
