#!/usr/bin/env python3
"""Scale benchmark: run one society and record a ``BenchmarkRecord`` (contracts.api).

    python scripts/bench_scale.py --agents 100 --years 2 --policy developmental
    python scripts/bench_scale.py --agents 10000 --years 1 --timeout 3600 --profile

The simulation runs in a child process so a hard ``--timeout`` can stop it; a timed-out or
crashed run is still recorded (status ``timeout``/``failed``) — failures are results too.
Records are written to ``reports/scale/<benchmark_id>.json``. Numbers are only ever produced
by actually executing the simulator; the record discloses hardware and concurrent load.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import multiprocessing as mp
import os
import platform
import pstats
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "packages", ROOT / "apps" / "api"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hpe.contracts.api import BenchmarkRecord  # noqa: E402
from hpe.contracts.schemas import ExperienceEvent, LogConfig, PolicyClass, SimulationConfig, WorldTick  # noqa: E402


class CountingSink:
    """Counts events without storing them, so the benchmark measures simulation cost."""

    def __init__(self) -> None:
        self.n_events = 0
        self.n_ticks = 0

    def emit(self, event: ExperienceEvent) -> None:
        self.n_events += 1

    def emit_tick(self, tick: WorldTick) -> None:
        self.n_ticks += 1

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def hardware_string() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem = f", {int(line.split()[1]) / 1024 / 1024:.1f} GiB RAM"
                break
    except OSError:
        pass
    return (
        f"{cpu} x{os.cpu_count()} logical CPUs{mem}; {platform.system()} {platform.release()}; "
        f"python {platform.python_version()}"
    )


def build_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        seed=args.seed,
        n_agents=args.agents,
        years=args.years,
        dt_days=args.dt_days,
        policy_class=PolicyClass(args.policy),
        log=LogConfig(
            frames_every_ticks=args.frames_every,
            decision_sample_rate=0.0,
            assessment_interval_years=args.assessment_interval,
            checkpoint_every_years=None,
        ),
        max_events=None,
        label=f"benchmark {args.agents}a {args.years}y",
    )


def _child(args: argparse.Namespace, conn: Any, profile_path: str | None) -> None:
    try:
        import importlib

        from hpe.contracts.registry import get_factory

        world_cls = importlib.import_module("hpe.world").World
        cfg = build_config(args)
        factory = get_factory(cfg.policy_class.value, cfg.policy_params)
        sink = CountingSink()
        t_build = time.perf_counter()
        world = world_cls(cfg, factory, sink=sink, run_id=f"bench_{args.agents}_{args.years}")
        build_s = time.perf_counter() - t_build
        prof = cProfile.Profile() if profile_path else None
        t0, c0 = time.perf_counter(), time.process_time()
        if prof:
            prof.enable()
        world.run()
        if prof:
            prof.disable()
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        if prof and profile_path:
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(25)
            Path(profile_path).write_text(s.getvalue())
        compute = world.manifest().compute
        conn.send(
            {
                "ok": True,
                "wall_s": wall,
                "cpu_s": cpu,
                "build_s": build_s,
                "ticks": world.tick,
                "agent_years": float(compute.agent_years) if compute else args.agents * args.years,
                "events": sink.n_events,
                "population_end": len(world.agents(include_dead=False)),
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            }
        )
    except BaseException as e:  # report failures as results
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def run_benchmark(args: argparse.Namespace) -> BenchmarkRecord:
    bench_id = args.id or (
        f"{args.policy}_{args.agents}a_{args.years:g}y_dt{args.dt_days}_s{args.seed}"
        f"_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = str(out_dir / f"{bench_id}.profile.txt") if args.profile else None
    load_before = os.getloadavg()
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(args, child, profile_path))
    t0 = time.perf_counter()
    proc.start()
    child.close()
    result: dict[str, Any] | None = None
    if parent.poll(args.timeout):
        try:
            result = parent.recv()
        except EOFError:
            result = None
    elapsed = time.perf_counter() - t0
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
    proc.join()
    load_after = os.getloadavg()
    load_note = (
        f"load average (1/5/15 min) before {load_before[0]:.2f}/{load_before[1]:.2f}/{load_before[2]:.2f}, "
        f"after {load_after[0]:.2f}/{load_after[1]:.2f}/{load_after[2]:.2f}"
    )
    common = {
        "benchmark_id": bench_id,
        "n_agents": args.agents,
        "years": args.years,
        "dt_days": args.dt_days,
        "policy_class": PolicyClass(args.policy),
        "hardware": hardware_string(),
    }
    if result is None:
        status = "timeout" if elapsed >= args.timeout else "failed"
        rec = BenchmarkRecord(
            **common,
            wall_time_s=round(elapsed, 3),
            peak_rss_mb=None,
            agent_years_per_s=0.0,
            ticks_per_s=0.0,
            events=0,
            status=status,
            notes=f"{status} after {elapsed:.1f}s (timeout {args.timeout}s, exit code {proc.exitcode}); {load_note}",
        )
    elif not result["ok"]:
        rec = BenchmarkRecord(
            **common,
            wall_time_s=round(elapsed, 3),
            peak_rss_mb=None,
            agent_years_per_s=0.0,
            ticks_per_s=0.0,
            events=0,
            status="failed",
            notes=f"error: {result['error']}; {load_note}",
        )
    else:
        wall = result["wall_s"]
        rec = BenchmarkRecord(
            **common,
            wall_time_s=round(wall, 3),
            peak_rss_mb=round(result["peak_rss_mb"], 1),
            agent_years_per_s=round(result["agent_years"] / wall, 3) if wall > 0 else 0.0,
            ticks_per_s=round(result["ticks"] / wall, 3) if wall > 0 else 0.0,
            events=int(result["events"]),
            status="completed",
            notes=(
                f"world.run() wall time (excludes {result['build_s']:.2f}s World() construction); "
                f"cpu {result['cpu_s']:.1f}s; {result['ticks']} ticks; {result['agent_years']:.0f} agent-years; "
                f"population at end {result['population_end']}; events counted, not stored; "
                f"frames_every_ticks={args.frames_every}; seed {args.seed}; LLM calls 0, cost $0; {load_note}"
                + (f"; profile: {Path(profile_path).name}" if profile_path else "")
            ),
        )
    path = out_dir / f"{bench_id}.json"
    path.write_text(rec.model_dump_json(indent=2) + "\n")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--dt-days", type=int, default=30)
    ap.add_argument("--policy", default="developmental", choices=[p.value for p in PolicyClass])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600.0, help="hard wall-clock limit in seconds")
    ap.add_argument("--frames-every", type=int, default=0)
    ap.add_argument("--assessment-interval", type=float, default=1.0)
    ap.add_argument(
        "--profile", action="store_true", help="write cProfile top-25 (cumulative) next to the record"
    )
    ap.add_argument("--out", default=str(ROOT / "reports" / "scale"))
    ap.add_argument("--id", default=None, help="benchmark id (default: derived from parameters + UTC time)")
    args = ap.parse_args(argv)
    rec = run_benchmark(args)
    print(rec.model_dump_json(indent=2))
    return 0 if rec.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
