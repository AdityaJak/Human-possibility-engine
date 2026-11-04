"""Run orchestration: background jobs, interactive worlds, persisted and stored runs.

The real society engine is resolved lazily (``hpe.world.World``) so this service only depends on
the frozen ``WorldRunner`` protocol. Tests inject a fake world through ``world_factory``.

Every live run tees its events to (1) the stored-run directory layout (JSONL.gz), (2) the
namespaced SQLite tables used for querying, and (3) an in-memory tick/frame index for dashboards.
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hpe.contracts.api import RunControl, RunCreate, RunInfo, StepRequest
from hpe.contracts.protocols import EventSink, WorldRunner
from hpe.contracts.registry import get_factory
from hpe.contracts.runtime import PolicyFactory
from hpe.contracts.schemas import (
    AgentFrame,
    AgentState,
    JobStatus,
    RunManifest,
    SimulationConfig,
    SocialEdge,
    WorldMap,
    WorldTick,
)
from hpe.storage import (
    CheckpointRepository,
    Database,
    EventRepository,
    JsonlGzSink,
    RunDirectory,
    RunRepository,
    SqliteSink,
    StoredRun,
    StoredRunLoader,
    TeeSink,
    TickFrameIndex,
)
from hpe_api.errors import BadRequest, Conflict, NotFound, Unavailable
from hpe_api.settings import Settings

log = logging.getLogger("hpe_api.runs")

WorldFactory = Callable[[SimulationConfig, PolicyFactory, EventSink, str], WorldRunner]
PolicyResolver = Callable[[str, dict[str, Any]], PolicyFactory]


def default_world_factory(
    config: SimulationConfig, policy_factory: PolicyFactory, sink: EventSink, run_id: str
) -> WorldRunner:
    try:
        mod = importlib.import_module("hpe.world")
    except ImportError as e:  # pragma: no cover - depends on build
        raise Unavailable(f"world engine not available in this build: {e}") from e
    world_cls = getattr(mod, "World", None)
    if world_cls is None:
        raise Unavailable("world engine not available in this build (hpe.world.World missing)")
    world: WorldRunner = world_cls(config, policy_factory, sink=sink, run_id=run_id)
    return world


def world_engine_available() -> bool:
    try:
        mod = importlib.import_module("hpe.world")
    except ImportError:
        return False
    return getattr(mod, "World", None) is not None


def default_policy_resolver(policy_class: str, params: dict[str, Any]) -> PolicyFactory:
    try:
        return get_factory(policy_class, params)
    except (KeyError, ImportError) as e:
        raise BadRequest(f"policy class {policy_class!r} is not available in this build: {e}") from e


# ---------------------------------------------------------------------------------------
# Views: a uniform read interface over live and stored runs
# ---------------------------------------------------------------------------------------


class RunView(Protocol):
    run_id: str

    def dt_days(self) -> int: ...
    def agents(self) -> list[AgentState]: ...
    def agent(self, agent_id: str) -> AgentState: ...
    def edges(self, agent_id: str | None = None) -> list[SocialEdge]: ...
    def world_map(self) -> WorldMap: ...
    def ticks(self) -> list[WorldTick]: ...
    def frame_ticks(self) -> list[int]: ...
    def frames_at(self, tick: int) -> list[AgentFrame]: ...
    def current_frames(self) -> tuple[int, list[AgentFrame]] | None: ...
    def recent_memories(self, agent_id: str) -> list[dict[str, Any]] | None: ...


def _filter_edges(edges: list[SocialEdge], agent_id: str | None) -> list[SocialEdge]:
    if agent_id is None:
        return edges
    return [e for e in edges if e.src == agent_id or e.dst == agent_id]


@dataclass
class LiveRun:
    run_id: str
    mode: str
    config: SimulationConfig
    world: WorldRunner
    run_dir: Path
    jsonl: JsonlGzSink
    sqlite: SqliteSink
    index: TickFrameIndex
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    resume_event: threading.Event = field(default_factory=threading.Event)
    cancel_flag: bool = False
    pause_flag: bool = False
    future: Future[None] | None = None
    advancer: threading.Thread | None = None
    checkpoints: list[str] = field(default_factory=list)
    next_checkpoint_year: float | None = None
    manifest: RunManifest | None = None
    finalized: bool = False
    started_at: float = field(default_factory=time.monotonic)

    @property
    def sink(self) -> TeeSink:
        return TeeSink([self.jsonl, self.sqlite, self.index])

    def progress(self) -> float:
        try:
            if self.status == JobStatus.COMPLETED or self.world.done:
                return 1.0
            return float(min(1.0, max(0.0, self.world.year / max(self.config.years, 1e-9))))
        except Exception:  # pragma: no cover - defensive against partial worlds
            return 0.0


class LiveRunView:
    def __init__(self, run: LiveRun) -> None:
        self.run = run
        self.run_id = run.run_id

    def dt_days(self) -> int:
        return self.run.config.dt_days

    def agents(self) -> list[AgentState]:
        with self.run.lock:
            return list(self.run.world.agents(include_dead=True))

    def agent(self, agent_id: str) -> AgentState:
        with self.run.lock:
            try:
                return self.run.world.agent(agent_id)
            except (KeyError, IndexError, ValueError) as e:
                raise NotFound(f"agent {agent_id!r} not found in run {self.run_id}") from e

    def edges(self, agent_id: str | None = None) -> list[SocialEdge]:
        with self.run.lock:
            return _filter_edges(list(self.run.world.edges(agent_id)), agent_id)

    def world_map(self) -> WorldMap:
        with self.run.lock:
            return self.run.world.world_map()

    def ticks(self) -> list[WorldTick]:
        return self.run.index.ticks()

    def frame_ticks(self) -> list[int]:
        return self.run.index.frame_ticks()

    def frames_at(self, tick: int) -> list[AgentFrame]:
        return self.run.index.frames_at(tick)

    def current_frames(self) -> tuple[int, list[AgentFrame]] | None:
        with self.run.lock:
            return self.run.world.tick, list(self.run.world.frames())

    def recent_memories(self, agent_id: str) -> list[dict[str, Any]] | None:
        with self.run.lock:
            try:
                summary = self.run.world.policy_of(agent_id).parameter_summary()
            except Exception:
                return None
        mem = summary.get("recent_memories") if isinstance(summary, dict) else None
        return list(mem) if isinstance(mem, list) else None


class StoredRunView:
    def __init__(self, stored: StoredRun) -> None:
        self.stored = stored
        self.run_id = stored.run_id

    def dt_days(self) -> int:
        return self.stored.manifest.config.dt_days

    def agents(self) -> list[AgentState]:
        return self.stored.agents()

    def agent(self, agent_id: str) -> AgentState:
        for a in self.stored.agents():
            if a.agent_id == agent_id:
                return a
        raise NotFound(f"agent {agent_id!r} not found in run {self.run_id}")

    def edges(self, agent_id: str | None = None) -> list[SocialEdge]:
        return _filter_edges(self.stored.edges(), agent_id)

    def world_map(self) -> WorldMap:
        return self.stored.world_map()

    def ticks(self) -> list[WorldTick]:
        return self.stored.ticks()

    def frame_ticks(self) -> list[int]:
        return self.stored.frame_ticks()

    def frames_at(self, tick: int) -> list[AgentFrame]:
        return self.stored.frames_at(tick)

    def current_frames(self) -> tuple[int, list[AgentFrame]] | None:
        ticks = self.stored.frame_ticks()
        return (ticks[-1], self.stored.frames_at(ticks[-1])) if ticks else None

    def recent_memories(self, agent_id: str) -> list[dict[str, Any]] | None:
        return None  # derived from POLICY_UPDATE events by the assembler


# ---------------------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------------------


class RunService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        loader: StoredRunLoader,
        *,
        world_factory: WorldFactory | None = None,
        policy_resolver: PolicyResolver | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.loader = loader
        self.world_factory = world_factory or default_world_factory
        self.policy_resolver = policy_resolver or default_policy_resolver
        self.runs_repo = RunRepository(db)
        self.events_repo = EventRepository(db)
        self.checkpoints_repo = CheckpointRepository(db)
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs, thread_name_prefix="hpe-run"
        )
        self._live: dict[str, LiveRun] = {}
        self._lock = threading.RLock()
        self._import_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------ creation
    def _check_quota(self, config: SimulationConfig, mode: str) -> None:
        if config.n_agents > self.settings.max_agents_per_run:
            raise BadRequest(
                f"n_agents={config.n_agents} exceeds HPE_MAX_AGENTS_PER_RUN={self.settings.max_agents_per_run}"
            )
        if config.years > self.settings.max_years_per_run:
            raise BadRequest(
                f"years={config.years} exceeds HPE_MAX_YEARS_PER_RUN={self.settings.max_years_per_run}"
            )
        if mode == "interactive":
            active = [
                r
                for r in self._live.values()
                if r.mode == "interactive" and r.status in (JobStatus.PAUSED, JobStatus.RUNNING)
            ]
            if len(active) >= self.settings.max_interactive_runs:
                raise Conflict(
                    f"too many interactive runs in memory ({len(active)}); cancel one or raise HPE_MAX_INTERACTIVE_RUNS"
                )

    def _build(self, run_id: str, config: SimulationConfig, mode: str) -> LiveRun:
        policy_factory = self.policy_resolver(str(config.policy_class.value), dict(config.policy_params))
        run_dir = self.settings.runs_dir / run_id
        jsonl = JsonlGzSink(run_dir)
        sqlite = SqliteSink(self.db, run_id)
        index = TickFrameIndex()
        tee = TeeSink([jsonl, sqlite, index])
        try:
            world = self.world_factory(config, policy_factory, tee, run_id)
        except Exception:
            jsonl.close()
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        run = LiveRun(
            run_id=run_id,
            mode=mode,
            config=config,
            world=world,
            run_dir=run_dir,
            jsonl=jsonl,
            sqlite=sqlite,
            index=index,
        )
        ck = config.log.checkpoint_every_years
        run.next_checkpoint_year = (
            ck if (ck and self.settings.save_checkpoints and mode == "background") else None
        )
        return run

    def create(self, req: RunCreate) -> RunInfo:
        self._check_quota(req.config, req.mode)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        run = self._build(run_id, req.config, req.mode)
        run.status = JobStatus.QUEUED if req.mode == "background" else JobStatus.PAUSED
        with self._lock:
            self._live[run_id] = run
        self.runs_repo.upsert(
            run_id,
            status=run.status.value,
            mode=req.mode,
            source="live",
            config_json=req.config.model_dump_json(),
            run_dir=str(run.run_dir),
        )
        if req.mode == "background":
            run.future = self.executor.submit(self._background_job, run)
        return self.info(run_id)

    # ------------------------------------------------------------------ execution
    def _record_tick(self, run: LiveRun, tick: WorldTick) -> None:
        """Make sure the returned tick reaches every sink exactly once (worlds may or may not emit it)."""
        if not run.index.has_tick(tick.tick):
            run.sink.emit_tick(tick)
        elif tick.frames and tick.tick not in run.index.frame_ticks():
            run.index.add_frames(tick.tick, tick.frames)

    def _maybe_checkpoint(self, run: LiveRun) -> None:
        if run.next_checkpoint_year is None or run.world.year + 1e-9 < run.next_checkpoint_year:
            return
        try:
            snap = run.world.snapshot()
            path, sha = RunDirectory(run.run_dir).checkpoints().save(run.world.tick, snap)
            rel = str(path.relative_to(run.run_dir))
            run.checkpoints.append(rel)
            self.checkpoints_repo.add(run.run_id, run.world.tick, rel, sha)
        except Exception as e:  # checkpoint failure must not kill the run, but is recorded
            log.warning("checkpoint failed for %s: %s", run.run_id, e)
        step = run.config.log.checkpoint_every_years or 0.0
        while (
            run.next_checkpoint_year is not None
            and run.next_checkpoint_year <= run.world.year + 1e-9
            and step > 0
        ):
            run.next_checkpoint_year += step

    def _set_status(self, run: LiveRun, status: JobStatus, error: str | None = None) -> None:
        run.status = status
        run.error = error
        self.runs_repo.set_status(run.run_id, status.value, error)

    def _background_job(self, run: LiveRun) -> None:
        self._set_status(run, JobStatus.RUNNING)
        try:
            while True:
                if run.cancel_flag:
                    self._finalize(run, JobStatus.CANCELLED)
                    return
                if run.pause_flag:
                    if run.status != JobStatus.PAUSED:
                        self._set_status(run, JobStatus.PAUSED)
                    run.resume_event.wait(timeout=0.2)
                    continue
                if run.status != JobStatus.RUNNING:
                    self._set_status(run, JobStatus.RUNNING)
                with run.lock:
                    if run.world.done:
                        break
                    tick = run.world.step(self.settings.background_chunk_ticks, include_frames=False)
                    self._record_tick(run, tick)
                    self._maybe_checkpoint(run)
            self._finalize(run, JobStatus.COMPLETED)
        except Exception as e:
            log.exception("run %s failed", run.run_id)
            self._finalize(run, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")

    def _advance_loop(self, run: LiveRun) -> None:
        try:
            while not run.pause_flag and not run.cancel_flag:
                with run.lock:
                    if run.world.done:
                        break
                    tick = run.world.step(1, include_frames=True)
                    self._record_tick(run, tick)
                if self.settings.interactive_tick_delay_s > 0:
                    time.sleep(self.settings.interactive_tick_delay_s)  # pacing only, never affects results
            with run.lock:
                if run.cancel_flag:
                    self._finalize(run, JobStatus.CANCELLED)
                elif run.world.done:
                    self._finalize(run, JobStatus.COMPLETED)
                else:
                    self._set_status(run, JobStatus.PAUSED)
        except Exception as e:
            log.exception("interactive run %s failed", run.run_id)
            self._finalize(run, JobStatus.FAILED, error=f"{type(e).__name__}: {e}")

    def _finalize(self, run: LiveRun, status: JobStatus, error: str | None = None) -> None:
        with run.lock:
            if run.finalized:
                self._set_status(run, status, error or run.error)
                return
            run.finalized = True
            try:
                run.jsonl.close()
                run.sqlite.close()
            except Exception as e:  # pragma: no cover
                log.warning("closing sinks failed for %s: %s", run.run_id, e)
            rd = RunDirectory(run.run_dir)
            world = run.world
            for name, fn in (
                ("agents", lambda: rd.write_agents(world.agents(include_dead=True))),
                ("edges", lambda: rd.write_edges(world.edges(None))),
                ("map", lambda: rd.write_map(world.world_map())),
                ("outcomes", lambda: rd.write_outcomes(world.outcomes())),
            ):
                try:
                    fn()
                except Exception as e:
                    log.warning("writing %s failed for %s: %s", name, run.run_id, e)
            try:
                manifest = world.manifest()
            except Exception as e:
                log.warning("world.manifest() failed for %s: %s", run.run_id, e)
                from hpe.core.hashing import stable_hash

                manifest = RunManifest(
                    run_id=run.run_id,
                    config=run.config,
                    config_hash=stable_hash(run.config.model_dump(mode="json")),
                    policy_class=run.config.policy_class,
                )
            manifest = manifest.model_copy(
                update={
                    "run_id": run.run_id,
                    "status": status,
                    "event_log_path": "events.jsonl.gz",
                    "event_log_sha256": run.jsonl.sha256,
                    "n_events": run.jsonl.n_events,
                    "checkpoints": list(run.checkpoints) or list(manifest.checkpoints),
                    "error": error,
                }
            )
            rd.write_manifest(manifest)
            run.manifest = manifest
            self.runs_repo.set_manifest(run.run_id, manifest)
            self._set_status(run, status, error)

    # ------------------------------------------------------------------ queries
    def _get_live(self, run_id: str) -> LiveRun | None:
        with self._lock:
            return self._live.get(run_id)

    def _persisted_dir(self, run_id: str) -> Path | None:
        p = self.settings.runs_dir / run_id
        return p if (p / "manifest.json").exists() else None

    def info(self, run_id: str) -> RunInfo:
        run = self._get_live(run_id)
        if run is not None:
            with run.lock:
                manifest = run.manifest
                if manifest is None:
                    try:
                        manifest = run.world.manifest()
                    except Exception:
                        manifest = None
                return RunInfo(
                    run_id=run_id,
                    status=run.status,
                    mode=run.mode,  # type: ignore[arg-type]
                    source="live",
                    progress=run.progress(),
                    current_tick=int(run.world.tick),
                    current_year=float(run.world.year),
                    manifest=manifest,
                    latest_tick=run.index.latest(),
                )
        stored = self.loader.get(run_id)
        if stored is not None:
            return self._stored_info(stored, "stored_real_output")
        pdir = self._persisted_dir(run_id)
        if pdir is not None:
            return self._stored_info(StoredRun(pdir), "live")
        raise NotFound(f"run {run_id!r} not found")

    @staticmethod
    def _stored_info(stored: StoredRun, source: str) -> RunInfo:
        ticks = stored.ticks()
        m = stored.manifest
        return RunInfo(
            run_id=stored.run_id,
            status=m.status,
            mode="stored",
            source=source,  # type: ignore[arg-type]
            progress=1.0 if m.status == JobStatus.COMPLETED else 0.0,
            current_tick=ticks[-1].tick if ticks else 0,
            current_year=ticks[-1].year if ticks else 0.0,
            manifest=m,
            latest_tick=ticks[-1] if ticks else None,
        )

    def list_runs(self, *, include_experiment_runs: bool = False) -> list[RunInfo]:
        out: list[RunInfo] = []
        seen: set[str] = set()
        with self._lock:
            live_ids = list(self._live)
        for rid in live_ids:
            out.append(self.info(rid))
            seen.add(rid)
        for row in self.runs_repo.list():
            rid = row["run_id"]
            if rid in seen or row["source"] != "live":
                continue
            if self._persisted_dir(rid) is not None:
                out.append(self.info(rid))
                seen.add(rid)
        for rid in self.loader.run_ids(include_experiment_runs=include_experiment_runs):
            if rid not in seen:
                out.append(self.info(rid))
                seen.add(rid)
        return out

    def view(self, run_id: str) -> RunView:
        run = self._get_live(run_id)
        if run is not None:
            if not run.finalized:
                run.sqlite.flush()  # make buffered events visible to queries
            return LiveRunView(run)
        stored = self.loader.get(run_id)
        source = "stored_real_output"
        if stored is None:
            pdir = self._persisted_dir(run_id)
            if pdir is None:
                raise NotFound(f"run {run_id!r} not found")
            stored = StoredRun(pdir)
            source = "live"
        self._ensure_imported(stored, source)
        return StoredRunView(stored)

    def _ensure_imported(self, stored: StoredRun, source: str) -> None:
        """Stored runs are queried through SQLite: import their event log once (verified by sha256)."""
        rid = stored.run_id
        with self._lock:
            lock = self._import_locks.setdefault(rid, threading.Lock())
        with lock:
            expected = stored.manifest.event_log_sha256 or stored.events_sha256()
            if self.runs_repo.get(rid) is None:
                self.runs_repo.upsert(
                    rid,
                    status=stored.manifest.status.value,
                    mode="stored",
                    source=source,
                    config_json=stored.manifest.config.model_dump_json(),
                    manifest=stored.manifest,
                    run_dir=str(stored.path),
                )
            if self.runs_repo.imported_sha(rid) == expected:
                return
            if (
                source == "live"
                and self.events_repo.count(rid) == stored.manifest.n_events
                and stored.manifest.n_events
            ):
                self.runs_repo.mark_imported(rid, expected)
                return
            self.db.execute("DELETE FROM sim_events WHERE run_id=?", (rid,))
            batch: list[tuple[Any, ...]] = []
            for line in stored.iter_event_json():
                d = json.loads(line)
                batch.append(
                    (
                        rid,
                        int(d["seq"]),
                        int(d["tick"]),
                        int(d.get("day", 0)),
                        str(d["type"]),
                        d.get("agent_id"),
                        json.dumps(d.get("other_ids", []), separators=(",", ":"), sort_keys=True),
                        line,
                    )
                )
                if len(batch) >= 5000:
                    self.events_repo.append_rows(batch)
                    batch = []
            self.events_repo.append_rows(batch)
            self.runs_repo.mark_imported(rid, expected)

    def is_live(self, run_id: str) -> bool:
        return self._get_live(run_id) is not None

    def live_run_ids(self) -> list[str]:
        with self._lock:
            return list(reversed(list(self._live)))

    # ------------------------------------------------------------------ interactive control
    def step(self, run_id: str, req: StepRequest) -> WorldTick:
        run = self._get_live(run_id)
        if run is None:
            raise NotFound(f"live run {run_id!r} not found (stored runs cannot be stepped)")
        if run.mode != "interactive":
            raise Conflict("only interactive runs can be stepped; background runs advance on their own")
        with run.lock:
            if run.status == JobStatus.RUNNING:
                raise Conflict("run is auto-advancing; pause it before stepping")
            if run.status in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED) or run.world.done:
                raise Conflict(f"run is {run.status.value}; reset it to start again")
            tick = run.world.step(req.ticks, include_frames=req.include_frames)
            self._record_tick(run, tick)
            if run.world.done:
                self._finalize(run, JobStatus.COMPLETED)
            return tick

    def control(self, run_id: str, ctl: RunControl) -> RunInfo:
        run = self._get_live(run_id)
        if run is None:
            raise NotFound(f"live run {run_id!r} not found (stored runs cannot be controlled)")
        action = ctl.action
        if action == "pause":
            if run.status not in (JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.PAUSED):
                raise Conflict(f"cannot pause a {run.status.value} run")
            run.pause_flag = True
            run.resume_event.clear()
            if run.mode == "interactive" and run.advancer is not None:
                run.advancer.join(timeout=5)
                run.advancer = None
            elif run.mode == "background":
                self._set_status(run, JobStatus.PAUSED)
        elif action == "resume":
            if run.status in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED):
                raise Conflict(f"cannot resume a {run.status.value} run")
            run.pause_flag = False
            run.resume_event.set()
            if run.mode == "interactive" and (run.advancer is None or not run.advancer.is_alive()):
                self._set_status(run, JobStatus.RUNNING)
                run.advancer = threading.Thread(
                    target=self._advance_loop, args=(run,), name=f"hpe-adv-{run_id}", daemon=True
                )
                run.advancer.start()
        elif action == "cancel":
            run.cancel_flag = True
            run.resume_event.set()
            if run.mode == "interactive":
                if run.advancer is not None:
                    run.advancer.join(timeout=5)
                    run.advancer = None
                self._finalize(run, JobStatus.CANCELLED)
            elif run.future is not None:
                run.future.result(timeout=30)
        elif action == "reset":
            if run.mode != "interactive":
                raise Conflict(
                    "reset is only supported for interactive runs; create a new background run instead"
                )
            run.pause_flag = True
            if run.advancer is not None:
                run.advancer.join(timeout=5)
            with run.lock:
                try:
                    run.jsonl.close()
                    run.sqlite.close()
                except Exception:  # pragma: no cover
                    pass
                self.runs_repo.delete_run_data(run_id)
                shutil.rmtree(run.run_dir, ignore_errors=True)
                fresh = self._build(run_id, run.config, "interactive")
                fresh.status = JobStatus.PAUSED
                with self._lock:
                    self._live[run_id] = fresh
                self._set_status(fresh, JobStatus.PAUSED)
        return self.info(run_id)

    # ------------------------------------------------------------------ download
    def download_lines(self, run_id: str) -> Iterator[str]:
        """Manifest line followed by every event in replay order (JSONL)."""
        run = self._get_live(run_id)
        if run is not None:
            run.sqlite.flush()
            info = self.info(run_id)
            header = {
                "type": "manifest",
                "manifest": info.manifest.model_dump(mode="json") if info.manifest else None,
            }
            yield json.dumps(header, separators=(",", ":"))
            yield from self.events_repo.iter_all(run_id)
            return
        self.view(run_id)  # resolves + imports, or raises NotFound
        stored = self.loader.get(run_id) or StoredRun(self.settings.runs_dir / run_id)
        yield json.dumps(
            {"type": "manifest", "manifest": stored.manifest.model_dump(mode="json")}, separators=(",", ":")
        )
        yield from stored.iter_event_json()

    # ------------------------------------------------------------------ lifecycle
    def shutdown(self) -> None:
        with self._lock:
            runs = list(self._live.values())
        for run in runs:
            run.cancel_flag = True
            run.pause_flag = True
            run.resume_event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        for run in runs:
            if run.advancer is not None:
                run.advancer.join(timeout=5)
            if not run.finalized:  # interactive worlds left in memory: flush partial logs to disk
                with run.lock:
                    try:
                        run.jsonl.close()
                        run.sqlite.close()
                    except Exception:  # pragma: no cover
                        pass
