"""Stored-run directories (docs/INTERFACES.md §5), checkpoints, and the stored-run loader.

Stored runs under ``reports/runs/<run_id>/`` are real outputs of this codebase, committed so the
demo works offline. They are always served with ``source="stored_real_output"``; nothing in this
module fabricates or edits their contents.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hpe.contracts.schemas import (
    AgentFrame,
    AgentState,
    OutcomeObservation,
    RunManifest,
    SocialEdge,
    WorldMap,
    WorldTick,
)
from hpe.storage.sinks import EVENTS_FILE, FRAMES_FILE, TICKS_FILE, gzip_bytes

MANIFEST_FILE = "manifest.json"
AGENTS_FILE = "final_agents.json"
EDGES_FILE = "edges.json"
MAP_FILE = "map.json"
OUTCOMES_FILE = "outcomes.jsonl.gz"
CHECKPOINT_DIR = "checkpoints"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if isinstance(o, set | frozenset | tuple):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o)!r}")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, default=_json_default, indent=indent, separators=None if indent else (",", ":"))


class RunDirectory:
    """Writer for one run's directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.write_text(dumps(obj, indent=2) + "\n")
        return p

    def write_manifest(self, manifest: RunManifest) -> Path:
        p = self.path / MANIFEST_FILE
        p.write_text(manifest.model_dump_json(indent=2) + "\n")
        return p

    def write_agents(self, agents: Sequence[AgentState]) -> Path:
        return self.write_json(AGENTS_FILE, [a.model_dump(mode="json") for a in agents])

    def write_edges(self, edges: Sequence[SocialEdge]) -> Path:
        return self.write_json(EDGES_FILE, [e.model_dump(mode="json") for e in edges])

    def write_map(self, world_map: WorldMap) -> Path:
        return self.write_json(MAP_FILE, world_map.model_dump(mode="json"))

    def write_outcomes(self, outcomes: Sequence[OutcomeObservation]) -> Path:
        p = self.path / OUTCOMES_FILE
        p.write_bytes(gzip_bytes(o.model_dump_json() for o in outcomes))
        return p

    def checkpoints(self) -> CheckpointStore:
        return CheckpointStore(self.path / CHECKPOINT_DIR)


class CheckpointStore:
    """gz-compressed JSON snapshots, named by tick. sha256 is over the uncompressed JSON."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def save(self, tick: int, snapshot: dict[str, Any]) -> tuple[Path, str]:
        self.dir.mkdir(parents=True, exist_ok=True)
        raw = dumps(snapshot).encode("utf-8")
        p = self.dir / f"{int(tick):07d}.json.gz"
        with open(p, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(raw)
        return p, hashlib.sha256(raw).hexdigest()

    def ticks(self) -> list[int]:
        if not self.dir.exists():
            return []
        return sorted(int(p.name.split(".")[0]) for p in self.dir.glob("*.json.gz"))

    def load(self, tick: int | None = None) -> dict[str, Any]:
        ticks = self.ticks()
        if not ticks:
            raise FileNotFoundError(f"no checkpoints in {self.dir}")
        t = ticks[-1] if tick is None else int(tick)
        p = self.dir / f"{t:07d}.json.gz"
        with gzip.open(p, "rb") as f:
            data: dict[str, Any] = json.loads(f.read())
        return data


def _iter_gz_lines(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


class StoredRun:
    """Lazy read-only view of a stored run directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = RunManifest.model_validate_json((self.path / MANIFEST_FILE).read_text())
        self.run_id = self.manifest.run_id
        self._lock = threading.Lock()
        self._agents: list[AgentState] | None = None
        self._edges: list[SocialEdge] | None = None
        self._map: WorldMap | None = None
        self._ticks: list[WorldTick] | None = None
        self._frames: dict[int, str] | None = None

    @property
    def events_path(self) -> Path:
        return self.path / EVENTS_FILE

    def events_sha256(self) -> str:
        h = hashlib.sha256()
        if self.events_path.exists():
            with gzip.open(self.events_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        return h.hexdigest()

    def iter_event_json(self) -> Iterator[str]:
        return _iter_gz_lines(self.events_path)

    def agents(self) -> list[AgentState]:
        with self._lock:
            if self._agents is None:
                p = self.path / AGENTS_FILE
                self._agents = (
                    [AgentState.model_validate(a) for a in json.loads(p.read_text())] if p.exists() else []
                )
            return self._agents

    def edges(self) -> list[SocialEdge]:
        with self._lock:
            if self._edges is None:
                p = self.path / EDGES_FILE
                self._edges = (
                    [SocialEdge.model_validate(e) for e in json.loads(p.read_text())] if p.exists() else []
                )
            return self._edges

    def world_map(self) -> WorldMap:
        with self._lock:
            if self._map is None:
                p = self.path / MAP_FILE
                self._map = (
                    WorldMap.model_validate_json(p.read_text())
                    if p.exists()
                    else WorldMap(width=100, height=60, places=[])
                )
            return self._map

    def ticks(self) -> list[WorldTick]:
        with self._lock:
            if self._ticks is None:
                self._ticks = [
                    WorldTick.model_validate_json(x) for x in _iter_gz_lines(self.path / TICKS_FILE)
                ]
            return self._ticks

    def _frame_index(self) -> dict[int, str]:
        with self._lock:
            if self._frames is None:
                idx: dict[int, str] = {}
                for line in _iter_gz_lines(self.path / FRAMES_FILE):
                    d = json.loads(line)
                    idx[int(d["tick"])] = json.dumps(d["frames"], separators=(",", ":"))
                self._frames = idx
            return self._frames

    def frame_ticks(self) -> list[int]:
        return sorted(self._frame_index())

    def frames_at(self, tick: int) -> list[AgentFrame]:
        raw = self._frame_index().get(tick)
        return [AgentFrame.model_validate(x) for x in json.loads(raw)] if raw else []

    def outcomes(self, limit: int | None = None) -> list[OutcomeObservation]:
        out: list[OutcomeObservation] = []
        for line in _iter_gz_lines(self.path / OUTCOMES_FILE):
            out.append(OutcomeObservation.model_validate_json(line))
            if limit is not None and len(out) >= limit:
                break
        return out

    def checkpoints(self) -> CheckpointStore:
        return CheckpointStore(self.path / CHECKPOINT_DIR)


class StoredRunLoader:
    """Indexes stored runs under one or more roots.

    Scans ``<reports>/runs/*/manifest.json`` (demo runs) and, optionally,
    ``<reports>/experiments/*/runs/*/manifest.json`` (experiment arm runs).
    """

    def __init__(self, reports_dir: str | Path, *, include_experiment_runs: bool = True) -> None:
        self.reports_dir = Path(reports_dir)
        self.include_experiment_runs = include_experiment_runs
        self._cache: dict[str, StoredRun] = {}
        self._index: dict[str, tuple[Path, bool]] | None = None
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            self._index = None
            self._cache.clear()

    def _scan(self) -> dict[str, tuple[Path, bool]]:
        with self._lock:
            if self._index is not None:
                return self._index
            idx: dict[str, tuple[Path, bool]] = {}
            patterns: list[tuple[str, bool]] = [("runs/*/manifest.json", False)]
            if self.include_experiment_runs:
                patterns.append(("experiments/*/runs/*/manifest.json", True))
            for pattern, is_exp in patterns:
                for mpath in sorted(self.reports_dir.glob(pattern)):
                    try:
                        manifest = json.loads(mpath.read_text())
                        run_id = str(manifest["run_id"])
                    except (OSError, ValueError, KeyError):
                        continue
                    idx.setdefault(run_id, (mpath.parent, is_exp))
            self._index = idx
            return idx

    def run_ids(self, *, include_experiment_runs: bool = False) -> list[str]:
        return [
            rid for rid, (_, is_exp) in sorted(self._scan().items()) if include_experiment_runs or not is_exp
        ]

    def count_demo_runs(self) -> int:
        return len(self.run_ids(include_experiment_runs=False))

    def has(self, run_id: str) -> bool:
        return run_id in self._scan()

    def get(self, run_id: str) -> StoredRun | None:
        entry = self._scan().get(run_id)
        if entry is None:
            return None
        with self._lock:
            if run_id not in self._cache:
                self._cache[run_id] = StoredRun(entry[0])
            return self._cache[run_id]
