"""EventSink implementations (``hpe.contracts.protocols.EventSink``).

* ``JsonlGzSink``  — writes the stored-run layout of docs/INTERFACES.md §5 (events/ticks/frames).
  ``sha256`` is computed over the *uncompressed* event JSONL bytes, so a third party can verify
  with ``gunzip -c events.jsonl.gz | sha256sum``. Gzip headers use ``mtime=0`` so identical runs
  produce byte-identical files.
* ``SqliteSink``   — batched inserts into the namespaced ``sim_events`` / ``sim_ticks`` tables.
* ``TickFrameIndex`` — in-memory tick series + compact per-tick frames for live dashboards.
* ``TeeSink``      — fan-out to several sinks.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from hpe.contracts.protocols import EventSink
from hpe.contracts.schemas import AgentFrame, ExperienceEvent, WorldTick
from hpe.storage.database import Database
from hpe.storage.repositories import EventRepository, TickRepository

EVENTS_FILE = "events.jsonl.gz"
TICKS_FILE = "ticks.jsonl.gz"
FRAMES_FILE = "frames.jsonl.gz"


class _GzWriter:
    def __init__(self, path: Path) -> None:
        self._raw = open(path, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)

    def write(self, data: bytes) -> None:
        self._gz.write(data)

    def flush(self) -> None:
        self._gz.flush()
        self._raw.flush()

    def close(self) -> None:
        self._gz.close()
        self._raw.close()


class JsonlGzSink:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events = _GzWriter(self.run_dir / EVENTS_FILE)
        self._ticks = _GzWriter(self.run_dir / TICKS_FILE)
        self._frames = _GzWriter(self.run_dir / FRAMES_FILE)
        self._sha = hashlib.sha256()
        self.n_events = 0
        self.n_ticks = 0
        self.n_frame_sets = 0
        self._closed = False
        self._lock = threading.Lock()

    @property
    def events_path(self) -> Path:
        return self.run_dir / EVENTS_FILE

    @property
    def sha256(self) -> str:
        return self._sha.hexdigest()

    def emit(self, event: ExperienceEvent) -> None:
        line = event.model_dump_json().encode("utf-8") + b"\n"
        with self._lock:
            self._sha.update(line)
            self._events.write(line)
            self.n_events += 1

    def emit_tick(self, tick: WorldTick) -> None:
        with self._lock:
            if tick.frames:
                payload = {"tick": tick.tick, "frames": [f.model_dump(mode="json") for f in tick.frames]}
                self._frames.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
                self.n_frame_sets += 1
            self._ticks.write(
                tick.model_copy(update={"frames": None}).model_dump_json().encode("utf-8") + b"\n"
            )
            self.n_ticks += 1

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                for w in (self._events, self._ticks, self._frames):
                    w.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for w in (self._events, self._ticks, self._frames):
                w.close()
            self._closed = True


class SqliteSink:
    def __init__(self, db: Database, run_id: str, batch_size: int = 2000) -> None:
        self.db = db
        self.run_id = run_id
        self.batch_size = batch_size
        self._events = EventRepository(db)
        self._ticks = TickRepository(db)
        self._buf: list[tuple[Any, ...]] = []
        self._tick_buf: list[WorldTick] = []
        self._lock = threading.Lock()
        self.n_events = 0

    def emit(self, event: ExperienceEvent) -> None:
        with self._lock:
            self._buf.append(EventRepository.row(event, self.run_id))
            self.n_events += 1
            if len(self._buf) >= self.batch_size:
                self._flush_locked()

    def emit_tick(self, tick: WorldTick) -> None:
        with self._lock:
            self._tick_buf.append(tick)
            if len(self._tick_buf) >= 64:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf:
            self._events.append_rows(self._buf)
            self._buf = []
        if self._tick_buf:
            self._ticks.append_many(self.run_id, self._tick_buf)
            self._tick_buf = []

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()


class TickFrameIndex:
    """Keeps the tick series and compact frames (JSON text per tick) for fast dashboard reads."""

    def __init__(self, max_frame_sets: int = 5000) -> None:
        self._ticks: dict[int, WorldTick] = {}
        self._frames: dict[int, str] = {}
        self.max_frame_sets = max_frame_sets
        self._lock = threading.Lock()

    def emit(self, event: ExperienceEvent) -> None:
        return None

    def emit_tick(self, tick: WorldTick) -> None:
        with self._lock:
            if tick.frames:
                self._frames[tick.tick] = json.dumps(
                    [f.model_dump(mode="json") for f in tick.frames], separators=(",", ":")
                )
                if len(self._frames) > self.max_frame_sets:
                    # thin out: keep every other stored frame set (oldest first)
                    keys = sorted(self._frames)
                    for k in keys[1 : len(keys) - 1 : 2]:
                        del self._frames[k]
            self._ticks[tick.tick] = tick.model_copy(update={"frames": None})

    def add_frames(self, tick: int, frames: Sequence[AgentFrame]) -> None:
        with self._lock:
            self._frames[tick] = json.dumps(
                [f.model_dump(mode="json") for f in frames], separators=(",", ":")
            )

    def has_tick(self, tick: int) -> bool:
        return tick in self._ticks

    def ticks(self) -> list[WorldTick]:
        with self._lock:
            return [self._ticks[k] for k in sorted(self._ticks)]

    def latest(self) -> WorldTick | None:
        with self._lock:
            return self._ticks[max(self._ticks)] if self._ticks else None

    def frame_ticks(self) -> list[int]:
        with self._lock:
            return sorted(self._frames)

    def frames_at(self, tick: int) -> list[AgentFrame]:
        with self._lock:
            raw = self._frames.get(tick)
        return [AgentFrame.model_validate(x) for x in json.loads(raw)] if raw else []

    def clear(self) -> None:
        with self._lock:
            self._ticks.clear()
            self._frames.clear()

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class TeeSink:
    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self.sinks = list(sinks)

    def emit(self, event: ExperienceEvent) -> None:
        for s in self.sinks:
            s.emit(event)

    def emit_tick(self, tick: WorldTick) -> None:
        for s in self.sinks:
            s.emit_tick(tick)

    def flush(self) -> None:
        for s in self.sinks:
            s.flush()

    def close(self) -> None:
        for s in self.sinks:
            s.close()


def sha256_of_gz_lines(path: Path) -> str:
    """sha256 of the uncompressed content of a .gz file (verification helper)."""
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gzip_bytes(lines: Iterable[str]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        for line in lines:
            gz.write(line.encode("utf-8") + b"\n")
    return buf.getvalue()
