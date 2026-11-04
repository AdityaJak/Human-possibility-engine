"""Research-output exports: JSONL (always available) and Parquet (optional ``pyarrow``)."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def export_jsonl(
    rows: Iterable[dict[str, Any] | str], path: str | Path, *, compress: bool | None = None
) -> int:
    """Write rows (dicts or pre-serialised JSON strings) one per line. Returns row count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    gz = p.suffix == ".gz" if compress is None else compress
    n = 0
    opener = gzip.open if gz else open
    with opener(p, "wt", encoding="utf-8") as f:  # type: ignore[operator]
        for row in rows:
            f.write(
                (row if isinstance(row, str) else json.dumps(row, separators=(",", ":"), default=str)) + "\n"
            )
            n += 1
    return n


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[k] = json.dumps(v, separators=(",", ":"), default=str) if isinstance(v, dict | list) else v
    return out


def export_parquet(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write rows to Parquet; nested values are stored as JSON strings. Raises if pyarrow is missing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = [_flatten(r) for r in rows]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(flat) if flat else pa.table({})
    pq.write_table(table, p)
    return len(flat)


def events_to_rows(event_json: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in event_json:
        d = json.loads(line)
        yield {
            "run_id": d.get("run_id"),
            "seq": d.get("seq"),
            "tick": d.get("tick"),
            "day": d.get("day"),
            "type": d.get("type"),
            "agent_id": d.get("agent_id"),
            "other_ids": d.get("other_ids", []),
            "payload": d.get("payload", {}),
            "provenance": d.get("provenance", {}),
            "namespace": d.get("namespace"),
        }
