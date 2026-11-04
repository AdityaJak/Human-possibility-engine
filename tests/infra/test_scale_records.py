"""Every committed benchmark record must validate against the BenchmarkRecord contract and disclose
hardware and concurrent load (so no performance number is reported without its conditions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hpe.contracts.api import BenchmarkRecord

SCALE = Path(__file__).resolve().parents[2] / "reports" / "scale"
RECORDS = sorted(p for p in SCALE.glob("*.json")) if SCALE.exists() else []


@pytest.mark.parametrize("path", RECORDS, ids=[p.stem for p in RECORDS])
def test_benchmark_record_is_valid_and_disclosed(path: Path) -> None:
    rec = BenchmarkRecord.model_validate_json(path.read_text())
    assert rec.benchmark_id == path.stem
    assert "logical CPUs" in rec.hardware
    assert "load average" in rec.notes
    if rec.status == "completed":
        assert rec.wall_time_s > 0 and rec.agent_years_per_s > 0 and rec.events > 0
