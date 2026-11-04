"""The experiment results endpoint must sample outcomes from the runner's on-disk layout
(``runs/<run_id>/outcomes.jsonl.gz``), prefer main-variant runs, and bound the payload."""

import gzip
import json
from pathlib import Path

from hpe_api.experiment_service import _read_outcomes


def _row(run_id: str, arm: str, seed: int, metric: str, value: float, agent_id: str | None = None, **kw):
    return {
        "run_id": run_id,
        "experiment_id": "x",
        "arm": arm,
        "seed": seed,
        "agent_id": agent_id,
        "metric": metric,
        "value": value,
        "split": "holdout",
        "source": "simulation",
        **kw,
    }


def _write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_reads_runs_layout_prefers_main_and_thins_curves(tmp_path: Path):
    exp = tmp_path / "x"
    main = exp / "runs" / "x__main__control__s0" / "outcomes.jsonl.gz"
    sens = exp / "runs" / "x__high_noise__control__s0" / "outcomes.jsonl.gz"
    rows = [_row("x__main__control__s0", "control", 0, "final_heldout_accuracy", 0.8)]
    rows += [_row("x__main__control__s0", "control", 0, "leakage/practice_items", 100.0)]
    rows += [
        _row("x__main__control__s0", "control", 0, "agent_accuracy@20", 0.7, agent_id=f"a{i}")
        for i in range(50)
    ]
    rows += [
        _row("x__main__control__s0", "control", 0, "curve/heldout_accuracy/sort", i / 500, tick=i)
        for i in range(500)
    ]
    _write(main, rows)
    _write(sens, [_row("x__high_noise__control__s0", "control", 0, "final_heldout_accuracy", 0.5)])

    out = _read_outcomes(exp, limit=200)
    run_ids = {o.run_id for o in out}
    assert run_ids == {"x__main__control__s0"}, "sensitivity variants must not be mixed with main runs"
    metrics = {o.metric for o in out}
    assert "final_heldout_accuracy" in metrics and "leakage/practice_items" not in metrics
    assert all(o.agent_id is None for o in out), "per-agent rows are not shipped"
    curves = [o for o in out if o.metric.startswith("curve/")]
    assert 0 < len(curves) <= 200 and curves[0].tick == 0 and curves[-1].tick is not None
    assert [o.value for o in out if o.metric == "final_heldout_accuracy"] == [0.8]


def test_legacy_top_level_file_still_read(tmp_path: Path):
    exp = tmp_path / "y"
    _write(exp / "outcomes.jsonl.gz", [_row("r", "a", 1, "m", 1.0)])
    out = _read_outcomes(exp)
    assert len(out) == 1 and out[0].metric == "m"


def test_missing_dir_returns_empty(tmp_path: Path):
    assert _read_outcomes(tmp_path / "nope") == []
