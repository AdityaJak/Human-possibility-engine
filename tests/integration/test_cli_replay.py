"""Gate 1 evidence at small scale: a stored run replays bit-identically from seed and from checkpoint."""

import json

import pytest

from hpe.cli import main, verify_replay

pytestmark = pytest.mark.integration


def test_stored_run_replays_from_seed_and_checkpoint(tmp_path):
    rc = main(
        [
            "run",
            "--agents",
            "15",
            "--years",
            "3",
            "--seed",
            "5",
            "--policy",
            "developmental",
            "--out",
            str(tmp_path),
            "--run-id",
            "r",
            "--checkpoint-every",
            "1.5",
            "--frames-every",
            "6",
        ]
    )
    assert rc == 0
    run_dir = tmp_path / "r"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["checkpoints"], "checkpoints must be recorded in the manifest"
    assert manifest["final_state_hash"]
    for name in ("events.jsonl.gz", "ticks.jsonl.gz", "frames.jsonl.gz", "final_agents.json", "map.json"):
        assert (run_dir / name).exists(), name
    report = verify_replay(run_dir)
    assert report["from_seed"]["chain_identical"] and report["from_seed"]["state_identical"]
    assert report["from_checkpoint"] and report["from_checkpoint"]["chain_identical"]
    assert report["passed"]


def test_smoke_command(tmp_path):
    assert main(["smoke", "--out", str(tmp_path), "--agents", "10", "--years", "1"]) == 0
