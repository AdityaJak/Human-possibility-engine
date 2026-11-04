"""Freeze records and the conditions under which the Phase III test partition opens (critic finding T-1).

A test partition opens for ``experiment_id`` only if **all** of these hold:

1. ``reports/phase3/freeze/<experiment_id>.json`` (fixed directory) exists with every required field,
   is tracked by git and unmodified relative to HEAD;
2. ``reports/phase3/freeze/<experiment_id>.critic.json`` exists, is committed and unmodified, says
   ``"approved": true`` and names the same preregistration hash (the critic's approval, separate file);
3. the definition being run has the preregistration hash recorded in the freeze record;
4. the recorded ``code_commit`` is an ancestor of HEAD and nothing under ``packages/`` or
   ``experiments/phase3/`` changed between them;
5. the working tree is clean apart from the confirmatory output directory;
6. no access log ``reports/phase3/confirmatory/<experiment_id>/ACCESS.json`` exists yet (one run).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from hpe.evaluation.human.partitions import TestPartitionLocked
from hpe.population.core.families import REPO_ROOT

FREEZE_DIR = REPO_ROOT / "reports" / "phase3" / "freeze"
CONFIRMATORY_DIR = REPO_ROOT / "reports" / "phase3" / "confirmatory"
REQUIRED = (
    "experiment_id",
    "preregistration_hash",
    "code_commit",
    "frozen_at",
    "counted_families",
    "primary_family",
    "comparator_pool",
    "candidates",
    "model_digests",
)


def _git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def _committed_unmodified(path: Path) -> bool:
    rel = str(path.relative_to(REPO_ROOT))
    tracked, _ = _git("ls-files", "--error-unmatch", rel)
    if tracked != 0:
        return False
    clean, _ = _git("diff", "--quiet", "HEAD", "--", rel)
    return clean == 0


def access_log_path(experiment_id: str) -> Path:
    return CONFIRMATORY_DIR / experiment_id / "ACCESS.json"


def validate_freeze(experiment_id: str, *, definition_hash: str | None = None) -> dict[str, Any]:
    """Return the freeze record or raise ``TestPartitionLocked`` naming the first failed condition."""

    def fail(why: str) -> TestPartitionLocked:
        return TestPartitionLocked(f"test partition locked for {experiment_id!r}: {why}")

    rec_path = FREEZE_DIR / f"{experiment_id}.json"
    critic_path = FREEZE_DIR / f"{experiment_id}.critic.json"
    if not rec_path.exists():
        raise fail(f"no freeze record {rec_path}")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise fail(f"freeze record misses {missing}")
    if rec["experiment_id"] != experiment_id:
        raise fail("freeze record names a different experiment")
    if not _committed_unmodified(rec_path):
        raise fail("freeze record is not committed or has uncommitted changes")
    if not critic_path.exists():
        raise fail("no critic approval file")
    critic = json.loads(critic_path.read_text(encoding="utf-8"))
    if not _committed_unmodified(critic_path):
        raise fail("critic approval is not committed or has uncommitted changes")
    if (
        critic.get("approved") is not True
        or critic.get("preregistration_hash") != rec["preregistration_hash"]
    ):
        raise fail("critic approval missing or for a different preregistration")
    if definition_hash is not None and definition_hash != rec["preregistration_hash"]:
        raise fail("the definition's preregistration hash differs from the frozen one")
    anc, _ = _git("merge-base", "--is-ancestor", rec["code_commit"], "HEAD")
    if anc != 0:
        raise fail("frozen code commit is not an ancestor of HEAD")
    same, _ = _git("diff", "--quiet", rec["code_commit"], "HEAD", "--", "packages", "experiments/phase3")
    if same != 0:
        raise fail("code or experiment definitions changed after the frozen commit")
    _, status = _git("status", "--porcelain")
    out_rel = str((CONFIRMATORY_DIR / experiment_id).relative_to(REPO_ROOT))
    dirty = [ln for ln in status.splitlines() if ln and out_rel not in ln]
    if dirty:
        raise fail(f"working tree not clean ({len(dirty)} paths)")
    return rec


def open_access_log(experiment_id: str, record: dict[str, Any], note: str) -> Path:
    """Create the single-run access log; refuses if it already exists."""
    p = access_log_path(experiment_id)
    if p.exists():
        raise TestPartitionLocked(f"{experiment_id}: confirmatory run already performed ({p})")
    p.parent.mkdir(parents=True, exist_ok=True)
    _, head = _git("rev-parse", "HEAD")
    p.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "head": head,
                "preregistration_hash": record["preregistration_hash"],
                "note": note,
            },
            indent=1,
        )
        + "\n"
    )
    return p
