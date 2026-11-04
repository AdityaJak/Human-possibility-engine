"""Participant-level dev/test partition of HeartSteps V1, frozen before any discovery analysis.

dev (60%) is for moderator discovery and policy learning; test (40%) is read once per pre-specified
confirmation. The split is a deterministic function of the sorted participant ids and the seed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_SEED = 20261105
DEV_FRACTION = 0.6


def split_users(
    users: list[int], seed: int = SPLIT_SEED, dev_fraction: float = DEV_FRACTION
) -> dict[str, list[int]]:
    ids = sorted(set(users))
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    n_dev = int(round(len(perm) * dev_fraction))
    return {"dev": sorted(perm[:n_dev]), "test": sorted(perm[n_dev:])}


def partition_record(users: list[int], seed: int = SPLIT_SEED) -> dict:
    parts = split_users(users, seed)
    digest = hashlib.sha256(",".join(map(str, parts["test"])).encode()).hexdigest()
    return {
        "dataset": "heartsteps_v1_suggestions",
        "split_seed": seed,
        "dev_fraction": DEV_FRACTION,
        "n_users": len(set(users)),
        "dev": parts["dev"],
        "test": parts["test"],
        "test_sha256": digest,
    }


def write_partition(users: list[int], path: Path, seed: int = SPLIT_SEED) -> dict:
    rec = partition_record(users, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("test_sha256") != rec["test_sha256"]:
            raise RuntimeError(f"partition at {path} differs from the frozen one; refusing to overwrite")
        return existing
    path.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def load_partition(path: Path) -> dict:
    rec = json.loads(path.read_text())
    digest = hashlib.sha256(",".join(map(str, rec["test"])).encode()).hexdigest()
    if digest != rec["test_sha256"]:
        raise RuntimeError("partition file has been altered (test sha256 mismatch)")
    return rec
