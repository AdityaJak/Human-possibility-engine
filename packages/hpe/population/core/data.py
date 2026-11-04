"""Locked data access for Phase III. Every lane loads human data through ``load_family`` only.

The partition lock is applied to raw JSONL records *before* any transcript is parsed. The test partition
opens only when ``hpe.population.core.freeze.validate_freeze`` passes (fixed freeze directory, committed
freeze record and separate critic approval, matching preregistration hash, unchanged code, clean tree,
no previous run). Partition files are pinned by SHA-256. Participants listed in
``reports/phase3/partitions/exposed_participants.json`` are never scored in a test partition; a missing
exposure list is an error (fail closed).
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from hpe.evaluation.human import psych101
from hpe.evaluation.human.partitions import Partitions, check_partition_access, iter_partition_records
from hpe.evaluation.human.psych101 import Participant
from hpe.population.core.encoding import EncodedSeq, encode
from hpe.population.core.families import P2_PARTITIONS, P3_PARTITIONS, REPO_ROOT, FamilySpec, get_family
from hpe.population.core.freeze import FREEZE_DIR, validate_freeze

DATA_ENV = "HPE_PSYCH101_P3"
SUBSET_SHA256 = "76f335e05b8d940e1175cd084927713ad4cf27123570429297711c223d7c7164"
SUBSET_NOTE = (
    "filter_jsonl(Psych-101 prompts_training.jsonl sha256 7a370a94…, Phase II learning prefixes ∪ "
    "{bahrami2020four, kool2016when, kool2017cost, zorowitz2023data, schulz2020finding, "
    "sadeghiyeh2020temporal, somerville2017charting, feng2021dynamics}); 4,088 transcripts"
)
P3_FREEZE_DIR = FREEZE_DIR
EXPOSED_FILE = REPO_ROOT / "reports" / "phase3" / "partitions" / "exposed_participants.json"
# Pinned partition files (critic finding T-2): any edit, even one with a recomputed internal digest, is refused.
PINNED_PARTITIONS = {
    P2_PARTITIONS: "854054b4052bb8933f459c0fd4541c99398053d38ed3dbb5d499bee29597c15d",
    P3_PARTITIONS: "bfd0c52e1e4d27b29ed6ab0de78cb162b02f98ea2ceee68cdb82ac919c33b832",
}


class DataUnavailable(RuntimeError):
    pass


def data_path() -> Path:
    p = os.environ.get(DATA_ENV)
    if not p or not Path(p).exists():
        raise DataUnavailable(
            f"set {DATA_ENV} to the Phase III Psych-101 subset ({SUBSET_NOTE}); build it with "
            "python scripts/phase3_subset.py"
        )
    return Path(p)


@lru_cache(maxsize=4)
def _verified(path: str) -> bool:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if h != SUBSET_SHA256:
        raise DataUnavailable(f"{path}: sha256 {h} != expected {SUBSET_SHA256}")
    return True


@lru_cache(maxsize=4)
def _partitions(path: str) -> Partitions:
    pinned = PINNED_PARTITIONS.get(Path(path))
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if pinned is None or actual != pinned:
        raise DataUnavailable(f"partition file {path} is not pinned or was modified (sha256 {actual})")
    return Partitions.load(Path(path))


@lru_cache(maxsize=1)
def exposed_keys() -> frozenset[str]:
    if not EXPOSED_FILE.exists():
        raise DataUnavailable(f"{EXPOSED_FILE} is missing; refusing to load data without the exposure list")
    raw = json.loads(EXPOSED_FILE.read_text(encoding="utf-8"))
    return frozenset(f"{e['experiment']}#{e['participant']}" for e in raw["exposed"])


def load_participants(
    family: str, partition: str, *, experiment_id: str, definition_hash: str | None = None
) -> tuple[list[Participant], list[str]]:
    """Parse one family's participants of one partition. Returns (participants, exclusion notes)."""
    spec: FamilySpec = get_family(family)
    if partition == "test":
        validate_freeze(experiment_id, definition_hash=definition_hash)
    check_partition_access(partition, experiment_id, P3_FREEZE_DIR)
    path = data_path()
    _verified(str(path))
    parts = _partitions(str(spec.partition_file))
    exps = sorted(e for e in {e for es in parts.families.values() for e in es} if e.startswith(spec.prefixes))
    out: list[Participant] = []
    excluded: list[str] = []
    exposed = exposed_keys()
    for rec in iter_partition_records(
        psych101.iter_records(path, spec.prefixes),
        parts,
        partition,
        experiments=exps,
        experiment_id=experiment_id,
        freeze_dir=P3_FREEZE_DIR,
    ):
        key = f"{rec['experiment']}#{rec['participant']}"
        if partition == "test" and key in exposed:
            excluded.append(f"{key}: exposed before the partition freeze (deviation D3-1)")
            continue
        try:
            p = spec.parser(rec["text"], rec["experiment"], str(rec["participant"]))
        except psych101.ParseError as e:
            excluded.append(str(e)[:200])
            continue
        if not p.trials or all(getattr(t, "forced", False) for t in p.trials):
            excluded.append(f"{p.key}: no scorable trials")
            continue
        out.append(p)
    out.sort(key=lambda p: p.key)
    return out, excluded


def load_family(
    family: str, partition: str, *, experiment_id: str, k_max: int = 4, definition_hash: str | None = None
) -> tuple[list[EncodedSeq], list[str]]:
    parts, excluded = load_participants(
        family, partition, experiment_id=experiment_id, definition_hash=definition_hash
    )
    return [encode(p, family, k_max) for p in parts], excluded
