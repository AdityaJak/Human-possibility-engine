"""Locked data access for Phase III. Every lane loads human data through ``load_family`` only.

The partition lock is applied to raw JSONL records *before* any transcript is parsed: test records are
skipped unread unless ``reports/phase3/freeze/<experiment_id>.json`` exists (Phase II families included:
a Phase II freeze record does not unlock anything here). Participants listed in
``reports/phase3/partitions/exposed_participants.json`` are never scored in a test partition.
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
from hpe.population.core.families import REPO_ROOT, FamilySpec, get_family

DATA_ENV = "HPE_PSYCH101_P3"
SUBSET_SHA256 = "76f335e05b8d940e1175cd084927713ad4cf27123570429297711c223d7c7164"
SUBSET_NOTE = (
    "filter_jsonl(Psych-101 prompts_training.jsonl sha256 7a370a94…, Phase II learning prefixes ∪ "
    "{bahrami2020four, kool2016when, kool2017cost, zorowitz2023data, schulz2020finding, "
    "sadeghiyeh2020temporal, somerville2017charting, feng2021dynamics}); 4,088 transcripts"
)
P3_FREEZE_DIR = REPO_ROOT / "reports" / "phase3" / "freeze"
EXPOSED_FILE = REPO_ROOT / "reports" / "phase3" / "partitions" / "exposed_participants.json"


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
    return Partitions.load(Path(path))


@lru_cache(maxsize=1)
def exposed_keys() -> frozenset[str]:
    if not EXPOSED_FILE.exists():
        return frozenset()
    raw = json.loads(EXPOSED_FILE.read_text(encoding="utf-8"))
    return frozenset(f"{e['experiment']}#{e['participant']}" for e in raw["exposed"])


def load_participants(
    family: str, partition: str, *, experiment_id: str, freeze_dir: Path = P3_FREEZE_DIR
) -> tuple[list[Participant], list[str]]:
    """Parse one family's participants of one partition. Returns (participants, exclusion notes)."""
    spec: FamilySpec = get_family(family)
    check_partition_access(partition, experiment_id, freeze_dir)
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
        freeze_dir=freeze_dir,
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
    family: str, partition: str, *, experiment_id: str, k_max: int = 4, freeze_dir: Path = P3_FREEZE_DIR
) -> tuple[list[EncodedSeq], list[str]]:
    parts, excluded = load_participants(family, partition, experiment_id=experiment_id, freeze_dir=freeze_dir)
    return [encode(p, family, k_max) for p in parts], excluded
