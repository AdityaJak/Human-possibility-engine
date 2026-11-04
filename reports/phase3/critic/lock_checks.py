#!/usr/bin/env python3
"""Critic (audit/critic-v3): integrity checks of the Phase III test lock and partitions. Reads no transcripts.

    PYTHONPATH=packages python reports/phase3/critic/lock_checks.py --out reports/phase3/critic/lock_checks.json

Positive checks (should pass): both committed partition files are self-consistent, disjoint, reproduce
from their own id lists with the committed split scheme, and match the digests quoted in the registry;
the exposed-participant list agrees with the partition file.

Negative checks (each demonstrates a way around the lock, *without* reading any human data): a forged
freeze record in an arbitrary directory unlocks ``test``; a freeze record is not bound to a definition,
commit or critic approval; a tampered partition file with a recomputed digest is accepted and can be
attached to a new family name over the same studies; a missing exposed-participant file fails open; the
committed integrity guard cannot see restless4 keys. ``check_partition_access`` is the only function
called with ``partition="test"``; ``load_family``/``load_participants`` are never called with ``test``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from hpe.evaluation.human.partitions import (
    Partitions,
    TestPartitionLocked,
    check_partition_access,
    freeze_record,
)
from hpe.population.core import data as p3data
from hpe.population.core.families import P2_PARTITIONS, P3_PARTITIONS, REPO_ROOT, FamilySpec, register_family

REGISTRY_DIGESTS = {
    "phase2": "7dc2e779",
    "phase3": "eb14a4d9",
}  # prefixes quoted in EXPERIMENT_REGISTRY_V3.md:12-13
SEEDS = {"phase2": 20261105, "phase3": 20261106}
FRACTIONS = {"dev": 0.4, "tune": 0.2, "test": 0.4}


def reproduce_split(path: Path, seed: int) -> dict[str, Any]:
    """Re-run the committed split scheme (scripts/phase{2,3}_partitions.py) on the file's own id lists."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    overlaps: list[str] = []
    test_keys: list[str] = []
    n_test = 0
    for fam, exps in raw["families"].items():
        for exp, v in exps.items():
            sets = {k: set(map(str, v[k])) for k in ("dev", "tune", "test")}
            if sets["dev"] & sets["tune"] or sets["dev"] & sets["test"] or sets["tune"] & sets["test"]:
                overlaps.append(f"{fam}/{exp}")
            ids = sorted(sets["dev"] | sets["tune"] | sets["test"], key=lambda s: (len(s), s))
            if len(ids) != v["n"]:
                mismatches.append(f"{exp}: n {v['n']} != {len(ids)} ids")
            digest = int.from_bytes(hashlib.sha256(exp.encode()).digest()[:4], "little")
            rng = np.random.default_rng([seed, digest])  # the committed scheme, reproduced verbatim
            perm = [ids[i] for i in rng.permutation(len(ids))]
            n = len(perm)
            n_dev, n_tune = int(round(n * FRACTIONS["dev"])), int(round(n * FRACTIONS["tune"]))
            exp_parts = {
                "dev": set(perm[:n_dev]),
                "tune": set(perm[n_dev : n_dev + n_tune]),
                "test": set(perm[n_dev + n_tune :]),
            }
            if exp_parts != sets:
                mismatches.append(f"{exp}: split not reproduced")
            test_keys += [f"{exp}#{p}" for p in v["test"]]
            n_test += len(v["test"])
    recomputed = hashlib.sha256("\n".join(sorted(test_keys)).encode()).hexdigest()
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_seed": raw["split_seed"],
        "test_digest_recorded": raw["test_partition_sha256"],
        "test_digest_recomputed": recomputed,
        "digest_ok": recomputed == raw["test_partition_sha256"],
        "n_test_keys": n_test,
        "overlaps": overlaps,
        "split_reproduced": not mismatches,
        "mismatches": mismatches,
    }


def exposed_consistency() -> dict[str, Any]:
    raw = json.loads(p3data.EXPOSED_FILE.read_text(encoding="utf-8"))
    p3 = json.loads(P3_PARTITIONS.read_text(encoding="utf-8"))
    where: dict[str, str] = {}
    for exps in p3["families"].values():
        for exp, v in exps.items():
            for part in ("dev", "tune", "test"):
                for pid in v[part]:
                    where[f"{exp}#{pid}"] = part
    rows = []
    for e in raw["exposed"]:
        k = f"{e['experiment']}#{e['participant']}"
        rows.append({"key": k, "recorded": e["partition"], "partition_file": where.get(k, "not partitioned")})
    bad = [
        r
        for r in rows
        if not (
            r["recorded"] == r["partition_file"]
            or (r["partition_file"] == "not partitioned" and r["recorded"].startswith("not partitioned"))
        )
    ]
    studies = sorted({r["key"].split("/", 1)[0] for r in rows})
    return {
        "entries": len(rows),
        "experiment_files": len({r["key"].split("#", 1)[0] for r in rows}),
        "studies": len(studies),
        "in_test_experiment_files": [
            r["key"].split("#", 1)[0] for r in rows if r["partition_file"] == "test"
        ],
        "disagreements": bad,
        "registry_text": "D3-1 says 'the first transcript of 16 study files' (EXPERIMENT_REGISTRY_V3.md:19)",
    }


def forged_freeze() -> dict[str, Any]:
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        try:
            check_partition_access("test", "critic_forged", freeze_dir=d)
            out["without_record"] = "UNLOCKED (unexpected)"
        except TestPartitionLocked:
            out["without_record"] = "locked (expected)"
        (d / "critic_forged.json").write_text(
            json.dumps(
                {
                    "experiment_id": "anything",
                    "preregistration_hash": "not-a-hash",
                    "code_version": "none",
                    "frozen_at": "never",
                }
            )
        )
        try:
            check_partition_access("test", "critic_forged", freeze_dir=d)
            out["forged_record_outside_repo"] = (
                "UNLOCKED: an uncommitted 4-field file in a temp dir opens test"
            )
        except TestPartitionLocked:
            out["forged_record_outside_repo"] = "locked"
        rec = freeze_record("critic_forged", d)
        out["fields_checked"] = (
            "presence of experiment_id, preregistration_hash, code_version, frozen_at only"
        )
        out["record_accepted"] = rec
        out["experiment_id_in_record_matches_file_name"] = bool(
            rec and rec.get("experiment_id") == "critic_forged"
        )
        out["load_family_accepts_freeze_dir_argument"] = (
            "freeze_dir" in p3data.load_family.__code__.co_varnames
        )
    return out


def tampered_partitions() -> dict[str, Any]:
    raw = json.loads(P3_PARTITIONS.read_text(encoding="utf-8"))
    exp = "bahrami2020four/exp.csv"
    v = raw["families"]["restless4"][exp]
    moved = v["test"][0]  # an id that is already public in the committed partition file
    v["test"] = v["test"][1:]
    v["dev"] = sorted([*v["dev"], moved], key=lambda s: (len(s), s))
    keys = sorted(f"{e}#{p}" for fam in raw["families"].values() for e, vv in fam.items() for p in vv["test"])
    raw["test_partition_sha256"] = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "psych101_p3_tampered.json"
        f.write_text(json.dumps(raw))
        try:
            parts = Partitions.load(f)
            accepted = True
            moved_to = parts.assignment[(exp, moved)]
        except ValueError as e:
            accepted, moved_to = False, str(e)
        # the registry accepts a second family name over the same studies with another partition file
        try:
            register_family(
                FamilySpec(
                    "critic_restless4_alias", ("bahrami2020four/",), 4, f, lambda *_: None, "secondary"
                )
            )  # type: ignore[arg-type, return-value]
            alias = "accepted: same studies, different partition file, no warning"
        except ValueError as e:
            alias = f"refused: {e}"
    return {
        "tampered_file_accepted_by_Partitions.load": accepted,
        "a_test_participant_now_assigned_to": moved_to,
        "expected_digests_pinned_in_code": False,
        "alias_family_registration": alias,
    }


def exposed_fail_open() -> dict[str, Any]:
    orig = p3data.EXPOSED_FILE
    try:
        p3data.EXPOSED_FILE = REPO_ROOT / "reports" / "phase3" / "partitions" / "does_not_exist.json"
        p3data.exposed_keys.cache_clear()
        keys = p3data.exposed_keys()
        return {
            "missing_file_result": sorted(keys),
            "raises": False,
            "note": "fails open: no exclusion, no error",
        }
    finally:
        p3data.EXPOSED_FILE = orig
        p3data.exposed_keys.cache_clear()


def guard_regex() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("phase2_guard", REPO_ROOT / "scripts" / "phase2_guard.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    samples = {
        "bahrami2020four/exp.csv#99999": bool(mod.KEY_RE.search("bahrami2020four/exp.csv#99999")),
        "kool2016when/exp1.csv#99999": bool(mod.KEY_RE.search("kool2016when/exp1.csv#99999")),
        "garcia2023experiential/exp1.csv#99999": bool(
            mod.KEY_RE.search("garcia2023experiential/exp1.csv#99999")
        ),
    }
    phase3_guard = sorted(p.name for p in (REPO_ROOT / "scripts").glob("phase3_*guard*"))
    return {
        "KEY_RE": mod.KEY_RE.pattern,
        "matches": samples,
        "default_partitions_file": str(mod.DEFAULT_PARTITIONS),
        "phase3_guard_scripts": phase3_guard,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/phase3/critic/lock_checks.json"))
    args = ap.parse_args()
    p2 = reproduce_split(P2_PARTITIONS, SEEDS["phase2"])
    p3 = reproduce_split(P3_PARTITIONS, SEEDS["phase3"])
    for k, rep in (("phase2", p2), ("phase3", p3)):
        rep["matches_registry_prefix"] = rep["test_digest_recorded"].startswith(REGISTRY_DIGESTS[k])
    res = {
        "positive": {"phase2_partitions": p2, "phase3_partitions": p3, "exposed_list": exposed_consistency()},
        "negative": {
            "L1_forged_freeze_record": forged_freeze(),
            "L3_tampered_partitions": tampered_partitions(),
            "L4_exposed_file_fail_open": exposed_fail_open(),
            "L7_guard_blind_to_phase3_keys": guard_regex(),
        },
    }
    args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
