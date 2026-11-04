#!/usr/bin/env python3
"""Critic (audit/critic-v3), pass 2: re-attack the hardened Phase III core. Toy data only; no human data.

    PYTHONPATH=packages python reports/phase3/critic/reattack_v2.py --out reports/phase3/critic/reattack_v2.json

Part A  The 7 original cheating models (reports/phase3/critic/causal_attacks.py) through the *production*
        runner (select -> strict check -> isolated report -> analyze) on two toy counted families.
Part B  New attacks written for this pass:
        B1 ClassPool      transductive pooling of held-out participants in a *class attribute* (survives deep
                          copies, invisible to state_digest(model.__dict__)), self-excluded by a 10-row fingerprint
                          and applied only from row 10 on, so prefix-truncation probes see identical rows.
        B2 ArmDetector    AdaptiveBase model whose states carry the history they were built from; it detects the
                          donor arm (history != own prefix) causally and degrades only that arm -> Gate A2.
        B3 WeakTwin       AdaptiveBase model whose twin_config() names a crippled pop_gru -> Gate B1.
        B4 encoding-leak  check_encoding_leak on a 4-action toy family with a slot equal to the current action.
Part C  Lock: the original lock attacks (forged record, tampered partitions, alias family, missing exposure
        list, Phase II helper, guard regex) plus, in a throw-away git clone in the scratch directory, the
        Phase III freeze rules: the lock without a definition hash or access log, after the record is edited
        post-approval, the test stage's own lock after it writes its access log, and a second run after the
        untracked access log is deleted. Only the lock functions are called (validate_freeze and
        check_partition_access, exactly what load_family calls before opening a file); HPE_PSYCH101_P3 is
        unset in the clone, so no data file can be opened. (A first, discarded version of this probe called
        load_family(..., "test") in the clone; it stopped at DataUnavailable and read nothing.)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("critic_v1", ROOT / "reports/phase3/critic/causal_attacks.py")
v1 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["critic_v1"] = v1
_spec.loader.exec_module(v1)  # type: ignore[union-attr]

from hpe.population.core import gates, runner  # noqa: E402
from hpe.population.core.adaptation import AdaptiveBase  # noqa: E402
from hpe.population.core.conformance import check_causal, check_encoding_leak  # noqa: E402
from hpe.population.core.encoding import EncodedSeq  # noqa: E402
from hpe.population.core.families import P3_PARTITIONS, FamilySpec, register_family  # noqa: E402
from hpe.population.core.interface import ModelSpec, register_model  # noqa: E402

SEED = v1.SEED
K = v1.K_MAX
Honest = v1.Honest


# ----------------------------------------------------------------------------- new attack models
class ClassPool(Honest):
    """Population prior updated with every participant it predicts, stored on the class."""

    R = 10
    _POOL: dict[str, np.ndarray] = {}  # noqa: RUF012  (deliberately class-level)

    def __init__(self, name: str = "critic_classpool") -> None:
        super().__init__(name)

    @staticmethod
    def _fp(seq: EncodedSeq, r: int) -> str | None:
        if seq.T < r:
            return None
        return hashlib.sha256(seq.actions[:r].tobytes() + seq.x[:r].tobytes()).hexdigest()

    def predict_logprobs(self, seq, *, mode="adapt", donor=None):  # type: ignore[no-untyped-def]
        own = self._fp(seq, self.R)
        pool = np.zeros(K)
        for fp, counts in sorted(type(self)._POOL.items()):
            if fp != own:
                pool += counts
        honest = self._honest(seq, self.prior)
        boosted = self._honest(seq, (self.prior * 50.0 + pool) / (self.prior * 50.0 + pool).sum())
        out = honest.copy()
        out[self.R :] = boosted[self.R :]  # rows < R never depend on the pool
        if own is not None and seq.T >= 40 and own not in type(self)._POOL:  # never pool short probes
            c = np.zeros(K)
            np.add.at(c, seq.actions[seq.scored], 1.0)
            type(self)._POOL[own] = c
        return out


class ArmDetector(AdaptiveBase):
    """Honest running counts; a state built from a history that is not the target's own prefix (the donor arm)
    switches the rows it governs to a confidently wrong prior. Causal: the comparison uses only rows < start."""

    name = "critic_armdetector"
    update_every = 5

    def __init__(self) -> None:
        self.prior = np.ones(K) / K

    def fit(self, train, *, seed, threads, budget_s):  # type: ignore[no-untyped-def]
        c = np.ones(K)
        for s in train:
            np.add.at(c, s.actions[s.scored], 1.0)
        self.prior = c / c.sum()
        return {}

    def init_state(self):  # type: ignore[no-untyped-def]
        return np.zeros(0, dtype=np.int64)

    def update_state(self, state, history):  # type: ignore[no-untyped-def]
        return history.actions.copy()

    def predict_with_states(self, seq, states):  # type: ignore[no-untyped-def]
        good = Honest._honest(self, seq, self.prior)  # type: ignore[arg-type]
        bad_prior = np.full(K, 0.02)
        bad_prior[K - 1] = 1.0
        bad = Honest._honest(self, seq, bad_prior / bad_prior.sum(), weight=40.0)  # type: ignore[arg-type]
        out = good.copy()
        for i, (start, hist) in enumerate(states):
            end = states[i + 1][0] if i + 1 < len(states) else seq.T
            if hist.size and not np.array_equal(hist, seq.actions[: hist.size]):
                out[start:end] = bad[start:end]
        return out

    def param_count(self) -> int:
        return K

    def twin_config(self):  # type: ignore[no-untyped-def]
        return None


class WeakTwin(ArmDetector):
    """Identical to an honest adaptive model; only its twin_config() is chosen to be weak."""

    name = "critic_weaktwin"
    TWIN: dict[str, Any] = {"hidden": 1, "lr": 1e-4, "max_epochs": 1, "patience": 1}  # noqa: RUF012

    def predict_with_states(self, seq, states):  # type: ignore[no-untyped-def]
        return Honest._honest(self, seq, self.prior)  # type: ignore[arg-type]

    def twin_config(self):  # type: ignore[no-untyped-def]
        return dict(self.TWIN)


class FairTwin(WeakTwin):
    name = "critic_fairtwin"
    TWIN: dict[str, Any] = {"hidden": 16, "lr": 0.01, "max_epochs": 30, "patience": 5}  # noqa: RUF012


# ----------------------------------------------------------------------------- toy runner harness
def _register(name: str, cls: type, adaptive: bool) -> None:
    def factory(config: dict[str, Any], k: int, d: int, _cls: type = cls) -> Any:
        return _cls()

    register_model(ModelSpec(name, factory, "critic", "toy", {}, adaptive=adaptive))


def _named(cls: type, name: str) -> type:
    return type(f"N_{name}", (cls,), {"__init__": lambda self: cls.__init__(self, name)})


def _families(tag: str) -> dict[str, str]:
    fams = {f"critic2{tag}_a": "counted_primary", f"critic2{tag}_b": "counted"}
    for fam, role in fams.items():
        try:
            register_family(FamilySpec(fam, (f"{fam}/",), 4, P3_PARTITIONS, lambda *_: None, role))  # type: ignore[arg-type, return-value]
        except ValueError:
            pass
    return fams


def _data(fam: str, shift: float) -> dict[str, list[EncodedSeq]]:
    tag = fam.rsplit("_", 1)[1]

    def fix(s: EncodedSeq) -> EncodedSeq:
        s.family = fam
        s.key = f"{fam}/exp1.csv#{s.key.split('#', 1)[1]}"
        return s

    return {
        "dev": [fix(s) for s in v1.toy_seqs(60, k=4, T=80, tag=tag)],
        "tune": [fix(s) for s in v1.toy_seqs(40, k=4, T=80, shift=shift, tag=tag, start=5000)],
    }


def run_toy(tag: str, models: dict[str, runner.ModelRun], shift: float = 2.0) -> dict[str, Any]:
    fams = _families(tag)
    defn = runner.P3Definition(
        experiment_id=f"critic2_{tag}",
        title="critic toy",
        lane="audit/critic-v3",
        hypothesis="toy",
        families=list(fams),
        seed=SEED,
        threads=1,
        models=models,
    )
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td) / defn.experiment_id
        (outdir / "parts").mkdir(parents=True)
        (outdir / "definition.json").write_text(
            json.dumps({**defn.model_dump(), "preregistration_hash": defn.preregistration_hash()})
        )
        records = {}
        for fam in fams:
            ClassPool._POOL.clear()
            data = _data(fam, shift)
            for name, mrun in defn.models.items():
                r = runner.run_model_family(defn, fam, name, mrun, data)
                (outdir / "parts" / f"{fam}__{name}.json").write_text(json.dumps(r))
                records.setdefault(fam, {})[name] = {
                    "violations": r["causality_violations"],
                    "select_nll": r["selected"]["select_nll"],
                    "twin_select_nll": (r.get("twin") or {}).get("select_nll"),
                }
        summary = runner.analyze(outdir)
    contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    for fam in fams:
        for c in summary["families"][fam]["contrasts"]:
            contrasts.setdefault(c["kind"] + ":" + c["first"], {})[fam] = c
    return {
        "families": fams,
        "records": records,
        "excluded": {f: summary["families"][f]["excluded_models"] for f in fams},
        "best_comparator": {f: summary["families"][f]["best_comparator"] for f in fams},
        "contrasts": {
            k: {
                f: {kk: c[kk] for kk in ("second", "estimate", "ci_low", "ci_high", "p_one_sided", "n")}
                for f, c in v.items()
            }
            for k, v in contrasts.items()
        },
        "_raw_contrasts": contrasts,
    }


# ----------------------------------------------------------------------------- parts
def part_a() -> dict[str, Any]:
    named = {
        "c2_honest": (Honest, False, "comparator"),
        "c2_peek_all": (_named(v1.PeekAll, "c2_peek_all"), False, "candidate"),
        "c2_probe_dodger": (_named(v1.ProbeDodger, "c2_probe_dodger"), False, "candidate"),
        "c2_keycache": (_named(v1.KeyCache, "c2_keycache"), False, "candidate"),
        "c2_transductive": (_named(v1.Transductive, "c2_transductive"), False, "candidate"),
        "c2_invariant": (_named(v1.InvariantFields, "c2_invariant"), False, "candidate"),
        "c2_donor_peek": (_named(v1.DonorPathPeek, "c2_donor_peek"), True, "candidate"),
        "c2_sabotaged": (_named(v1.SabotagedAblations, "c2_sabotaged"), True, "candidate"),
    }
    models = {}
    for name, (cls, adaptive, role) in named.items():
        _register(name, cls, adaptive)
        models[name] = runner.ModelRun(role=role, variants=["mismatched"] if adaptive else [])  # type: ignore[arg-type]
    res = run_toy("a", models)
    res.pop("_raw_contrasts")
    # self-peek miss rate on 2-action sequences, production check as the runner calls it (20 sequences)
    seqs2 = v1.toy_seqs(400, k=2, T=150, tag="k2")
    missed = sum(1 for i in range(0, 400, 20) if not check_causal(v1.PeekAll(), seqs2[i : i + 20], seed=SEED))
    res["peek_all_k2_missed_batches_of_20"] = {"batches": 20, "missed": missed}
    return res


def part_b() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # B1 class-level transductive pool: production check, then the runner and analyze
    _register("c2_classpool", ClassPool, False)
    _register("c2_honest_b", Honest, False)
    ClassPool._POOL.clear()
    tune = _data("critic2b_a", 2.0)["tune"]
    m = ClassPool()
    m.fit(_data("critic2b_a", 2.0)["dev"], seed=SEED, threads=1, budget_s=60)
    _ = [runner.predict_isolated(m, s, f"tok{i}") for i, s in enumerate(tune)]  # selection pass
    out["B1_classpool_strict_check"] = check_causal(m, tune[:20], seed=SEED)
    res = run_toy(
        "b",
        {
            "c2_honest_b": runner.ModelRun(role="comparator"),
            "c2_classpool": runner.ModelRun(role="candidate"),
        },
    )
    out["B1_classpool_runner"] = {
        "excluded": res["excluded"],
        "vs_best_comparator": res["contrasts"].get("vs_best_comparator:c2_classpool"),
    }
    # history invariance (the proposed extra probe): predict A, then 5 unseen others, then A again
    ClassPool._POOL.clear()
    a = runner.predict_isolated(m, tune[0], "probe-A")
    for i, s in enumerate(tune[1:6]):
        runner.predict_isolated(m, s, f"probe-o{i}")
    b = runner.predict_isolated(m, tune[0], "probe-A2")
    out["B1_history_invariance_probe_detects"] = bool(np.abs(a - b)[np.isfinite(a)].max() > 1e-5)

    # B2 arm detector (A2 sabotage through the harness) and B3 weak / fair twin (B1)
    _register("c2_honest_c", Honest, False)
    _register("c2_armdetector", ArmDetector, True)
    _register("c2_weaktwin", WeakTwin, True)
    _register("c2_fairtwin", FairTwin, True)
    res = run_toy(
        "c",
        {
            "c2_honest_c": runner.ModelRun(role="comparator"),
            "c2_armdetector": runner.ModelRun(role="candidate", variants=["mismatched"]),
            "c2_weaktwin": runner.ModelRun(role="candidate", variants=["twin"]),
            "c2_fairtwin": runner.ModelRun(role="candidate", variants=["twin"]),
        },
        shift=0.0,
    )
    raw = res.pop("_raw_contrasts")
    fams = list(res["families"])
    counted = fams
    primary = fams[0]
    a2 = gates.gate_a2(raw.get("vs_mismatched:c2_armdetector", {}), counted)
    out["B2_armdetector"] = {
        "excluded": res["excluded"],
        "vs_mismatched": res["contrasts"].get("vs_mismatched:c2_armdetector"),
        "vs_best_comparator": res["contrasts"].get("vs_best_comparator:c2_armdetector"),
        "gate_a2_passed": a2.passed,
    }
    for nm in ("c2_weaktwin", "c2_fairtwin"):
        vs_twin = raw.get(f"vs_twin:{nm}", {})
        b1 = gates._family_table(vs_twin, counted, gates.MARGIN_B1)
        out[f"B3_{nm}"] = {
            "excluded": res["excluded"],
            "twin_select_nll": {f: res["records"][f][nm]["twin_select_nll"] for f in fams},
            "vs_twin": res["contrasts"].get(f"vs_twin:{nm}"),
            "vs_best_comparator": res["contrasts"].get(f"vs_best_comparator:{nm}"),
            "b1_families_meeting_margin": [f for f, v in b1.items() if v["meets_margin"]],
            "primary": primary,
        }
    # B4 encoding-leak check: a slot that equals the current action
    for k in (2, 4):
        seqs = v1.toy_seqs(40, k=k, T=60, tag=f"leak{k}")
        for s in seqs:
            s.x[:, 0] = s.actions / max(1, k - 1)
        out[f"B4_encoding_leak_k{k}"] = check_encoding_leak(seqs)
    return out


def part_c() -> dict[str, Any]:
    out: dict[str, Any] = {}
    _lc = importlib.util.spec_from_file_location(
        "critic_lock_v1", ROOT / "reports/phase3/critic/lock_checks.py"
    )
    lc = importlib.util.module_from_spec(_lc)  # type: ignore[arg-type]
    _lc.loader.exec_module(lc)  # type: ignore[union-attr]
    import hpe.population.core.data as d

    for name, fn in (
        ("L1_forged_freeze_record", lc.forged_freeze),
        ("L3_tampered_partitions_and_alias", lc.tampered_partitions),
        ("L4_missing_exposure_list", lc.exposed_fail_open),
        ("L7_phase2_guard_regex", lc.guard_regex),
    ):
        saved = d.EXPOSED_FILE
        try:
            out[name] = {"returned": fn()}
        except Exception as e:
            out[name] = {"refused": type(e).__name__, "message": str(e)[:200]}
        finally:
            d.EXPOSED_FILE = saved
            d.exposed_keys.cache_clear()
    # Phase III guard regex
    g = importlib.util.spec_from_file_location("p3guard", ROOT / "scripts/phase3_guard.py")
    gm = importlib.util.module_from_spec(g)  # type: ignore[arg-type]
    g.loader.exec_module(gm)  # type: ignore[union-attr]
    out["phase3_guard_matches_restless4_key"] = bool(gm.KEY_RE.search("bahrami2020four/exp.csv#99999"))
    # Phase II helper path to garcia/described test
    from hpe.evaluation.human import runner2

    try:
        runner2.load_partition_participants(
            Path("/nonexistent.jsonl"),
            "garcia",
            [],
            None,
            "test",
            experiment_id="anything",  # type: ignore[arg-type]
        )
        out["phase2_helper_test"] = "UNLOCKED"
    except Exception as e:
        out["phase2_helper_test"] = f"refused: {type(e).__name__}"
    # GBE (external family) partition file: pinned?
    from hpe.evaluation.human.partitions import PINNED_PARTITION_SHA256
    from hpe.population.affect import gbe

    gbe_sha = hashlib.sha256(gbe.PARTITION_FILE.read_bytes()).hexdigest()
    out["gbe_partition_file_pinned"] = gbe_sha in PINNED_PARTITION_SHA256 or gbe_sha in set(
        d.PINNED_PARTITIONS.values()
    )
    raw = json.loads(gbe.PARTITION_FILE.read_text())
    moved = raw["test"][0]
    raw["test"] = raw["test"][1:]
    raw["dev"] = [*raw["dev"], moved]
    raw["test_partition_sha256"] = gbe.test_digest(raw["test"])
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "gbe_tampered.json"
        f.write_text(json.dumps(raw))
        try:
            assign = gbe.load_partition(f)
            out["gbe_tampered_partition"] = f"accepted (a test participant is now '{assign[int(moved)]}')"
        except Exception as e:
            out["gbe_tampered_partition"] = f"refused: {type(e).__name__}"
    out["clone_freeze_rules"] = clone_freeze_rules()
    return out


CLONE_PROBE = textwrap.dedent(
    """
    import json, os, subprocess, sys
    os.environ.pop("HPE_PSYCH101_P3", None)
    from hpe.population.core import freeze
    from hpe.evaluation.human.partitions import TestPartitionLocked, check_partition_access
    phase = sys.argv[1]
    eid = "critic_probe"
    out = {}
    def attempt(tag, fn):
        try:
            fn(); out[tag] = "passed the lock"
        except TestPartitionLocked as e:
            out[tag] = "refused: " + str(e)[:160]
        except Exception as e:
            out[tag] = type(e).__name__ + ": " + str(e)[:160]
    # the two calls load_participants makes before it opens the data file (data.py), nothing else
    def lock_as_load_family():
        freeze.validate_freeze(eid, definition_hash=None)
        check_partition_access("test", eid, freeze.FREEZE_DIR)
    if phase == "1":
        attempt("lock_without_definition_hash", lock_as_load_family)
        out["access_log_exists_after_lock_passed"] = freeze.access_log_path(eid).exists()
    elif phase == "2":
        attempt("lock_after_record_edited_post_approval", lock_as_load_family)
    elif phase == "3":
        freeze.open_access_log(eid, freeze.validate_freeze(eid), "critic probe run 1")
        out["git_status_after_access_log"] = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        attempt("test_stage_lock_after_its_own_access_log", lock_as_load_family)
        freeze.access_log_path(eid).unlink()
        attempt("second_run_after_deleting_untracked_access_log",
                lambda: freeze.open_access_log(eid, freeze.validate_freeze(eid), "critic probe run 2"))
    print(json.dumps(out))
    """
)


def clone_freeze_rules() -> dict[str, Any]:
    """Throw-away clone: commit a freeze record + critic approval, then probe the rules. No data file is read."""
    scratch = Path(os.environ.get("CRITIC_SCRATCH", tempfile.gettempdir()))
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        clone = Path(td) / "clone"
        subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
        git = ["git", "-C", str(clone), "-c", "user.name=anyone", "-c", "user.email=anyone@example.invalid"]
        head = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        fz = clone / "reports" / "phase3" / "freeze"
        fz.mkdir(parents=True, exist_ok=True)
        rec = {
            "experiment_id": "critic_probe",
            "preregistration_hash": "0000000000000000",
            "code_commit": head,
            "frozen_at": "2026-11-04",
            "counted_families": ["restless4"],
            "primary_family": "restless4",
            "comparator_pool": [],
            "candidates": [],
            "model_digests": {"restless4__x__model": "a" * 64},
        }
        (fz / "critic_probe.json").write_text(json.dumps(rec))
        (fz / "critic_probe.critic.json").write_text(
            json.dumps({"approved": True, "preregistration_hash": "0000000000000000"})
        )
        subprocess.run([*git, "add", "reports/phase3/freeze"], check=True)
        subprocess.run(
            [*git, "commit", "-q", "-m", "any lane can write and commit this approval"], check=True
        )
        env = {k: v for k, v in os.environ.items() if k != "HPE_PSYCH101_P3"}
        env.update({"PYTHONPATH": str(clone / "packages"), "PYTHONDONTWRITEBYTECODE": "1"})
        out: dict[str, Any] = {"approval_author": "anyone <anyone@example.invalid> (not the critic lane)"}

        def probe(phase: str) -> None:
            r = subprocess.run(
                [sys.executable, "-c", CLONE_PROBE, phase], cwd=clone, env=env, capture_output=True, text=True
            )
            try:
                out.update(json.loads(r.stdout.strip().splitlines()[-1]))
            except Exception:
                out[f"probe_{phase}_error"] = (r.stdout + r.stderr)[-800:]

        probe("1")
        # edit the frozen record's model digests after the approval, commit, probe again
        rec["model_digests"] = {"restless4__x__model": "b" * 64}
        (fz / "critic_probe.json").write_text(json.dumps(rec))
        subprocess.run(
            [*git, "commit", "-q", "-am", "swap model digests after the critic approved"], check=True
        )
        probe("2")
        probe("3")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports/phase3/critic/reattack_v2.json")
    args = ap.parse_args()
    res = {"part_a": part_a(), "part_b": part_b(), "part_c": part_c()}
    args.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str)[:12000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
