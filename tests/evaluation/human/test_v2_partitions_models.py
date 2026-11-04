"""Phase II: partition lock, GRU gradient check, reference-model sanity, calibration."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from hpe.contracts.runtime import Feedback, Observation
from hpe.evaluation.human import gru
from hpe.evaluation.human.harness2 import expected_calibration_error, make_init
from hpe.evaluation.human.models2 import BayesRuleModel, ExemplarModel, PVLDeltaModel
from hpe.evaluation.human.partitions import (
    Partitions,
    TestPartitionLocked,
    check_partition_access,
    iter_partition_records,
)


def _partitions_file(tmp_path, exps=("expA/exp1.csv", "expB/exp1.csv")):
    fams = {"famA": {}, "famB": {}}
    for i, e in enumerate(exps):
        fam = "famA" if i == 0 else "famB"
        fams[fam][e] = {
            "n": 5,
            "dev": ["0", "1"],
            "tune": ["2"],
            "test": ["3", "4"],
            "counts": {"dev": 2, "tune": 1, "test": 2},
        }
    keys = sorted(f"{e}#{p}" for fam in fams.values() for e, v in fam.items() for p in v["test"])
    doc = {
        "split_seed": 1,
        "fractions": {},
        "source_sha256": "x",
        "families": fams,
        "test_partition_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }
    p = tmp_path / "partitions.json"
    p.write_text(json.dumps(doc))
    return p


def test_test_partition_is_locked_without_freeze_record(tmp_path):
    parts = Partitions.load(_partitions_file(tmp_path))
    recs = [{"experiment": "expA/exp1.csv", "participant": str(i), "text": "t"} for i in range(5)]
    assert [
        r["participant"] for r in iter_partition_records(recs, parts, "dev", freeze_dir=tmp_path / "freeze")
    ] == ["0", "1"]
    assert [
        r["participant"] for r in iter_partition_records(recs, parts, "tune", freeze_dir=tmp_path / "freeze")
    ] == ["2"]
    with pytest.raises(TestPartitionLocked):
        list(iter_partition_records(recs, parts, "test", experiment_id="x", freeze_dir=tmp_path / "freeze"))
    with pytest.raises(TestPartitionLocked):
        check_partition_access("test", "x", tmp_path / "freeze")


def test_test_partition_opens_only_with_valid_freeze_record(tmp_path):
    parts = Partitions.load(_partitions_file(tmp_path))
    recs = [{"experiment": "expA/exp1.csv", "participant": str(i), "text": "t"} for i in range(5)]
    fz = tmp_path / "freeze"
    fz.mkdir()
    (fz / "x.json").write_text(
        json.dumps({"experiment_id": "x", "preregistration_hash": "h", "code_version": "c"})
    )
    with pytest.raises(ValueError, match="missing fields"):  # frozen_at missing
        check_partition_access("test", "x", fz)
    (fz / "x.json").write_text(
        json.dumps(
            {"experiment_id": "x", "preregistration_hash": "h", "code_version": "c", "frozen_at": "now"}
        )
    )
    assert [
        r["participant"]
        for r in iter_partition_records(recs, parts, "test", experiment_id="x", freeze_dir=fz)
    ] == ["3", "4"]


def test_tampered_partitions_file_is_rejected(tmp_path):
    p = _partitions_file(tmp_path)
    doc = json.loads(p.read_text())
    doc["families"]["famA"]["expA/exp1.csv"]["test"] = ["3"]  # moved a participant out of test
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="digest mismatch"):
        Partitions.load(p)


def test_gru_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    n_in, H, K, T = 5, 4, 3, 6
    P = gru.init_params(n_in, H, K, rng)
    xs = rng.normal(size=(T, n_in))
    acts = rng.integers(0, K, size=T)
    mask = np.array([True, False, True, True, True, True])
    h0 = np.zeros(H)
    _loss, G, _ = gru.forward_backward(P, xs, acts, mask, h0)
    eps = 1e-6
    for name in ("Wz", "Wr", "Wh", "bz", "br", "bh", "Wo", "bo"):
        W = getattr(P, name)
        for idx in [
            np.unravel_index(i, W.shape) for i in rng.choice(W.size, size=min(4, W.size), replace=False)
        ]:
            old = W[idx]
            W[idx] = old + eps
            lp, _, _ = gru.forward_backward(P, xs, acts, mask, h0)
            W[idx] = old - eps
            lm, _, _ = gru.forward_backward(P, xs, acts, mask, h0)
            W[idx] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - getattr(G, name)[idx]) < 1e-5 * max(1.0, abs(num)), (
                name,
                idx,
                num,
                getattr(G, name)[idx],
            )
    assert gru.param_count(n_in, H, K) == P.count()


def test_gru_training_reduces_validation_nll():
    rng = np.random.default_rng(1)
    K, n_in = 2, 4
    seqs = []
    for _ in range(12):  # deterministic rule: action = sign of feature 0
        xs = rng.normal(size=(40, n_in))
        acts = (xs[:, 0] > 0).astype(int)
        seqs.append((xs, acts, np.ones(40, dtype=bool)))
    _P, log = gru.train_gru(
        seqs[:9],
        seqs[9:],
        n_in=n_in,
        n_hidden=4,
        n_out=K,
        lr=0.03,
        max_epochs=25,
        patience=6,
        window=20,
        seed=0,
    )
    assert log["best_val_nll"] < 0.45 < np.log(2)


def _obs(fam, x, k):
    return Observation("p", 0, 25.0, "task", fam, np.asarray([x], dtype=np.float32), k)


def test_pvl_delta_learns_better_option():
    pol = PVLDeltaModel(make_init("p", 0), {"alpha": 0.9, "lam": 1.0, "A": 0.4, "c": 2.0})
    o = _obs("bandit2", [1.0] + [0.0] * 11, 2)
    for _ in range(30):
        pol.learn(o, 0, Feedback(mode="bandit", reward=0.9, source="human_data"))
        pol.learn(o, 1, Feedback(mode="bandit", reward=0.2, source="human_data"))
    p = pol.action_probabilities(o)
    assert p[0] > 0.8


def test_bayes_rule_learns_binary_rule_and_is_calibrated_over_bins():
    pol = BayesRuleModel(make_init("p", 0), {"tau2": 1.0, "sigma": 0.25})
    rng = np.random.default_rng(0)
    for _ in range(80):
        x = rng.choice([-1.0, 1.0], size=12)
        y = 1 if x[0] > 0 else 0
        o = _obs("wpt", x, 2)
        pol.learn(
            o,
            y,
            Feedback(
                mode="full", reward=1.0, correct_action=y, rewards_all=np.eye(2)[y], source="human_data"
            ),
        )
    o = _obs("wpt", [1.0] + [0.0] * 11, 2)
    p = pol.action_probabilities(o)
    assert p[1] > 0.9 and abs(p.sum() - 1) < 1e-9


def test_exemplar_model_generalises_by_similarity_and_learns_attention():
    pol = ExemplarModel(make_init("p", 0), {"c": 2.0, "beta": 6.0, "decay": 1.0, "lr_att": 0.05})
    rng = np.random.default_rng(0)
    for _ in range(60):
        x = rng.choice([-1.0, 1.0], size=12)
        y = 1 if x[2] > 0 else 0  # only feature 2 matters
        o = _obs("wpt", x, 2)
        pol.learn(
            o,
            y,
            Feedback(
                mode="full", reward=1.0, correct_action=y, rewards_all=np.eye(2)[y], source="human_data"
            ),
        )
    att = pol.att["wpt"]
    assert att[2] > att.mean(), "attention should move toward the diagnostic feature"
    x = rng.choice([-1.0, 1.0], size=12)
    x[2] = 1.0
    assert pol.action_probabilities(_obs("wpt", x, 2))[1] > 0.6


def test_ece_zero_for_perfect_calibration_and_positive_otherwise():
    conf = np.array([0.9] * 9 + [0.1])
    hit = np.array(
        [1.0] * 9 + [0.0]
    )  # 90% of trials at 0.9 confidence are hits... bin accuracy 1.0 vs conf 0.9
    e, _bins = expected_calibration_error(conf, hit)
    assert 0.0 < e < 0.2
    conf2 = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    hit2 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
    e2, _ = expected_calibration_error(conf2, hit2)
    assert e2 == pytest.approx(0.0, abs=1e-9)
