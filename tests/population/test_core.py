"""Phase III core: encoding v3 layout, causality check, statistics, donor pairing, the test lock."""

from __future__ import annotations

import os

import numpy as np
import pytest

from hpe.evaluation.human.partitions import TestPartitionLocked
from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, GenericTrial
from hpe.population.core import evaluate
from hpe.population.core.conformance import check_causal
from hpe.population.core.encoding import EncodedSeq, Layout, encode, input_dim
from hpe.population.core.runner import donors_for


def _participant(n: int = 12, seed: int = 0) -> Participant:
    rng = np.random.default_rng(seed)
    p = Participant("toy/exp1.csv", str(seed), "toy", 3)
    for t in range(n):
        forced = t < 2
        c = int(rng.integers(0, 3))
        mode = "full" if t % 3 == 0 else "bandit"
        ra = tuple(float(v) for v in rng.uniform(0, 1, 3)) if mode == "full" else None
        feats = tuple(float(v) for v in rng.uniform(-1, 1, N_FEATURES))
        p.trials.append(
            GenericTrial(c, 3, feats, mode, float(rng.uniform()), None, ra, forced=forced, block=t // 6)
        )
    return p


def test_encoding_layout_places_previous_feedback_and_instructions():
    p = _participant()
    s = encode(p, "toy", 4)
    L = Layout(4)
    assert s.x.shape == (12, input_dim(4)) and input_dim(4) == 2 * N_FEATURES + 5 * 4 + 4
    assert s.avail[:, :3].all() and not s.avail[:, 3].any()
    assert not s.scored[:2].all() and s.scored[2:].all()
    for t in range(1, 12):
        prev = p.trials[t - 1]
        assert s.x[t, L.prev_action.start + prev.choice] == 1.0
        assert np.allclose(s.x[t, L.prev_feat], prev.features)
        if prev.mode == "full":
            assert np.allclose(s.x[t, L.prev_outcomes][:3], prev.rewards_all)
            assert s.x[t, L.prev_observed][:3].all()
        else:
            assert s.x[t, L.prev_observed].sum() == 1.0
            assert s.x[t, L.prev_outcomes][prev.choice] == pytest.approx(prev.reward)
    assert s.x[0, L.prev_action].sum() == 0.0  # nothing before the first trial
    assert s.x[0, L.instructed] == 1.0 and s.x[0, L.instructed_action.start + p.trials[0].choice] == 1.0
    assert s.x[2, L.instructed] == 0.0 and s.x[2, L.instructed_action].sum() == 0.0
    assert s.x[6, L.block_start] == 1.0 and s.x[7, L.block_start] == 0.0


class _Honest:
    name, adaptive = "honest", True

    def predict_logprobs(self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None):
        # running choice frequencies of past actions: causal by construction
        counts = np.ones(seq.k_max)
        out = np.zeros((seq.T, seq.k_max))
        for t in range(seq.T):
            p = np.where(seq.avail[t], counts, 0.0)
            out[t] = np.log(np.where(seq.avail[t], p / p.sum(), 1e-300))
            out[t][~seq.avail[t]] = -np.inf
            if mode == "adapt":
                counts[seq.actions[t]] += 1.0
        return out


class _Cheat(_Honest):
    name = "cheat"

    def predict_logprobs(self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None):
        out = super().predict_logprobs(seq, mode=mode)
        out[:, :] = np.where(seq.avail, -5.0, -np.inf)
        out[np.arange(seq.T), seq.actions] = -0.01  # peeks at the action it is asked to predict
        return out


def test_causality_check_passes_honest_and_catches_lookahead():
    s = encode(_participant(40, 3), "toy", 4)
    assert check_causal(_Honest(), s) == []  # type: ignore[arg-type]
    problems = check_causal(_Cheat(), s)  # type: ignore[arg-type]
    assert problems and any("depend on data at or after" in p for p in problems)


def test_lapse_score_and_quarters():
    s = encode(_participant(20, 5), "toy", 4)
    lp = _Honest().predict_logprobs(s)
    a = evaluate.score(s, lp, 0.0)
    b = evaluate.score(s, lp, 0.3)
    assert a.n == s.n_scored and len(a.quarters) == 4
    assert np.isfinite(a.nll) and np.isfinite(b.nll) and a.nll != b.nll
    mixed = evaluate.apply_lapse(lp, s.avail, 0.3)
    assert np.allclose(np.exp(mixed[s.avail[:, :]].reshape(s.T, 3)).sum(axis=1), 1.0)


def test_holm_and_bootstrap_direction():
    adj = evaluate.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert (
        adj["a"] == pytest.approx(0.03)
        and adj["c"] == pytest.approx(0.06)
        and adj["b"] == pytest.approx(0.06)
    )
    est, _lo, hi, p = evaluate.bootstrap(
        np.full(50, -0.05) + np.linspace(-0.01, 0.01, 50), seed=1, key=("t",)
    )
    assert est < 0 and hi < 0 and p <= 1 / 2000 + 1e-12


def test_donors_are_a_derangement():
    seqs = [encode(_participant(8, i), "toy", 4) for i in range(7)]
    for i, s in enumerate(seqs):
        s.key = f"toy#{i}"
    d = donors_for(seqs, 3, "toy")
    assert set(d) == {s.key for s in seqs}
    assert all(d[k].key != k for k in d)


def test_test_partition_is_locked_without_a_freeze_record(tmp_path):
    from hpe.population.core import data

    os.environ.setdefault(data.DATA_ENV, str(tmp_path / "missing.jsonl"))
    with pytest.raises(TestPartitionLocked, match="no freeze record"):
        data.load_family("restless4", "test", experiment_id="no_such_freeze")


def test_uncommitted_freeze_record_does_not_unlock(tmp_path):
    """Critic T-1: a freeze record that is not committed (or lacks critic approval) never opens test."""
    import json

    from hpe.population.core.freeze import FREEZE_DIR, validate_freeze

    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    rec = FREEZE_DIR / "zz_uncommitted_probe.json"
    fields = {
        k: "x"
        for k in (
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
    }
    fields["experiment_id"] = "zz_uncommitted_probe"
    rec.write_text(json.dumps(fields))
    try:
        with pytest.raises(TestPartitionLocked, match="not committed"):
            validate_freeze("zz_uncommitted_probe")
    finally:
        rec.unlink()


def test_phase2_tooling_can_no_longer_unlock_test():
    from hpe.evaluation.human.partitions import check_partition_access

    with pytest.raises(TestPartitionLocked, match="Phase II is closed"):
        check_partition_access("test", "anything")


def test_overlapping_family_registration_is_refused():
    from hpe.population.core.families import FAMILIES, FamilySpec, register_family

    spec = FAMILIES["garcia"]
    with pytest.raises(ValueError, match="overlaps"):
        register_family(
            FamilySpec(
                "garcia_alias", spec.prefixes, spec.n_actions, spec.partition_file, spec.parser, "counted"
            )
        )


def test_unpinned_partition_file_is_refused(tmp_path):
    from hpe.population.core import data

    copy_path = tmp_path / "psych101_p3.json"
    copy_path.write_bytes(data.P3_PARTITIONS.read_bytes())
    with pytest.raises(data.DataUnavailable, match="not pinned"):
        data._partitions(str(copy_path))


def test_restless4_parser_on_a_synthetic_transcript():
    from hpe.population.families.restless4 import parse_restless4

    text = (
        "You will be asked to repeatedly choose between four different options labeled L, G, O, and U.\n"
        "You press <<G>> and get 90.0 points.\nYou press <<U>> and get 24.0 points.\n"
    )
    p = parse_restless4(text, "bahrami2020four/exp.csv", "9")
    assert [t.choice for t in p.trials] == [1, 3]
    assert p.trials[0].reward == pytest.approx(0.9) and p.trials[0].mode == "bandit"
