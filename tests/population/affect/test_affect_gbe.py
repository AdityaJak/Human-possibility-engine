"""Affect lane (Gate F): GBE adapter, partition lock, causal control, model conformance — synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hpe.evaluation.human import partitions
from hpe.population.affect import gbe
from hpe.population.core.encoding import F, Layout


def _play(rng: np.random.Generator) -> np.ndarray:
    """A README-consistent 30x15 play: option values, choice, outcome, ratings before trial 1 and every 2-3."""
    p = np.full((30, 15), np.nan)
    p[:, 0] = np.arange(1, 31)
    p[:, 1] = 1
    kinds = rng.permutation([0] * 11 + [1] * 8 + [2] * 11)
    for t, k in enumerate(kinds):
        if k == 0:  # gain
            safe, win, loss = rng.choice([35, 45, 55]), rng.choice([59, 85, 150]), 0
        elif k == 1:  # mixed
            safe, win, loss = 0, rng.choice([40, 55, 75]), -rng.choice([10, 46, 90])
        else:  # loss
            safe, win, loss = -rng.choice([35, 45, 55]), 0, -rng.choice([59, 85, 150])
        ch = int(rng.integers(0, 2))
        p[t, 2:5] = (safe, win, loss)
        p[t, 6] = ch
        p[t, 7] = safe if ch == 0 else (win if rng.random() < 0.5 else loss)
    rated = [0]
    t = 0
    while True:
        t += int(rng.integers(2, 4))
        if t > 29:
            break
        rated.append(t)
    rated[-1] = 29
    for t in rated:
        p[t, 9] = float(rng.integers(0, 101))
    return p


def _participant(pid: int, n_plays: int, seed: int = 0) -> gbe.GBEParticipant:
    rng = np.random.default_rng(seed + pid)
    return gbe.GBEParticipant(pid, [_play(rng) for _ in range(n_plays)], [5] * n_plays)


def _seqs(n: int, plays: int = 1) -> list:
    out = []
    for i in range(n):
        s, notes = gbe.encode_participant(_participant(i + 1, plays + (i % 2)))
        assert s is not None and not notes
        out.append(s)
    return out


def test_encoding_layout_and_affect_timing() -> None:
    p = _participant(7, 2)
    s, notes = gbe.encode_participant(p)
    assert s is not None and not notes
    L = Layout(gbe.K)
    assert s.x.shape == (60, gbe.D_BASE + gbe.N_EXTRA)
    assert s.x[0, L.block_start] == 1 and s.x[30, L.block_start] == 1 and s.x[1:30, L.block_start].sum() == 0
    assert np.array_equal(s.actions, np.concatenate([pl[:, 6] for pl in p.plays]).astype(int))
    assert np.all(s.scored) and np.all(s.avail)
    e = s.x[:, gbe.D_BASE :]
    for pi, play in enumerate(p.plays):
        for tp in range(30):
            t = 30 * pi + tp
            avail = [u for u in range(tp) if u >= 1 and np.isfinite(play[u, 9])]
            expect = play[avail[-1], 9] if avail else play[0, 9]
            assert e[t, 0] == pytest.approx(expect / 100)
    # previous action / outcome of row t come from row t-1 within and across plays
    for t in range(1, 60):
        a_prev = s.actions[t - 1]
        assert s.x[t, F + gbe.K + a_prev] == 1.0
        assert s.x[t, L.prev_observed.start + 1] == float(a_prev == 1)


def test_current_rating_never_reaches_its_own_row() -> None:
    p = _participant(3, 1)
    s0, _ = gbe.encode_participant(p)
    assert s0 is not None
    for t in range(1, 29):
        q = gbe.GBEParticipant(p.pid, [p.plays[0].copy()], [5])
        q.plays[0][t, 9] = 100.0 if not np.isfinite(p.plays[0][t, 9]) or p.plays[0][t, 9] != 100 else 0.0
        s1, _ = gbe.encode_participant(q)
        assert s1 is not None
        assert np.array_equal(s0.x[: t + 1], s1.x[: t + 1]), t


def test_check_play_rejects_inconsistent_outcome() -> None:
    p = _play(np.random.default_rng(1))
    assert gbe.check_play(p) is None
    bad = p.copy()
    i = int(np.flatnonzero(bad[:, 6] == 0)[0])
    bad[i, 7] = bad[i, 2] + 1
    assert gbe.check_play(bad) is not None
    bad = p.copy()
    bad[0, 9] = np.nan
    assert gbe.check_play(bad) == "no pre-play rating"


def test_partition_is_deterministic_disjoint_and_tamper_evident(tmp_path: Path) -> None:
    ids = list(range(1, 1001))
    a, b = gbe.make_partition(ids), gbe.make_partition(reversed(ids))
    assert a["dev"] == b["dev"] and a["test"] == b["test"]
    assert a["counts"] == {"dev": 400, "tune": 200, "test": 400}
    assert not (set(a["dev"]) & set(a["test"])) and not (set(a["tune"]) & set(a["test"]))
    f = tmp_path / "p.json"
    f.write_text(json.dumps(a))
    assert len(gbe.load_partition(f)) == 1000
    a["test"] = a["test"][1:] + a["dev"][:1]
    f.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="digest"):
        gbe.load_partition(f)


def test_committed_partition_matches_its_digest() -> None:
    assign = gbe.load_partition()
    counts = {p: sum(1 for v in assign.values() if v == p) for p in ("dev", "tune", "test")}
    assert counts == {"dev": 18827, "tune": 9413, "test": 18827}


def test_test_partition_is_locked_before_any_data_access() -> None:
    with pytest.raises(partitions.TestPartitionLocked):
        gbe.load_gbe(["dev", "test"], experiment_id="no_such_freeze", path=Path("/nonexistent.mat"))


torch = pytest.importorskip("torch")


def test_past_shuffle_is_causal_and_draws_only_earlier_ratings() -> None:
    from hpe.population.affect.models import past_shuffle

    s = _seqs(1, plays=3)[0]
    e = s.x[:, gbe.D_BASE :]
    out = past_shuffle(e, s.key)
    events = np.flatnonzero(e[:, 1] > 0.5)
    assert out[events[0], 0] == e[events[0], 0]
    for k, t in enumerate(events[1:], start=1):
        assert out[t, 0] in set(e[events[:k], 0].tolist())
    assert np.array_equal(out[:, 1:], e[:, 1:])
    rng = np.random.default_rng(0)
    for t in (5, 40, 70):
        pert = e.copy()
        pert[t + 1 :] = rng.uniform(size=pert[t + 1 :].shape)
        assert np.array_equal(past_shuffle(pert, s.key)[: t + 1], out[: t + 1])


def test_permute_ratings_keeps_values_and_positions() -> None:
    from hpe.population.affect.models import permute_ratings

    r = np.array([50.0, np.nan, np.nan, 60.0, np.nan, 10.0, 90.0])
    out = permute_ratings(r, "k")
    assert np.array_equal(np.isfinite(out), np.isfinite(r))
    assert sorted(out[np.isfinite(out)]) == sorted(r[np.isfinite(r)])


def test_aux_targets_follow_the_rated_outcome() -> None:
    from hpe.population.affect.models import AffectAuxGRU

    s = _seqs(1, plays=2)[0]
    r = s.meta["ratings"]
    tgt = AffectAuxGRU.aligned_targets(s, r)
    assert tgt[0] == pytest.approx(r[0] / 100) and tgt[30] == pytest.approx(r[30] / 100)
    for t in range(60):
        if t % 30 not in (0, 29) and np.isfinite(r[t]):
            assert tgt[t + 1] == pytest.approx(r[t] / 100)
    assert np.isfinite(tgt).sum() == np.isfinite(r).sum() - 2  # each play's last rating has no next row


@pytest.mark.parametrize(
    "name", ["affect_gru_base", "affect_gru_obs", "affect_gru_shuf", "affect_gru_aux", "affect_gru_aux_shuf"]
)
def test_models_register_fit_and_are_causal(name: str) -> None:
    from hpe.population.core.conformance import check_causal
    from hpe.population.core.interface import get_model

    spec = get_model(name)
    assert spec.owner == "affect"
    train = _seqs(6)
    m = spec.factory({"hidden": 8, "max_epochs": 2, "batch_size": 4}, gbe.K, gbe.D_BASE)
    log = m.fit(train, seed=1, threads=1, budget_s=30)
    assert log["epochs"] >= 1
    probe = _seqs(1, plays=2)[0]
    lp = m.predict_logprobs(probe)
    assert lp.shape == (probe.T, gbe.K) and np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-5)
    assert check_causal(m, probe) == []


def test_mechanism_and_control_have_identical_capacity() -> None:
    from hpe.population.core.interface import get_model

    cfg = {"hidden": 16}
    obs = get_model("affect_gru_obs").factory(cfg, gbe.K, gbe.D_BASE)
    shuf = get_model("affect_gru_shuf").factory(cfg, gbe.K, gbe.D_BASE)
    base = get_model("affect_gru_base").factory(cfg, gbe.K, gbe.D_BASE)
    assert obs.param_count() == shuf.param_count() == base.param_count() + 3 * 16 * gbe.N_EXTRA
    aux = get_model("affect_gru_aux").factory(cfg, gbe.K, gbe.D_BASE)
    aux_s = get_model("affect_gru_aux_shuf").factory(cfg, gbe.K, gbe.D_BASE)
    assert aux.param_count() == aux_s.param_count()
