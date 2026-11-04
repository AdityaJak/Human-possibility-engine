"""Social adapter (Najar et al. 2020 layout): label flips, causal social columns, causal control — synthetic rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hpe.evaluation.human import partitions
from hpe.population.affect import najar
from hpe.population.core.encoding import Layout


def _rows(expid: str = "7", sessions: int = 2, seed: int = 0) -> list[dict[str, str]]:
    """Interleaved own / demonstrator rows: per session 3 states, the first private, the others observed."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    trial = 0
    for sess in range(1, sessions + 1):
        states = [3 * sess - 2, 3 * sess - 1, 3 * sess]
        plan = [(s, False) for s in states for _ in range(6)] + [
            (s, True) for s in states[1:] for _ in range(6)
        ]
        for i in rng.permutation(len(plan)):
            st, dem = plan[i]
            trial += 1
            good = int(rng.random() < 0.7)
            out = "1" if rng.random() < (0.7 if good else 0.3) else "-1"
            rows.append(
                {
                    "expid": expid,
                    "id": "dem" if dem else expid,
                    "session": str(sess),
                    "trial": str(trial),
                    "state": str(st),
                    "cgb": str(good),
                    "out": "NA" if dem else out,
                }
            )
    return rows


def test_encoding_shapes_and_labels() -> None:
    rows = _rows()
    s = najar.encode_participant("najar2020_imitation#exp1_7", rows)
    own = [r for r in rows if r["id"] != "dem"]
    assert s.T == len(own) and s.x.shape == (len(own), najar.D_BASE + 2 * najar.N_EXTRA)
    assert np.all(s.scored)
    # the label flip is a fixed per-state relabelling: within a state, action == cgb or action == 1 - cgb throughout
    for st in {int(r["state"]) for r in own}:
        idx = [i for i, r in enumerate(own) if int(r["state"]) == st]
        diff = {int(own[i]["cgb"]) ^ int(s.actions[i]) for i in idx}
        assert len(diff) == 1
    L = Layout(najar.K)
    assert s.x[0, L.block_start] == 1.0


def test_social_columns_use_only_earlier_demonstrations() -> None:
    rows = _rows(seed=3)
    key = "najar2020_imitation#exp1_7"
    s0 = najar.encode_participant(key, rows)
    own_idx = [i for i, r in enumerate(rows) if r["id"] != "dem"]
    for t, i in enumerate(own_idx[5:40:7], start=0):
        changed = [dict(r) for r in rows]
        for j in range(i + 1, len(changed)):  # change every later demonstration and own choice
            changed[j]["cgb"] = str(1 - int(changed[j]["cgb"]))
        s1 = najar.encode_participant(key, changed)
        tpos = own_idx.index(i)
        assert np.array_equal(s0.x[: tpos + 1], s1.x[: tpos + 1]), t


def test_observed_columns_track_demonstrations() -> None:
    rows = _rows(seed=5)
    s = najar.encode_participant("najar2020_imitation#exp1_7", rows)
    obs = s.x[:, najar.D_BASE : najar.D_BASE + najar.N_EXTRA]
    own = [r for r in rows if r["id"] != "dem"]
    private = {1, 4}  # first state of each session has no demonstrations
    for t, r in enumerate(own):
        if int(r["state"]) in private:
            assert obs[t, 0] == 0 and obs[t, 3] == 0.5
    assert obs[:, 0].sum() > 0


def test_shuffle_events_is_causal_and_resamples_earlier_actions() -> None:
    ev = [(i * 2, i % 3, int(a)) for i, a in enumerate(np.random.default_rng(0).integers(0, 2, 30))]
    out = najar.shuffle_events(ev, "k")
    assert out[0] == ev[0]
    assert [e[:2] for e in out] == [e[:2] for e in ev]
    for k in (5, 17):
        pert = ev[:k] + [(p, s, 1 - a) for p, s, a in ev[k:]]
        assert najar.shuffle_events(pert, "k")[:k] == out[:k]


def test_partition_and_lock(tmp_path: Path) -> None:
    keys = [f"{najar.DATASET}#exp1_{i}" for i in range(68)]
    rec = najar.make_partition(keys)
    assert rec["counts"] == {"dev": 27, "tune": 14, "test": 27}
    f = tmp_path / "p.json"
    f.write_text(json.dumps(rec))
    assert len(najar.load_partition(f)) == 68
    assert najar.load_partition()  # the committed file verifies
    with pytest.raises(partitions.TestPartitionLocked):
        najar.load_najar(["test"], experiment_id="no_such_freeze", folder=tmp_path)


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("name", ["social_gru_base", "social_gru_obs", "social_gru_shuf"])
def test_social_models_are_causal(name: str) -> None:
    from hpe.population.core.conformance import check_causal
    from hpe.population.core.interface import get_model

    seqs = [
        najar.encode_participant(f"najar2020_imitation#exp1_{i}", _rows(str(i), seed=i)) for i in range(5)
    ]
    m = get_model(name).factory({"hidden": 8, "max_epochs": 2, "batch_size": 2}, najar.K, najar.D_BASE)
    m.fit(seqs[:4], seed=0, threads=1, budget_s=30)
    assert check_causal(m, seqs[4]) == []
    if name != "social_gru_base":
        assert (
            m.param_count()
            == get_model("social_gru_base").factory({"hidden": 8}, 2, najar.D_BASE).param_count()
            + 3 * 8 * najar.N_EXTRA
        )
