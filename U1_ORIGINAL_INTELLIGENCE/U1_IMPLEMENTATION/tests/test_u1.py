"""Fast tests for the U1 track (environment validity, inference exactness, information hygiene, analysis)."""

from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
import pytest
from u1 import env
from u1.baselines import EntityBayes, RandomPairs, UCBPairs
from u1.env import (
    CONDITIONS,
    N_ACT,
    N_ENT,
    NOVEL_ROLE,
    PAIRS,
    SWITCH_T,
    Episode,
    make_universe,
    make_world,
    reacting,
)
from u1.ira import IRA, IRAConfig, Library, _space
from u1.play import play
from u1.recluster import WorldRecord, recluster
from u1.stats import contrast, holm


@pytest.fixture(scope="module")
def uni() -> env.Universe:
    return make_universe(0)


def test_universe_constraints() -> None:
    for s in range(12):
        u = make_universe(s)
        assert (u.law == u.law.T).all()
        assert len({tuple(r) for r in u.sig}) == 8 and len({tuple(r) for r in u.code}) == 8
        assert all(u.law[r, :5].any() for r in range(5))
        for sp in (NOVEL_ROLE, *env.EXTRA_ROLES):
            assert not u.law[sp, sp] and 1 <= u.law[sp, :5].sum() <= 2
        assert env.feasible(u.law)
        for (
            cond
        ) in CONDITIONS:  # R1: every condition must be samplable in every universe (attempt 1 failed this)
            make_world(u, cond, 0)


@pytest.mark.parametrize("cond", CONDITIONS)
def test_world_conditions(uni: env.Universe, cond: str) -> None:
    for s in range(30):
        w = make_world(uni, cond, s)
        r0 = w.roles_at(0)
        assert 1 <= len(reacting(uni, r0)) <= 2
        distinct = len(set(r0.tolist()) - {NOVEL_ROLE, *env.EXTRA_ROLES})
        if cond in ("train", "iid", "cue_reversed", "role_switch"):
            assert 2 <= distinct <= 3 and NOVEL_ROLE not in r0
        if cond == "recombined":
            assert distinct >= 4
        if cond == "novel_role":
            assert NOVEL_ROLE in r0
            assert all(NOVEL_ROLE in (r0[PAIRS[p][0]], r0[PAIRS[p][1]]) for p in reacting(uni, r0))
        if cond == "role_switch":
            after = w.roles_at(SWITCH_T)
            before = set(reacting(uni, r0))
            assert w.switch is not None and not before & set(reacting(uni, after)) and reacting(uni, after)
        if cond == "train_broad":
            assert 0.3 <= w.meta["rho"] <= 1.0
            assert NOVEL_ROLE not in r0
        else:
            assert w.meta["rho"] == (0.1 if cond == "cue_reversed" else 0.9)
            assert not set(env.EXTRA_ROLES) & set(r0.tolist())


def test_world_determinism_and_common_random_numbers(uni: env.Universe) -> None:
    a, b = make_world(uni, "iid", 7), make_world(uni, "iid", 7)
    assert (a.roles == b.roles).all() and np.allclose(a.x, b.x)
    e1, e2 = Episode(uni, a, 7), Episode(uni, b, 7)
    assert all(e1.step(t, k) == e2.step(t, k) for t in range(20) for k in range(N_ACT))


def test_cue_validity(uni: env.Universe) -> None:
    agree = {0.9: [], 0.1: []}
    for cond in ("iid", "cue_reversed"):
        for s in range(200):
            w = make_world(uni, cond, s)
            for e, r in enumerate(w.roles):
                agree[w.meta["rho"]].extend(((w.x[e, :3] > 0).astype(int) == uni.code[r]).tolist())
    assert 0.85 < np.mean(agree[0.9]) < 0.95 and 0.05 < np.mean(agree[0.1]) < 0.15


def test_canonical_enumeration_counts() -> None:
    digits, law_pair, law_bit, logperm = _space(3, 2)
    # brute force: label sequences where fresh labels appear in order of first use
    count = 0
    for z in itertools.product(range(5), repeat=N_ENT):
        fresh = [v for v in z if v >= 3]
        firsts = list(dict.fromkeys(fresh))
        count += firsts == sorted(firsts) and (not firsts or firsts[0] == 3)
    assert len(digits) == count
    assert law_pair.shape == (count, len(PAIRS)) and law_bit.shape == (count, N_ENT, 3)
    assert math.isclose(float(logperm.max()), math.log(2))


def test_agents_receive_no_hidden_state() -> None:
    for cls in (IRA, UCBPairs, EntityBayes, RandomPairs):
        for meth in ("begin", "act", "observe"):
            params = list(inspect.signature(getattr(cls, meth)).parameters)
            assert params[1:] in (["x"], ["t"], ["t", "a", "o"])


def test_ira_posterior_matches_brute_force(uni: env.Universe) -> None:
    """With an empty library the exact posterior over partitions must match direct computation."""
    ag = IRA(Library(), IRAConfig(beta=1.0, change=False), np.random.default_rng(0))
    w = make_world(uni, "iid", 3)
    ag.begin(w.x)
    obs = [(0, 5), (0, 5), (1, 3), (5, 1), (6, 0), (9, 0)]
    for t, (a, o) in enumerate(obs):
        ag.observe(t, a, o)
    pz = ag.posterior()
    # brute force: log marginal likelihood per canonical assignment, Beta-Bernoulli collapsed
    rb = 0.5
    logp = []
    for z in ag.digits:
        lp = float(ag.logprior[list(ag.digits.tolist()).index(z.tolist())])
        lp += float(np.log(np.exp(ag.logapp[list(ag.digits.tolist()).index(z.tolist())]).mean()))
        counts: dict[tuple, list[int]] = {}
        for a, o in obs:
            if a >= N_ENT:
                i, j = PAIRS[a - N_ENT]
                key = ("p", min(z[i], z[j]), max(z[i], z[j]))
                counts.setdefault(key, [0, 0])[0 if o else 1] += 1
            else:
                for bit in range(3):
                    key = ("b", z[a], bit)
                    counts.setdefault(key, [0, 0])[0 if (o >> bit) & 1 else 1] += 1
        for key, (s, f) in counts.items():
            a0, b0 = (rb, 1 - rb) if key[0] == "p" else (0.5, 0.5)
            lp += math.lgamma(a0 + s) + math.lgamma(b0 + f) - math.lgamma(a0 + b0 + s + f)
            lp -= math.lgamma(a0) + math.lgamma(b0) - math.lgamma(a0 + b0)
        logp.append(lp)
    ref = np.exp(np.array(logp) - max(logp))
    ref /= ref.sum()
    assert np.allclose(pz, ref, atol=1e-9)


def test_ira_plays_legally_and_beats_random(uni: env.Universe) -> None:
    ira = [
        play(IRA(Library(), IRAConfig(), np.random.default_rng(s)), uni, "iid", s)["norm"] for s in range(15)
    ]
    rnd = [play(RandomPairs(np.random.default_rng(s)), uni, "iid", s)["norm"] for s in range(15)]
    assert np.mean(ira) > np.mean(rnd) + 0.2


def test_recluster_recovers_roles_from_clean_records(uni: env.Universe) -> None:
    rng = np.random.default_rng(0)
    records, truth = [], []
    for s in range(60):
        w = make_world(uni, "train", 100000 + s)
        obs = []
        for e in range(N_ENT):
            bits = (rng.random(3) < uni.bit_probs(int(w.roles[e]))).astype(int)
            obs.append((e, int(bits[0] + 2 * bits[1] + 4 * bits[2])))
        records.append(WorldRecord(w.x, obs))
        truth.append(w.roles)
    _lib, assign = recluster(records, np.random.default_rng(1), sweeps=20)
    pairs = {}
    for (wi, e), c in assign.items():
        pairs.setdefault(c, []).append(int(truth[wi][e]))
    purity = sum(max(np.bincount(v)) for v in pairs.values()) / sum(len(v) for v in pairs.values())
    assert purity > 0.9


def test_holm_and_universe_level_contrast() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = {(u, i): 1.0 + 0.01 * (i % 3) + 0.02 * u for u in range(5) for i in range(100)}
    b = {(u, i): 0.5 for u in range(5) for i in range(100)}
    r = contrast(a, b)
    assert r["estimate"] > 0.49 and r["p_universe_t"] < 1e-3 and r["n_universes_positive"] == 5


def test_expected_entropy_closed_form() -> None:
    from u1.ira import _expected_hb

    rng = np.random.default_rng(0)
    for a, b in ((0.5, 0.5), (1.0, 1.0), (3.0, 7.0), (40.0, 2.0)):
        th = np.clip(rng.beta(a, b, size=400000), 1e-12, 1 - 1e-12)
        mc = float(np.mean(-(th * np.log(th) + (1 - th) * np.log(1 - th))))
        assert abs(float(_expected_hb(np.array(a), np.array(b))) - mc) < 3e-3


def test_ira_probes_when_identity_is_uncertain(uni: env.Universe) -> None:
    """R1 (critique F1): with law uncertainty in the information gain, an empty-library IRA must probe."""
    probes = [
        play(IRA(Library(), IRAConfig(beta=1.0), np.random.default_rng(s)), uni, "iid", s)["probes"]
        for s in range(8)
    ]
    assert sum(probes) > 0


def test_rbar_is_pair_level_not_outcome_pooled() -> None:
    lib = Library()
    lib.react = {(0, 0): [90.0, 10.0], (0, 1): [0.0, 3.0], (1, 1): [0.0, 3.0]}
    lib.pooled = [90.0, 16.0]
    assert lib.rbar() < 0.5 < lib.pooled[0] / sum(lib.pooled)


def test_recluster_with_pair_records(uni: env.Universe) -> None:
    """Records with probes AND pair outcomes, as produced in real training (critique F14)."""
    rng = np.random.default_rng(0)
    records, truth = [], []
    for s in range(80):
        w = make_world(uni, "train", 100000 + s)
        obs = []
        for e in range(N_ENT):
            bits = (rng.random(3) < uni.bit_probs(int(w.roles[e]))).astype(int)
            obs.append((e, int(bits[0] + 2 * bits[1] + 4 * bits[2])))
        for p, (i, j) in enumerate(PAIRS):
            obs.append((N_ENT + p, int(rng.random() < uni.pair_prob(int(w.roles[i]), int(w.roles[j])))))
        records.append(WorldRecord(w.x, obs))
        truth.append(w.roles)
    _lib, assign = recluster(records, np.random.default_rng(1), sweeps=20)
    pairs: dict[int, list[int]] = {}
    for (wi, e), c in assign.items():
        pairs.setdefault(c, []).append(int(truth[wi][e]))
    purity = sum(max(np.bincount(v)) for v in pairs.values()) / sum(len(v) for v in pairs.values())
    assert purity > 0.9


@pytest.mark.parametrize("arch", ["gru", "rel"])
def test_metarl_shapes_and_equivariance(uni: env.Universe, arch: str) -> None:
    import torch
    from u1.metarl import TrainWorlds, make_net, observe_fn

    torch.manual_seed(0)
    net = make_net(arch, 16)
    x = np.random.default_rng(0).normal(size=(2, N_ENT, 4)).astype(np.float32)
    obs = observe_fn(arch)(x, np.array([7, 2]), np.array([1, 5]), np.array([1.0, 0.0], dtype=np.float32), 3)
    logits, v, _h = net(torch.from_numpy(obs), net.init_state(2))
    assert logits.shape == (2, N_ACT) and v.shape == (2,)
    if arch == "rel":  # permuting entities permutes probe logits
        perm = np.array([1, 0, 2, 3, 4])
        obs_p = observe_fn(arch)(x[:, perm], np.array([-1, -1]), np.array([0, 0]), np.zeros(2, np.float32), 0)
        obs_0 = observe_fn(arch)(x, np.array([-1, -1]), np.array([0, 0]), np.zeros(2, np.float32), 0)
        l0, _, _ = net(torch.from_numpy(obs_0), net.init_state(2))
        lp, _, _ = net(torch.from_numpy(obs_p), net.init_state(2))
        assert torch.allclose(l0[:, :N_ENT][:, perm], lp[:, :N_ENT], atol=1e-5)
    tw = TrainWorlds(uni, "train_broad", np.random.default_rng(0), pool=50)
    xs, pair_p, bit_p, _t_sw = tw.sample(4)
    assert (
        xs.shape == (4, N_ENT, 4) and pair_p.shape == (2, 4, len(PAIRS)) and bit_p.shape == (2, 4, N_ENT, 3)
    )


def test_stats_refuse_incomplete(tmp_path) -> None:
    from u1.stats import IncompleteResults, load

    with pytest.raises(IncompleteResults):
        load(tmp_path, "confirm", "ira", "iid")


def test_every_agent_builds_through_the_runner_and_plays(tmp_path, uni: env.Universe) -> None:
    """R1 review item 1: every name the confirmatory runner uses must build via make_agent and play a world."""
    import json

    import torch
    from u1 import experiment as ex
    from u1.metarl import make_net

    d = tmp_path / "u0"
    d.mkdir()
    (d / "ira_library.json").write_text(json.dumps(Library().to_json()))
    for name, arch in (("gru_narrow", "gru"), ("gru_broad", "gru"), ("rel_broad", "rel")):
        torch.save(make_net(arch, 8).state_dict(), d / f"metarl_{name}.pt")
        (d / f"metarl_{name}.json").write_text(json.dumps({"config": {"arch": arch, "hidden": 8}}))
    for name in [*ex.IRA_VARIANTS, *ex.METARL_AGENTS, "entity_bayes", "ucb", "random"]:
        m = play(ex.make_agent(name, 0, 1, tmp_path, {}), uni, "iid", 1)
        assert 0.0 <= m["norm"] <= 1.0 + 1e-9
    assert ex.ira_cfg("ira_noactive", 0.5).beta == 0.0
    assert ex.ira_cfg("ira", 0.5).beta == 0.5 and ex.ira_cfg("ira_beta1", 0.5).beta == 1.0


def test_broad_training_holds_out_the_test_switch_time(uni: env.Universe) -> None:
    times = {make_world(uni, "train_broad", s).switch for s in range(400)}
    ts = {sw[0] for sw in times if sw is not None}
    assert ts and env.SWITCH_T not in ts
