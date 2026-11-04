"""Gate E suite: generator structure and seed guard, metrics, reference learners, learnability check."""

from __future__ import annotations

import numpy as np
import pytest

from hpe.core.rng import stream
from hpe.evaluation.human.partitions import TestPartitionLocked as PartitionLocked
from hpe.population.continual import learners as L
from hpe.population.continual import suite as S


def test_generator_matches_the_a3_2_design_on_every_development_seed():
    for seed in S.DEV_SEEDS:
        s = S.generate(seed)
        assert s.T == 300 and s.K == 4 and s.probs.shape == (3, 4)
        best = s.probs.argmax(axis=1)
        assert len(set(best.tolist())) == 3  # distinct best options (impl choice, documented)
        for c in range(3):
            assert s.probs[c, best[c]] == pytest.approx(0.8)
            others = np.delete(s.probs[c], best[c])
            assert np.all((others >= 0.15) & (others <= 0.45))
        assert s.blocks[0][0] == 0 and s.blocks[-1][1] == 300
        for (_s0, e0, c0), (s1, _e1, c1) in zip(s.blocks, s.blocks[1:], strict=False):
            assert e0 == s1 and c0 != c1
        assert all(20 <= e - b <= 40 for b, e, _ in s.blocks)
        assert {c for *_, c in s.blocks} == {0, 1, 2}
        assert len(s.returns()) >= 2
        assert np.array_equal(
            s.contexts, np.repeat([c for *_, c in s.blocks], [e - b for b, e, _ in s.blocks])
        )
        assert set(np.unique(s.rewards).tolist()) <= {0.0, 1.0}


def test_generator_is_deterministic_and_seeds_differ():
    a, b, c = S.generate(7), S.generate(7), S.generate(8)
    assert np.array_equal(a.rewards, b.rewards) and np.array_equal(a.probs, b.probs) and a.blocks == b.blocks
    assert not np.array_equal(a.rewards, c.rewards)


def test_outcome_frequencies_follow_the_reward_probabilities():
    s = S.generate(3)
    p = s.probs[s.contexts]  # [T, K]
    assert abs(float(s.rewards.mean() - p.mean())) < 0.05


@pytest.mark.parametrize("seed", [200, 2999, 3000, 3049, -1])
def test_seeds_outside_development_are_refused_before_any_draw(seed):
    with pytest.raises(S.ReservedSeedError):
        S.generate(seed)


def test_confirmatory_unlock_requires_a_valid_freeze():
    with pytest.raises(PartitionLocked):
        S.generate(3000, confirmatory_unlock="x3_no_such_freeze")


def test_metrics_of_the_oracle_and_of_a_constructed_trajectory():
    s = S.generate(11)
    tr = S.rollout(S.OracleAgent(s), s, stream(0, "t"))
    m = S.sequence_metrics(s, tr)
    assert m["adaptation"] == pytest.approx(1.0)
    assert m["savings"] == pytest.approx(0.0) and m["forgetting"] == pytest.approx(0.0)
    assert m["reward_fraction"] == pytest.approx(1.0)
    # a learner that is optimal except for the first 10 trials of every return block
    acts = tr.actions.copy()
    worst = s.probs.argmin(axis=1)
    for b in s.returns():
        st, _e, c = s.blocks[b]
        acts[st : st + 10] = worst[c]
    exp = s.probs[s.contexts, acts]
    m2 = S.sequence_metrics(s, S.Trajectory(s.seed, acts, s.rewards[np.arange(300), acts], exp))
    drops = [s.probs[c].min() - 0.8 for b in s.returns() for c in [s.blocks[b][2]]]
    assert m2["savings"] == pytest.approx(np.mean(drops))
    assert m2["forgetting"] == pytest.approx(np.mean(drops))
    assert m2["adaptation"] < 1.0


def _mean_metric(agent_fn, seeds, metric, repeats=2):
    seqs = S.generate_many(seeds)
    per = S.evaluate_agent(agent_fn, seqs, seed=1, name="test", repeats=repeats)
    return float(np.mean([v[metric] for v in per.values()]))


def test_learnability_bayes_filter_beats_naive_learners():
    seeds = range(100, 112)
    bayes = _mean_metric(lambda s: S.BayesFilterAgent(256, "greedy"), seeds, "reward_fraction")
    rnd = _mean_metric(lambda s: S.RandomAgent(), seeds, "reward_fraction")
    stat = _mean_metric(lambda s: S.StationaryThompsonAgent(), seeds, "reward_fraction")
    delta = _mean_metric(lambda s: S.DeltaRuleAgent(0.5, 10.0), seeds, "reward_fraction")
    assert bayes > rnd + 0.2
    assert bayes > stat + 0.1
    assert bayes > delta + 0.03
    assert bayes < 1.0


def test_bayes_filter_identifies_the_latent_context():
    s = S.generate(42)
    agent = S.BayesFilterAgent(256, "greedy")
    agent.reset(stream(0, "ctx"))
    canon: dict[int, int] = {}
    hits = []
    for t in range(s.T):
        a = agent.act()
        agent.observe(a, float(s.rewards[t, a]))
        c = int(s.contexts[t])
        canon.setdefault(c, len(canon))  # labels in order of first appearance
        hits.append(int(np.argmax(agent.context_posterior())) == canon[c])
    assert np.mean(hits) > 0.6  # chance is about 1/3; misses are mostly the few trials after each switch


def test_mlp_gradients_match_finite_differences():
    rng = stream(0, "grad")
    net = L.MLP.init(8, 5, 4, rng)
    X, A, R = rng.normal(size=(3, 8)), np.array([0, 2, 3]), np.array([1.0, 0.0, 1.0])
    g = net.grads(X, A, R).mean(axis=0)

    def loss(v):
        n2 = net.copy()
        n2.set_flat(v)
        q = np.array([n2.q(x)[a] for x, a in zip(X, A, strict=True)])
        return float(np.mean(0.5 * (q - R) ** 2))

    v0, eps = net.flat(), 1e-6
    num = np.array([(loss(v0 + eps * e) - loss(v0 - eps * e)) / (2 * eps) for e in np.eye(v0.size)])
    assert np.allclose(g, num, atol=1e-6)


def test_online_learners_are_deterministic_and_reduce_to_fine_tuning():
    s = S.generate(5)
    init = L.scratch_net(0)

    def traj(method, **kw):
        return S.rollout(L.OnlineNetAgent(method, init, lr=0.1, beta=10.0, **kw), s, stream(3, "pol")).actions

    ft = traj("ft")
    assert np.array_equal(ft, traj("ft"))
    assert np.array_equal(ft, traj("ewc", lam=0.0, period=20))  # no penalty -> identical to fine-tuning
    assert np.array_equal(ft, traj("replay", buffer=300, replay_batch=0))  # no replay -> fine-tuning
    assert not np.array_equal(ft, traj("ewc", lam=100.0, period=20))
    assert not np.array_equal(ft, traj("replay", buffer=300, replay_batch=8))


def test_ewc_penalty_keeps_weights_near_the_anchor():
    s = S.generate(9)
    init = L.scratch_net(1)
    drift = {}
    for lam in (0.0, 1000.0):
        ag = L.OnlineNetAgent("ewc", init, lr=0.2, beta=10.0, lam=lam, period=20)
        S.rollout(ag, s, stream(2, "pol"))
        drift[lam] = float(np.linalg.norm(ag.net.flat() - ag.anchor))
    assert drift[1000.0] < drift[0.0]


def test_option_permutation_is_a_relabelling():
    s = S.generate(13)
    perm = np.array([2, 0, 3, 1])
    p = S.permute_options(s, perm)
    assert np.array_equal(p.rewards[:, 0], s.rewards[:, 2]) and np.array_equal(p.contexts, s.contexts)
    assert np.array_equal(perm[p.best], s.best)
