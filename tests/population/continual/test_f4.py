"""F4 (RL²-style meta-learner): trains deterministically, is frozen and causal at test (needs torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hpe.population.continual import f4  # noqa: E402
from hpe.population.continual import suite as S  # noqa: E402


def _train(seed=3):
    train = S.generate_many(range(0, 6))
    val = S.generate_many(range(6, 8))
    return f4.train_f4(train, val, seed=seed, hidden=8, updates=4, batch=4, eval_every=2, val_repeats=1)


def test_training_is_deterministic_and_logged():
    n1, log1 = _train()
    n2, _ = _train()
    for (k, a), (_k, b) in zip(n1.state_dict().items(), n2.state_dict().items(), strict=True):
        assert torch.equal(a, b), k
    assert log1["params"] == sum(p.numel() for p in n1.parameters())
    assert [h["update"] for h in log1["history"]] == [0, 2, 4]
    assert log1["stopped"] == "updates"


def test_evaluation_never_changes_weights_and_is_reproducible():
    net, _ = _train()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    seqs = S.generate_many(range(10, 12))
    a = f4.evaluate_f4(net, seqs, seed=1, repeats=2)
    b = f4.evaluate_f4(net, seqs, seed=1, repeats=2)
    assert a == b
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k])


def test_policy_is_causal_in_its_own_history():
    net, _ = _train()
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 4, size=40)
    rews = rng.integers(0, 2, size=40).astype(float)
    base = net.policy_logprobs(acts, rews)
    for t in (0, 5, 20, 39):
        a2, r2 = acts.copy(), rews.copy()
        a2[t:] = (a2[t:] + 1) % 4
        r2[t:] = 1.0 - r2[t:]
        alt = net.policy_logprobs(a2, r2)
        assert np.array_equal(base[: t + 1], alt[: t + 1])  # row t depends only on trials < t
        if t + 1 < 40:
            assert not np.array_equal(base[t + 1 :], alt[t + 1 :])


def test_gae_reduces_to_discounted_returns_with_lambda_one():
    r = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    v = torch.zeros_like(r)
    adv = f4.gae(r, v, 0.5, 1.0)
    assert torch.allclose(adv, torch.tensor([[1.375, 0.75, 1.5, 1.0]]))
    assert torch.allclose(f4.gae(r, v, 0.0, 0.5), r)
