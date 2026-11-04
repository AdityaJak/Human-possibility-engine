import json

import numpy as np
import pytest

import hpe.development as dev
from hpe.contracts.conformance import make_init
from hpe.contracts.runtime import ACTIVITY_ACTIONS, Feedback, Observation
from hpe.core import stream
from hpe.development.bandit import LinearBandit
from hpe.tasks import TRAINING_FAMILIES, SymbolWorld, practice


@pytest.fixture(scope="module")
def world():
    return SymbolWorld(seed=505)


def _activity_obs(mask=None):
    ctx = np.array([[0.8, 0.2, 0.5, 0.4, 0.3, 1.0, 0.0, 1.0]], dtype=np.float32)
    return Observation("a", 0, 30.0, "activity", "activity", ctx, len(ACTIVITY_ACTIONS), action_mask=mask)


def test_activity_bandit_learns_rewarded_action(world):
    pol = dev.build_factory()(make_init(world, seed=0))
    rng = stream(0, "act")
    p0 = pol.action_probabilities(_activity_obs())
    assert np.allclose(p0, 0.25)
    for _ in range(300):
        obs = _activity_obs()
        d = pol.decide(obs, rng)
        r = 1.0 if ACTIVITY_ACTIONS[d.action] == "study" else 0.2
        pol.learn(obs, d.action, Feedback(mode="bandit", reward=r, source="environment"))
    p = pol.action_probabilities(_activity_obs())
    print(f"activity probs after learning {np.round(p, 3)}")
    assert int(np.argmax(p)) == ACTIVITY_ACTIONS.index("study") and p.max() > 0.5
    ex = pol.decide(_activity_obs(), rng).explanation
    assert ex["actions"] == list(ACTIVITY_ACTIONS) and len(ex["estimated_value"]) == 4


def test_activity_mask_respected(world):
    pol = dev.build_factory()(make_init(world, seed=0))
    mask = np.array([True, True, False, True])
    rng = stream(1, "m")
    for _ in range(100):
        assert pol.decide(_activity_obs(mask), rng).action != 2
    assert pol.action_probabilities(_activity_obs(mask))[2] == 0.0


def test_trust_game_bandit_learns(world):
    pol = dev.build_factory()(make_init(world, seed=0))
    rng = stream(2, "soc")
    ctx = np.array([[0.9, 0.0, 1.0, 1.0]], dtype=np.float32)
    for _ in range(200):
        obs = Observation("a", 0, 30.0, "social", "trust_game", ctx, 2)
        d = pol.decide(obs, rng)
        pol.learn(obs, d.action, Feedback(mode="bandit", reward=1.0 if d.action == 1 else 0.3, source="peer"))
    assert pol.action_probabilities(Observation("a", 0, 30.0, "social", "trust_game", ctx, 2))[1] > 0.6


def test_curriculum_prefers_unpractised_and_weak_families(world):
    pol = dev.build_factory()(make_init(world, seed=3))
    practice(pol, world, "compare", 300, stream(3, "c"), agent_id="a", tick=0, age_years=20, mode="full")
    obs = Observation(
        "a", 0, 20.0, "practice", "practice", np.ones((1, 7), np.float32), len(TRAINING_FAMILIES)
    )
    p = pol.action_probabilities(obs)
    assert p[TRAINING_FAMILIES.index("compare")] < min(
        p[TRAINING_FAMILIES.index("sort")], p[TRAINING_FAMILIES.index("signal")]
    )
    ex = pol.decide(obs, stream(3, "d")).explanation
    assert ex["rule"] == "learning-progress curriculum" and ex["families"] == list(TRAINING_FAMILIES)
    uniform = dev.build_factory({"use_curriculum": False})(make_init(world, seed=3))
    assert np.allclose(uniform.action_probabilities(obs), 1 / 3)


def test_linear_bandit_roundtrip():
    b = LinearBandit(3, 4, discount=0.99, prior=1.0)
    rng = stream(0, "lb")
    for _ in range(50):
        x = rng.normal(size=4)
        b.update(int(rng.integers(3)), x, float(rng.random()))
    c = LinearBandit(3, 4, discount=0.99, prior=1.0)
    c.load_state_dict(json.loads(json.dumps(b.state_dict())))
    x = np.ones(4)
    assert np.allclose(b.logits(x, 0.5, 0.5), c.logits(x, 0.5, 0.5), atol=1e-12)
