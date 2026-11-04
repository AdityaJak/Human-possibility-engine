"""Reusable Policy conformance checks. Every lane runs these against its policy classes.

Usage in a lane's tests::

    from hpe.contracts.conformance import run_policy_conformance
    def test_conformance():
        run_policy_conformance(my_factory)
"""

from __future__ import annotations

import json

import numpy as np

from hpe.contracts.runtime import (
    ACTIVITY_ACTIONS,
    AgentInit,
    Demonstration,
    Feedback,
    Observation,
    PolicyFactory,
)
from hpe.core.rng import stream
from hpe.tasks.assessment import assess, policy_state_hash, practice
from hpe.tasks.symbols import ALL_FAMILIES, SymbolWorld


def make_init(world: SymbolWorld, agent_id: str = "a0000", seed: int = 0) -> AgentInit:
    return AgentInit(
        agent_id=agent_id,
        birth_tick=0,
        seed=seed,
        traits={
            "curiosity": 0.6,
            "diligence": 0.5,
            "sociability": 0.4,
            "risk_tolerance": 0.5,
            "patience": 0.5,
        },
        base_params={"learning_rate_scale": 1.0, "memory_capacity": 256.0, "exploration": 0.1},
        families=world.families_meta(),
        activity_actions=list(ACTIVITY_ACTIONS),
    )


def run_policy_conformance(factory: PolicyFactory, *, seed: int = 3, check_learning: bool = False) -> None:
    world = SymbolWorld(seed=seed)
    pol = factory(make_init(world))
    rng = stream(seed, "conformance")

    # 1. probabilities are valid and decide() respects masks
    for fam in ALL_FAMILIES:
        item = world.sample_item(fam, rng)
        obs = world.observe(item, pol.agent_id if hasattr(pol, "agent_id") else "a0000", 0, 12.0)
        p = pol.action_probabilities(obs)
        assert p.shape == (obs.n_actions,), f"{fam}: bad prob shape {p.shape}"
        assert np.all(p >= 0) and abs(p.sum() - 1) < 1e-6, f"{fam}: probs must sum to 1"
        obs.action_mask = np.zeros(obs.n_actions, dtype=bool)
        obs.action_mask[-1] = True
        d = pol.decide(obs, rng, explore=True)
        assert d.action == obs.n_actions - 1, "decide must respect action_mask"

    # 1b. non-task decision kinds
    act_obs = Observation(
        "a0000", 0, 30.0, "activity", "activity", np.ones((1, 8), np.float32), len(ACTIVITY_ACTIONS)
    )
    p = pol.action_probabilities(act_obs)
    assert p.shape == (len(ACTIVITY_ACTIONS),) and abs(p.sum() - 1) < 1e-6
    pol.learn(
        act_obs,
        int(pol.decide(act_obs, rng).action),
        Feedback(mode="bandit", reward=0.3, source="environment"),
    )
    soc_obs = Observation("a0000", 0, 30.0, "social", "trust_game", np.ones((1, 4), np.float32), 2)
    pol.learn(
        soc_obs, int(pol.decide(soc_obs, rng).action), Feedback(mode="bandit", reward=1.0, source="peer")
    )
    pr_obs = Observation("a0000", 0, 30.0, "practice", "practice", np.ones((1, 7), np.float32), 3)
    pol.learn(pr_obs, int(pol.decide(pr_obs, rng).action), Feedback(mode="bandit", reward=0.5, source="self"))

    # 2. greedy decisions and assessment do not mutate learnable state
    before = policy_state_hash(pol)
    for fam in ALL_FAMILIES:
        assess(pol, world, fam, 12.0)
    item = world.sample_item("sort", rng)
    pol.decide(world.observe(item, "a0000", 0, 12.0), rng, explore=False)
    assert policy_state_hash(pol) == before, "assessment/greedy decide mutated learnable state"

    # 3. batch probabilities equal per-item probabilities
    items = world.eval_set("compare")[:8]
    ctx = np.stack([world.perceive(it.context, 12.0) for it in items])
    pb = pol.action_probabilities_batch("compare", "task", ctx, 2, 12.0)
    for i, it in enumerate(items):
        pi = pol.action_probabilities(world.observe(it, "a0000", 0, 12.0))
        assert np.allclose(pb[i], pi, atol=1e-9), "batch and single probabilities differ"

    # 4. learning hooks accept every feedback mode; state_dict round-trips exactly
    for mode in ("none", "bandit", "full"):
        practice(pol, world, "sort", 5, rng, agent_id="a0000", tick=1, age_years=12.0, mode=mode)  # type: ignore[arg-type]
    demo_item = world.sample_item("signal", rng)
    demo_obs = world.observe(demo_item, "a0000", 2, 12.0)
    demo_fb = world.feedback(demo_item, demo_item.correct_action, "full", rng)
    pol.learn_from_demonstration(Demonstration(demo_obs, demo_item.correct_action, demo_fb, "a0001", 0.8))
    pol.tick_lifecycle(12.1, 1 / 12)
    sd = pol.state_dict()
    json.dumps(sd)  # must be JSON-serialisable
    clone = factory(make_init(world))
    clone.load_state_dict(json.loads(json.dumps(sd)))
    assert policy_state_hash(clone) == policy_state_hash(pol), "state_dict round-trip changed state"
    r1, r2 = stream(seed, "rt"), stream(seed, "rt")
    for fam in ALL_FAMILIES:
        it = world.eval_set(fam)[0]
        o = world.observe(it, "a0000", 3, 12.0)
        assert pol.decide(o, r1).action == clone.decide(o, r2).action, "restored policy decides differently"

    # 5. clone is independent
    c = pol.clone()
    practice(c, world, "compare", 20, rng, agent_id="a0000", tick=4, age_years=12.0, mode="full")
    json.dumps(pol.parameter_summary())
    json.dumps(pol.self_estimates())

    # 6. optional: learning actually improves held-out accuracy
    if check_learning:
        fresh = factory(make_init(world, seed=11))
        base = assess(fresh, world, "compare", 20.0).accuracy
        practice(
            fresh,
            world,
            "compare",
            300,
            stream(seed, "learn"),
            agent_id="a0000",
            tick=5,
            age_years=20.0,
            mode="full",
        )
        after = assess(fresh, world, "compare", 20.0).accuracy
        assert after > base + 0.1, f"no held-out improvement: {base:.2f} -> {after:.2f}"
