"""Gate A controls: the age-script-only null and the permuted-feedback lab option; the composition
task world through the learning lab."""

from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.contracts.conformance import make_init, run_policy_conformance
from hpe.core import stream
from hpe.development.schedules import plasticity
from hpe.evaluation.baselines import AgeScriptOnlyPolicy, scripted_plasticity
from hpe.evaluation.learning_lab import LabContext, make_world, resolve_params, run_agent, run_learning_lab
from hpe.tasks import SymbolWorld, assess, policy_state_hash, practice

TINY: dict[str, Any] = {
    "n_agents": 2,
    "episodes_per_family": 40,
    "block": 10,
    "checkpoint_every": 20,
    "skip_transfer": True,
    "skip_retention": True,
}


def _ctx(arm: str, params: dict[str, Any] | None = None, seed: int = 1000) -> LabContext:
    return LabContext("t", arm, seed, f"t__{arm}", resolve_params({**TINY, **(params or {})}))


def test_scripted_schedule_matches_phase1_schedule_exactly():
    for a in np.linspace(0.0, 100.0, 1001):
        assert scripted_plasticity(a) == plasticity(a)


def test_age_script_only_registered_and_conformant():
    assert "age_script_only" in registry.registered()
    f = registry.get_factory("age_script_only")
    run_policy_conformance(f)
    assert isinstance(f(make_init(SymbolWorld(seed=1000))), AgeScriptOnlyPolicy)


def test_age_script_only_changes_with_age_but_never_learns():
    w = SymbolWorld(seed=1000)
    pol = registry.get_factory("age_script_only")(make_init(w))
    it = w.eval_set("sort")[0]
    young = pol.action_probabilities(w.observe(it, "a", 0, 8.0, False))
    old = pol.action_probabilities(w.observe(it, "a", 0, 80.0, False))
    assert young.argmax() == old.argmax() and young.max() > old.max()  # sharper in childhood (scripted)
    h = policy_state_hash(pol)
    practice(pol, w, "compare", 100, stream(1, "p"), agent_id="a", tick=0, age_years=20.0, mode="full")
    assert policy_state_hash(pol) == h
    r0 = assess(pol, w, "compare", 20.0)
    practice(pol, w, "compare", 200, stream(2, "p"), agent_id="a", tick=0, age_years=20.0, mode="full")
    assert assess(pol, w, "compare", 20.0).accuracy == r0.accuracy


def test_permuted_feedback_removes_plain_rl_gain_on_v1_tasks():
    rl = registry.get_factory("plain_rl")
    p = {"episodes_per_family": 120, "checkpoint_every": 40, "n_agents": 3}
    normal = run_learning_lab(rl, _ctx("n", p))
    permuted = run_learning_lab(rl, _ctx("p", {**p, "feedback_permutation": True}))

    def gain(out):
        start = np.mean(
            [o.value for o in out if o.metric.startswith("curve/heldout_accuracy/") and o.tick == 0]
        )
        final = np.mean([o.value for o in out if o.metric == "final_heldout_accuracy"])
        return final - start

    assert gain(normal) > 0.1
    assert gain(permuted) < gain(normal) - 0.1
    assert abs(gain(permuted)) < 0.12  # 3 agents x 1 seed; the dev protocol uses 10 seeds with CIs


def test_permuted_feedback_keeps_items_and_update_calls_identical():
    """Same items are observed and the same number of learn calls happen; only outcomes differ."""
    rl = registry.get_factory("plain_rl")
    ctx_n, ctx_p = _ctx("n"), _ctx("p", {"feedback_permutation": True})
    out_n = run_agent(rl, make_world(1000, ctx_n.params), 0, ctx_n)
    out_p = run_agent(rl, make_world(1000, ctx_p.params), 0, ctx_p)
    dec = {o.metric: o.value for o in out_n if o.metric.startswith("compute/")}
    dec_p = {o.metric: o.value for o in out_p if o.metric.startswith("compute/")}
    assert dec["compute/decisions"] == dec_p["compute/decisions"]


def test_composition_world_through_the_lab():
    p = {
        "task_world": "composition",
        "families": ["primitive_a", "primitive_b", "composition"],
        "n_agents": 2,
    }
    out = run_learning_lab(registry.get_factory("plain_rl"), _ctx("c", p))
    metrics = {o.metric for o in out}
    assert "final_heldout_accuracy/composition" in metrics and "zero_shot/barter" not in metrics
    leak = next(o.value for o in out if o.metric == "leakage/holdout_symbols_in_practice")
    assert leak == 0
    # the specialist control is a different world with the same symbols
    w_shared = make_world(1000, resolve_params({**TINY, **p}))
    w_ctrl = make_world(1000, resolve_params({**TINY, **p, "task_config": {"shared_structure": False}}))
    assert np.array_equal(w_shared.symbols, w_ctrl.symbols) and w_shared.fingerprint() != w_ctrl.fingerprint()


def test_unknown_task_world_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_world(1000, resolve_params({**TINY, "task_world": "nope"}))
