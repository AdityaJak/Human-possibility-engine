"""Controlled learning-lab protocol (no society world required).

Purpose: answer the investor's decisive question (spec §44) under tightly matched conditions:
*does an agent that develops from experience generalise to held-out tasks better than a persona,
a memory-only agent, or a plain RL learner given identical inputs?*

Matching (common random numbers):
* Agent ``i`` receives the same ``AgentInit`` (traits, seed, base params) in every arm.
* Practice items, feedback noise, full-feedback selection and teacher demonstrations for agent ``i``
  are drawn from streams keyed by ``(seed, agent i, family)`` — never by arm — so every arm sees the
  identical item sequence. Only the policy's own exploration stream is policy-specific.
* Every arm gets the same number of episodes, feedback, and assessment items.

Outputs are ``OutcomeObservation`` records (written to raw logs by the runner):

``curve/heldout_accuracy/<family>``         tick = episodes practised in that family
``final_heldout_accuracy``                   mean over training families after the full budget (PRIMARY in H1)
``final_heldout_accuracy/<family>``, ``final_log_loss``, ``final_train_accuracy`` (generalisation gap),
``episodes_to_criterion``, ``heldout_ece`` (calibration of action probabilities on held-out items),
``self_estimate_gap`` (policy self-estimate minus measured accuracy), ``zero_shot/barter``,
``transfer_auc/<mode>/<trained|fresh>``, ``transfer_gain/<mode>``, ``transfer_curve/<mode>/<trained|fresh>``,
``retention_change`` (accuracy change after a no-practice gap), ``leakage/*`` (must be 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from hpe.contracts.runtime import ACTIVITY_ACTIONS, AgentInit, Demonstration, Feedback, Policy, PolicyFactory
from hpe.contracts.schemas import OutcomeObservation
from hpe.core.rng import derive_seed, stream
from hpe.evaluation.contamination import InstrumentedSymbolWorld
from hpe.evaluation.stats import expected_calibration_error
from hpe.tasks.assessment import assess, policy_state_hash, transfer_probe
from hpe.tasks.symbols import TRAINING_FAMILIES, SymbolWorld, TaskWorldConfig

DEFAULT_PARAMS: dict[str, Any] = {
    "n_agents": 20,
    "families": list(TRAINING_FAMILIES),
    "episodes_per_family": 300,
    "block": 10,
    "checkpoint_every": 25,
    "feedback_mode": "bandit",  # none | bandit | full
    "full_feedback_fraction": 0.0,  # extra share of episodes switched to instructional feedback
    "label_noise": 0.05,
    "age_start": 8.0,
    "age_end": 18.0,
    "enable_gate": True,
    "criterion": 0.7,
    "retention_gap_years": 2.0,
    "transfer_modes": ["shared", "orthogonal"],
    "transfer_shots": [0, 4, 8, 16, 32, 64],
    "transfer_feedback": "bandit",
    "demonstrations_per_block": 0,
    "demo_trust": 0.8,
    "misinformation_rate": 0.0,
    "task_config": {},
    "skip_transfer": False,
    "skip_retention": False,
    # Phase II additions (candidate-free lane science/evaluation-v2):
    # task_world: "v1" (SymbolWorld: sort/compare/signal + barter transfer) or "composition"
    # (hpe.tasks.v2.composition: primitive_a/primitive_b/composition/composition_pair; no barter,
    # no v1 transfer probe; task_config -> CompositionConfig).
    "task_world": "v1",
    # Gate A control: when True, the outcome information of every practice episode (reward or
    # correct action) is taken from an independently sampled train item instead of the item the
    # agent observed — a seeded permutation of feedback across the stream with the same marginal
    # distribution, identical architecture and update calls. A learner whose held-out gain survives
    # this is not learning from outcomes.
    "feedback_permutation": False,
}


@dataclass(slots=True)
class LabContext:
    experiment_id: str
    arm: str
    seed: int
    run_id: str
    params: dict[str, Any]


def resolve_params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    p.update(overrides or {})
    unknown = set(p) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown learning-lab params: {sorted(unknown)}")
    if p["episodes_per_family"] % p["block"] != 0:
        raise ValueError("episodes_per_family must be a multiple of block")
    if p["checkpoint_every"] % p["block"] != 0:
        raise ValueError("checkpoint_every must be a multiple of block")
    return p


def agent_init(seed: int, index: int, world: SymbolWorld) -> AgentInit:
    """Identical across arms: depends only on (seed, index)."""
    g = stream(seed, "lab", "traits", index)
    traits = {
        k: float(v)
        for k, v in zip(
            ("curiosity", "diligence", "sociability", "risk_tolerance", "patience"),
            g.uniform(0.2, 0.8, 5),
            strict=True,
        )
    }
    return AgentInit(
        agent_id=f"a{index:04d}",
        birth_tick=0,
        seed=derive_seed(seed, "lab", "init", index),
        traits=traits,
        base_params={"learning_rate_scale": 1.0, "memory_capacity": 256.0, "exploration": 0.1},
        families=world.families_meta(),
        activity_actions=list(ACTIVITY_ACTIONS),
    )


def _obs(
    ctx: LabContext,
    metric: str,
    value: float,
    *,
    agent_id: str | None,
    tick: int | None = None,
    age: float | None = None,
    split: str = "n/a",
    unit: str = "",
) -> OutcomeObservation:
    return OutcomeObservation(
        run_id=ctx.run_id,
        experiment_id=ctx.experiment_id,
        arm=ctx.arm,
        seed=ctx.seed,
        agent_id=agent_id,
        metric=metric,
        value=float(value),
        unit=unit,
        tick=tick,
        age_years=age,
        split=split,  # type: ignore[arg-type]
        source="simulation",
    )


def _episode(
    policy: Policy,
    world: SymbolWorld,
    family: str,
    agent_id: str,
    tick: int,
    age: float,
    p: dict[str, Any],
    items: np.random.Generator,
    fbr: np.random.Generator,
    explore: np.random.Generator,
    sched: np.random.Generator,
    perm: np.random.Generator | None = None,
) -> None:
    item = world.sample_item(family, items, split="train")
    obs = world.observe(item, agent_id, tick, age, p["enable_gate"])
    mode = p["feedback_mode"]
    if sched.random() < p["full_feedback_fraction"]:
        mode = "full"
    dec = policy.decide(obs, explore, explore=True)
    fb_item = item
    if p["feedback_permutation"] and perm is not None:
        # Outcome information comes from a different, independently sampled train item: same label
        # marginal, no relation to the observed context (Gate A permuted-feedback control).
        fb_item = world.sample_item(family, perm, split="train")
    fb = world.feedback(fb_item, dec.action, mode, fbr, noise=p["label_noise"], source="school")
    policy.learn(obs, dec.action, fb)


def _demonstration(
    policy: Policy,
    world: SymbolWorld,
    family: str,
    agent_id: str,
    tick: int,
    age: float,
    p: dict[str, Any],
    demo_rng: np.random.Generator,
) -> None:
    item = world.sample_item(family, demo_rng, split="train")
    obs = world.observe(item, agent_id, tick, age, p["enable_gate"])
    shown = item.correct_action
    if demo_rng.random() < p["misinformation_rate"]:
        shown = int(demo_rng.choice([a for a in range(item.n_actions) if a != item.correct_action]))
    rewards = np.zeros(item.n_actions)
    rewards[shown] = 1.0
    fb = Feedback(mode="full", reward=1.0, correct_action=shown, rewards_all=rewards, source="peer")
    policy.learn_from_demonstration(Demonstration(obs, shown, fb, "teacher", float(p["demo_trust"])))


def run_agent(
    factory: PolicyFactory, world: InstrumentedSymbolWorld, index: int, ctx: LabContext
) -> list[OutcomeObservation]:
    p = ctx.params
    seed = ctx.seed
    init = agent_init(seed, index, world)
    aid = init.agent_id
    policy = factory(init)
    fams: list[str] = list(p["families"])
    n_blocks = p["episodes_per_family"] // p["block"]
    total_blocks = n_blocks * len(fams)
    span = float(p["age_end"] - p["age_start"])
    dt_block = span / total_blocks if total_blocks else 0.0
    out: list[OutcomeObservation] = []

    streams = {
        f: {
            "items": stream(seed, "lab", "items", index, f),
            "fb": stream(seed, "lab", "feedback", index, f),
            "sched": stream(seed, "lab", "schedule", index, f),
            "demo": stream(seed, "lab", "demo", index, f),
            "explore": stream(seed, "lab", "explore", index, f, policy.policy_class),
            "perm": stream(seed, "lab", "feedback_permutation", index, f),
        }
        for f in fams
    }
    v1_world = p["task_world"] == "v1"

    age = float(p["age_start"])
    for f in fams:
        r = assess(policy, world, f, age, enable_gate=p["enable_gate"])
        out.append(
            _obs(
                ctx, f"curve/heldout_accuracy/{f}", r.accuracy, agent_id=aid, tick=0, age=age, split="holdout"
            )
        )

    done = dict.fromkeys(fams, 0)
    reached: dict[str, int | None] = dict.fromkeys(fams, None)
    tick = 0
    for _b in range(n_blocks):
        for f in fams:
            s = streams[f]
            for _ in range(p["block"]):
                _episode(
                    policy,
                    world,
                    f,
                    aid,
                    tick,
                    age,
                    p,
                    s["items"],
                    s["fb"],
                    s["explore"],
                    s["sched"],
                    s["perm"],
                )
                tick += 1
            for _ in range(p["demonstrations_per_block"]):
                _demonstration(policy, world, f, aid, tick, age, p, s["demo"])
            done[f] += p["block"]
            policy.tick_lifecycle(age, dt_block)
            age += dt_block
            if done[f] % p["checkpoint_every"] == 0:
                r = assess(policy, world, f, age, enable_gate=p["enable_gate"])
                out.append(
                    _obs(
                        ctx,
                        f"curve/heldout_accuracy/{f}",
                        r.accuracy,
                        agent_id=aid,
                        tick=done[f],
                        age=age,
                        split="holdout",
                    )
                )
                if reached[f] is None and r.accuracy >= p["criterion"]:
                    reached[f] = done[f]

    # ---- final measurements (learning disabled) ------------------------------------------------
    h_before = policy_state_hash(policy)
    finals, losses, trains, eces, gaps = [], [], [], [], []
    selfest = policy.self_estimates()
    for f in fams:
        r = assess(policy, world, f, age, enable_gate=p["enable_gate"])
        rt = assess(policy, world, f, age, split="train", enable_gate=p["enable_gate"])
        finals.append(r.accuracy)
        losses.append(r.log_loss)
        trains.append(rt.accuracy)
        out.append(
            _obs(ctx, f"final_heldout_accuracy/{f}", r.accuracy, agent_id=aid, age=age, split="holdout")
        )
        items = world.eval_set(f)
        ctxs = np.stack([world.perceive(it.context, age, p["enable_gate"]) for it in items])
        probs = policy.action_probabilities_batch(f, "task", ctxs, items[0].n_actions, age)
        correct = np.array([it.correct_action for it in items])
        greedy = probs.argmax(axis=1)
        eces.append(expected_calibration_error(probs.max(axis=1), (greedy == correct).astype(float)))
        if f in selfest and "mean" in selfest[f]:
            gaps.append(float(selfest[f]["mean"]) - r.accuracy)
        crit = reached[f]
        out.append(
            _obs(
                ctx,
                f"episodes_to_criterion/{f}",
                float(crit) if crit is not None else float(p["episodes_per_family"] * 2),
                agent_id=aid,
                unit="episodes (2x budget = not reached)",
            )
        )
    out.append(
        _obs(ctx, "final_heldout_accuracy", float(np.mean(finals)), agent_id=aid, age=age, split="holdout")
    )
    out.append(
        _obs(
            ctx, "final_log_loss", float(np.mean(losses)), agent_id=aid, age=age, split="holdout", unit="nats"
        )
    )
    out.append(
        _obs(ctx, "final_train_accuracy", float(np.mean(trains)), agent_id=aid, age=age, split="train")
    )
    out.append(
        _obs(ctx, "generalization_gap", float(np.mean(trains) - np.mean(finals)), agent_id=aid, age=age)
    )
    out.append(_obs(ctx, "heldout_ece", float(np.nanmean(eces)), agent_id=aid, age=age, split="holdout"))
    if gaps:
        out.append(_obs(ctx, "self_estimate_gap", float(np.mean(gaps)), agent_id=aid, age=age))
    if v1_world:
        zs = assess(policy, world, "barter", age, enable_gate=p["enable_gate"])
        out.append(_obs(ctx, "zero_shot/barter", zs.accuracy, agent_id=aid, age=age, split="transfer"))
    out.append(
        _obs(
            ctx,
            "leakage/assessment_mutated_policy",
            float(policy_state_hash(policy) != h_before),
            agent_id=aid,
        )
    )

    # ---- transfer to the never-practised family (on clones; v1 task world only) ------------------
    if not p["skip_transfer"] and v1_world:
        shots = tuple(int(s) for s in p["transfer_shots"])
        for mode in p["transfer_modes"]:
            tw = SymbolWorld(
                seed=world.seed, config=TaskWorldConfig(**{**p["task_config"], "transfer_mode": mode})
            )
            fresh = factory(init)
            aucs = {}
            for label, pol in (("trained", policy), ("fresh", fresh)):
                tp = transfer_probe(
                    pol,
                    tw,
                    "barter",
                    stream(seed, "lab", "transfer", index, mode),
                    age_years=age,
                    shots=shots,
                    mode=p["transfer_feedback"],
                    enable_gate=p["enable_gate"],
                )
                aucs[label] = tp.area_under_curve
                out.append(
                    _obs(
                        ctx,
                        f"transfer_auc/{mode}/{label}",
                        tp.area_under_curve,
                        agent_id=aid,
                        age=age,
                        split="transfer",
                    )
                )
                for s_, a_ in zip(tp.shots, tp.accuracy, strict=True):
                    out.append(
                        _obs(
                            ctx,
                            f"transfer_curve/{mode}/{label}",
                            a_,
                            agent_id=aid,
                            tick=s_,
                            age=age,
                            split="transfer",
                        )
                    )
            out.append(
                _obs(
                    ctx,
                    f"transfer_gain/{mode}",
                    aucs["trained"] - aucs["fresh"],
                    agent_id=aid,
                    age=age,
                    split="transfer",
                )
            )

    # ---- retention after a no-practice gap (mutates the agent; done last) ------------------------
    if not p["skip_retention"] and p["retention_gap_years"] > 0:
        months = int(round(p["retention_gap_years"] * 12))
        for _ in range(months):
            age += 1 / 12
            policy.tick_lifecycle(age, 1 / 12)
        after = [assess(policy, world, f, age, enable_gate=p["enable_gate"]).accuracy for f in fams]
        out.append(
            _obs(
                ctx,
                "retention_change",
                float(np.mean(after) - np.mean(finals)),
                agent_id=aid,
                age=age,
                split="holdout",
            )
        )

    cd = policy.compute.as_dict()
    out.append(_obs(ctx, "compute/decisions", cd["decisions"], agent_id=aid, unit="count"))
    out.append(_obs(ctx, "compute/param_updates", cd["param_updates"], agent_id=aid, unit="count"))
    return out


def make_world(seed: int, p: dict[str, Any]) -> InstrumentedSymbolWorld:
    """Task world for the lab. ``composition`` worlds are duck-typed to SymbolWorld (same methods,
    including ``leakage_counts``); the return annotation keeps the rest of the lab's typing."""
    kind = p["task_world"]
    if kind == "v1":
        return InstrumentedSymbolWorld(seed, TaskWorldConfig(**p["task_config"]))
    if kind == "composition":
        from hpe.tasks.v2.composition import CompositionConfig, CompositionWorld

        return cast(InstrumentedSymbolWorld, CompositionWorld(seed, CompositionConfig(**p["task_config"])))
    raise ValueError(f"unknown task_world {kind!r} (expected 'v1' or 'composition')")


def run_learning_lab(factory: PolicyFactory, ctx: LabContext) -> list[OutcomeObservation]:
    p = ctx.params
    world = make_world(ctx.seed, p)
    out: list[OutcomeObservation] = []
    for i in range(int(p["n_agents"])):
        out.extend(run_agent(factory, world, i, ctx))
    leak = world.leakage_counts()
    out.append(
        _obs(
            ctx,
            "leakage/holdout_symbols_in_practice",
            leak["holdout_symbols_in_practice"],
            agent_id=None,
            unit="count",
        )
    )
    out.append(_obs(ctx, "leakage/practice_items", leak["practice_items"], agent_id=None, unit="count"))
    return out
