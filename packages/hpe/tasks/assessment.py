"""Measurement protocol shared by the world, experiments and dashboard.

Skill is ONLY ever measured here, on held-out items, with learning disabled. The policy's
own confidence (``self_estimates``) is reported separately and never counted as skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hpe.contracts.runtime import FeedbackMode, FeedbackSource, Policy
from hpe.core.hashing import stable_hash
from hpe.tasks.symbols import Split, SymbolWorld


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(slots=True)
class AssessmentResult:
    family: str
    split: Split
    n_items: int
    accuracy: float  # greedy accuracy, ties credited fractionally
    ci_low: float
    ci_high: float
    expected_accuracy: float  # mean probability assigned to the correct action
    log_loss: float  # mean -log p(correct)
    chance: float
    age_years: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "family": self.family,
            "split": self.split,
            "n_items": self.n_items,
            "accuracy": self.accuracy,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "expected_accuracy": self.expected_accuracy,
            "log_loss": self.log_loss,
            "chance": self.chance,
            "age_years": self.age_years,
        }


def assess(
    policy: Policy,
    world: SymbolWorld,
    family: str,
    age_years: float,
    *,
    split: Split = "holdout",
    enable_gate: bool = True,
    n_items: int | None = None,
) -> AssessmentResult:
    """Score ``policy`` on the fixed held-out set for ``family``. Does not call ``learn``."""
    items = world.eval_set(family, split=split, n=n_items)
    contexts = np.stack([world.perceive(it.context, age_years, enable_gate) for it in items])
    n_actions = items[0].n_actions
    probs = policy.action_probabilities_batch(family, "task", contexts, n_actions, age_years)
    correct = np.array([it.correct_action for it in items])
    p_correct = probs[np.arange(len(items)), correct]
    maxp = probs.max(axis=1, keepdims=True)
    is_max = np.isclose(probs, maxp, rtol=0, atol=1e-12)
    credit = is_max[np.arange(len(items)), correct] / is_max.sum(axis=1)
    acc = float(credit.mean())
    lo, hi = wilson_interval(float(credit.sum()), len(items))
    return AssessmentResult(
        family=family,
        split=split,
        n_items=len(items),
        accuracy=acc,
        ci_low=lo,
        ci_high=hi,
        expected_accuracy=float(p_correct.mean()),
        log_loss=float(-np.log(np.clip(p_correct, 1e-6, 1.0)).mean()),
        chance=1.0 / n_actions,
        age_years=age_years,
    )


@dataclass(slots=True)
class PracticeStats:
    family: str
    n: int = 0
    rewards: float = 0.0
    correct: int = 0
    loss_sum: float = 0.0
    delta_norm_sum: float = 0.0
    n_updates: int = 0

    @property
    def mean_reward(self) -> float:
        return self.rewards / self.n if self.n else float("nan")

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "family": self.family,
            "n": self.n,
            "mean_reward": self.mean_reward,
            "correct": self.correct,
            "mean_loss": self.loss_sum / self.n_updates if self.n_updates else float("nan"),
            "delta_norm": self.delta_norm_sum,
            "n_updates": self.n_updates,
        }


def practice(
    policy: Policy,
    world: SymbolWorld,
    family: str,
    n: int,
    rng: np.random.Generator,
    *,
    agent_id: str,
    tick: int,
    age_years: float,
    mode: FeedbackMode = "bandit",
    source: FeedbackSource = "environment",
    noise: float = 0.0,
    enable_gate: bool = True,
    difficulty: float | None = None,
    learn: bool = True,
) -> PracticeStats:
    """Run ``n`` practice episodes on TRAIN-pool items. The only place task learning happens."""
    stats = PracticeStats(family=family)
    for _ in range(n):
        item = world.sample_item(family, rng, split="train", difficulty=difficulty)
        obs = world.observe(item, agent_id, tick, age_years, enable_gate)
        dec = policy.decide(obs, rng, explore=True)
        fb = world.feedback(item, dec.action, mode, rng, noise=noise, source=source)
        stats.n += 1
        stats.correct += int(dec.action == item.correct_action)
        if fb.mode != "none":
            stats.rewards += fb.reward
        if learn:
            upd = policy.learn(obs, dec.action, fb)
            if upd is not None:
                stats.loss_sum += upd.loss
                stats.delta_norm_sum += upd.param_delta_norm
                stats.n_updates += 1
    return stats


@dataclass(slots=True)
class TransferProbe:
    family: str
    shots: list[int]
    accuracy: list[float]
    log_loss: list[float]
    area_under_curve: float
    meta: dict[str, float | str] = field(default_factory=dict)


def transfer_probe(
    policy: Policy,
    world: SymbolWorld,
    family: str,
    rng: np.random.Generator,
    *,
    age_years: float,
    shots: tuple[int, ...] = (0, 4, 8, 16, 32, 64),
    mode: FeedbackMode = "bandit",
    enable_gate: bool = True,
) -> TransferProbe:
    """Few-shot learning curve on a family, measured on a CLONE (the agent is not modified).

    Accuracy at each shot count is measured on held-out symbols. The area under the curve
    (mean accuracy across shot counts) summarises sample efficiency on the new family.
    """
    learner = policy.clone()
    accs: list[float] = []
    losses: list[float] = []
    done = 0
    for s in shots:
        if s > done:
            practice(
                learner,
                world,
                family,
                s - done,
                rng,
                agent_id="probe",
                tick=-1,
                age_years=age_years,
                mode=mode,
                source="environment",
                enable_gate=enable_gate,
            )
            done = s
        r = assess(learner, world, family, age_years, split="holdout", enable_gate=enable_gate)
        accs.append(r.accuracy)
        losses.append(r.log_loss)
    return TransferProbe(family, list(shots), accs, losses, float(np.mean(accs)))


def policy_state_hash(policy: Policy) -> str:
    """Hash of learnable state, excluding compute counters (used by no-mutation tests)."""
    sd = dict(policy.state_dict())
    sd.pop("compute", None)
    return stable_hash(sd)
