"""Task encodings and reference cognitive models for human-data validation.

Every model — the repository's policy classes resolved from the registry AND the reference
models below — is driven through the same ``Policy`` protocol with the same observations and
feedback, so likelihood comparisons are like-for-like.

Encodings (identical for all models):

* MCPL: ``kind="task"``, ``family="mcpl"``, context ``(1, 12)`` = one-hot(cue1) as ±1 (5),
  one-hot(cue2) as ±1 (5), cue1 and cue2 scaled to [-1, 1] (2); ``n_actions=9``. Feedback
  ``full`` (correct level revealed) while the transcript shows feedback, else ``none``.
* IGT: ``kind="task"``, ``family="igt"``, constant context ``ones((1, 12))``; ``n_actions=4``;
  feedback ``bandit`` with reward ``clip(0.5 + net / (2 * scale), 0, 1)``.

Reference models (not part of the simulator; standard cognitive-model families):

* ``ref_bias`` — fixed choice distribution = smoothed pooled choice frequencies of TRAIN
  participants (a fair "static" reference: no within-person adaptation).
* ``ref_population_static`` — MCPL only: fixed P(choice | cue1, cue2) from TRAIN participants.
* ``ref_delta_rule`` — MCPL: ordinal criterion estimate learned by the delta rule
  (Rescorla-Wagner / LMS) with Gaussian response noise; IGT: Rescorla-Wagner values + softmax.
* ``ref_instance_based`` — MCPL: kernel-weighted retrieval of remembered (cues, outcome)
  instances; IGT: recency-decayed blended deck values + softmax.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from hpe.contracts.runtime import AgentInit, BasePolicy, Feedback, Observation, Policy, masked_softmax
from hpe.evaluation.human.psych101 import IGTTrial, MCPLTrial, Participant

N_FEATURES = 12
MCPL_ACTIONS = 9
IGT_ACTIONS = 4

FAMILIES_META: dict[str, dict[str, int]] = {
    "mcpl": {"n_slots": 1, "n_actions": MCPL_ACTIONS, "n_features": N_FEATURES},
    "igt": {"n_slots": 1, "n_actions": IGT_ACTIONS, "n_features": N_FEATURES},
}

STATIC_MODELS = ("random", "static_persona", "ref_bias", "ref_population_static")
ADAPTIVE_MODELS = ("memory_only", "plain_rl", "developmental", "ref_delta_rule", "ref_instance_based")
REFERENCE_MODELS = ("ref_bias", "ref_population_static", "ref_delta_rule", "ref_instance_based")
REGISTRY_MODELS = ("random", "static_persona", "memory_only", "plain_rl", "developmental")


# ----------------------------------------------------------------------------- encodings
def mcpl_context(c1: int, c2: int) -> np.ndarray:
    x = -np.ones((1, N_FEATURES), dtype=np.float32)
    x[0, c1] = 1.0
    x[0, 5 + c2] = 1.0
    x[0, 10] = (c1 - 2) / 2.0
    x[0, 11] = (c2 - 2) / 2.0
    return x


IGT_CONTEXT = np.ones((1, N_FEATURES), dtype=np.float32)


def igt_reward(net: float, scale: float) -> float:
    return float(np.clip(0.5 + net / (2.0 * scale), 0.0, 1.0))


Step = tuple[Observation, int, Feedback]


def encode(p: Participant, *, scale: float = 500.0, age_years: float = 25.0) -> list[Step]:
    """Convert a participant's transcript to (observation, human action, feedback) steps."""
    steps: list[Step] = []
    for t, tr in enumerate(p.trials):
        if p.paradigm == "mcpl":
            assert isinstance(tr, MCPLTrial)
            obs = Observation(
                p.key,
                t,
                age_years,
                "task",
                "mcpl",
                mcpl_context(tr.cue1, tr.cue2),
                MCPL_ACTIONS,
                meta={"cue1": tr.cue1, "cue2": tr.cue2},
            )
            if tr.correct is None:
                fb = Feedback(mode="none", source="human_data")
            else:
                ra = np.zeros(MCPL_ACTIONS)
                ra[tr.correct] = 1.0
                fb = Feedback(
                    mode="full",
                    reward=float(tr.choice == tr.correct),
                    correct_action=tr.correct,
                    rewards_all=ra,
                    source="human_data",
                )
        else:
            assert isinstance(tr, IGTTrial)
            obs = Observation(p.key, t, age_years, "task", "igt", IGT_CONTEXT.copy(), IGT_ACTIONS)
            fb = Feedback(mode="bandit", reward=igt_reward(tr.net, scale), source="human_data")
        steps.append((obs, tr.choice, fb))
    return steps


# ----------------------------------------------------------------------------- reference models
class _RefPolicy(BasePolicy):
    version = "1.0.0"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init)
        self.params = dict(params)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["params"] = self.params
        return d


class FixedDistributionModel(_RefPolicy):
    """Same choice distribution on every trial (per family)."""

    policy_class: ClassVar[str] = "ref_bias"

    def _probs(self, obs: Observation) -> np.ndarray:
        return np.asarray(self.params["probs"][obs.family], dtype=np.float64)


class PopulationStaticModel(_RefPolicy):
    """MCPL: fixed P(choice | cue1, cue2) estimated from other (train) participants."""

    policy_class: ClassVar[str] = "ref_population_static"

    def _probs(self, obs: Observation) -> np.ndarray:
        table = self.params["table"]
        if obs.family != "mcpl":
            return np.asarray(self.params["fallback"][obs.family], dtype=np.float64)
        return np.asarray(table[obs.meta["cue1"]][obs.meta["cue2"]], dtype=np.float64)


class DeltaRuleModel(_RefPolicy):
    policy_class: ClassVar[str] = "ref_delta_rule"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init, params)
        self.w = np.zeros(3)  # MCPL: [bias, w1, w2] on criterion scale; bias starts at midpoint
        self.w[0] = 4.0
        self.q = np.full(IGT_ACTIONS, 0.5)  # IGT deck values

    def _probs(self, obs: Observation) -> np.ndarray:
        if obs.family == "mcpl":
            x = np.array([1.0, obs.meta["cue1"] - 2.0, obs.meta["cue2"] - 2.0])
            yhat = float(self.w @ x)
            k = np.arange(MCPL_ACTIONS)
            return masked_softmax(-((k - yhat) ** 2) / (2 * self.params["sigma"] ** 2), None)
        return masked_softmax(self.params["beta"] * self.q, None)

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> None:
        self.compute.learn_calls += 1
        if feedback.mode == "none":
            return None
        a = self.params["alpha"]
        if obs.family == "mcpl":
            if feedback.correct_action is None:
                return None
            x = np.array([1.0, obs.meta["cue1"] - 2.0, obs.meta["cue2"] - 2.0])
            self.w += a * (feedback.correct_action - float(self.w @ x)) * x
        else:
            self.q[action] += a * (feedback.reward - self.q[action])
        self.compute.param_updates += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d.update(w=self.w.tolist(), q=self.q.tolist())
        return d


class InstanceBasedModel(_RefPolicy):
    policy_class: ClassVar[str] = "ref_instance_based"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init, params)
        self.counts = np.zeros((5, 5, MCPL_ACTIONS))  # MCPL memory of (cue1, cue2, outcome)
        lv = np.arange(5)
        h = params.get("h", 1.0)
        s = params.get("s", 1.0)
        self._kc = np.exp(-((lv[:, None] - lv[None, :]) ** 2) / (2 * h * h))
        ko = np.arange(MCPL_ACTIONS)
        self._ky = np.exp(-((ko[:, None] - ko[None, :]) ** 2) / (2 * s * s))
        self.sw = np.zeros(IGT_ACTIONS)  # IGT decayed weights and weighted outcome sums
        self.swr = np.zeros(IGT_ACTIONS)

    def _probs(self, obs: Observation) -> np.ndarray:
        if obs.family == "mcpl":
            c1, c2 = obs.meta["cue1"], obs.meta["cue2"]
            sim = self._kc[c1][:, None] * self._kc[c2][None, :]  # (5,5)
            evid = np.tensordot(sim, self.counts, axes=([0, 1], [0, 1]))  # (9,) remembered outcomes
            score = self._ky @ evid + self.params.get("kappa", 0.05)
            return score / score.sum()
        v = np.where(self.sw > 0, self.swr / np.maximum(self.sw, 1e-12), 0.5)
        return masked_softmax(self.params["beta"] * v, None)

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> None:
        self.compute.learn_calls += 1
        if obs.family == "mcpl":
            if feedback.mode == "full" and feedback.correct_action is not None:
                self.counts[obs.meta["cue1"], obs.meta["cue2"], feedback.correct_action] += 1.0
                self.compute.memory_items += 1
            return None
        if feedback.mode == "none":
            return None
        d = self.params["decay"]
        self.sw *= d
        self.swr *= d
        self.sw[action] += 1.0
        self.swr[action] += feedback.reward
        self.compute.memory_items += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d.update(counts=self.counts.tolist(), sw=self.sw.tolist(), swr=self.swr.tolist())
        return d


# ----------------------------------------------------------------------------- fitting static refs
def pooled_choice_probs(train: list[Participant], n_actions: int, smoothing: float = 1.0) -> np.ndarray:
    c = np.full(n_actions, smoothing)
    for p in train:
        for t in p.trials:
            c[t.choice] += 1
    return c / c.sum()


def population_table(train: list[Participant], smoothing: float = 1.0) -> list[list[list[float]]]:
    c = np.full((5, 5, MCPL_ACTIONS), smoothing)
    for p in train:
        for t in p.trials:
            assert isinstance(t, MCPLTrial)
            c[t.cue1, t.cue2, t.choice] += 1
    return (c / c.sum(axis=2, keepdims=True)).tolist()


ReferenceBuilder = Callable[[dict[str, Any], list[Participant], str], Callable[[AgentInit], Policy]]


def reference_factory(
    name: str, params: dict[str, Any], train: list[Participant], paradigm: str
) -> Callable[[AgentInit], Policy]:
    """Build a per-participant factory for a reference model. Static refs are fit on ``train`` only."""
    n_actions = MCPL_ACTIONS if paradigm == "mcpl" else IGT_ACTIONS
    if name == "ref_bias":
        probs = pooled_choice_probs(train, n_actions, params.get("smoothing", 1.0)).tolist()
        p2 = {**params, "probs": {paradigm: probs}}
        return lambda init: FixedDistributionModel(init, p2)  # type: ignore[return-value]
    if name == "ref_population_static":
        if paradigm != "mcpl":
            raise ValueError("ref_population_static is defined for MCPL only")
        p2 = {
            **params,
            "table": population_table(train, params.get("smoothing", 1.0)),
            "fallback": {},
        }
        return lambda init: PopulationStaticModel(init, p2)  # type: ignore[return-value]
    if name == "ref_delta_rule":
        return lambda init: DeltaRuleModel(init, params)  # type: ignore[return-value]
    if name == "ref_instance_based":
        return lambda init: InstanceBasedModel(init, params)  # type: ignore[return-value]
    raise KeyError(name)
