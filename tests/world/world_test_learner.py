"""Tiny test-only learner (NOT a research policy): per-family linear softmax trained online.

Used only to check that world mechanisms deliver learning *inputs* correctly (e.g. tutored
children receive more instructive episodes and therefore score higher on held-out items).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from hpe.contracts.runtime import (
    AgentInit,
    BasePolicy,
    Demonstration,
    Feedback,
    LearningUpdateRecord,
    Observation,
    Policy,
    masked_softmax,
)


class LinearTestLearner(BasePolicy):
    policy_class: ClassVar[str] = "world_test_linear"
    version = "test-0"

    def __init__(self, init: AgentInit, lr: float = 0.15) -> None:
        super().__init__(init)
        self.lr = lr
        self.W: dict[str, np.ndarray] = {}
        self.n_updates = 0

    @staticmethod
    def _key(obs: Observation) -> str:
        return f"{obs.kind}:{obs.family}"

    def _w(self, obs: Observation) -> np.ndarray:
        k = self._key(obs)
        d = obs.context.shape[1] + 1
        if k not in self.W:
            self.W[k] = np.zeros((obs.n_actions, d)) if obs.context.shape[0] == 1 else np.zeros((1, d))
        return self.W[k]

    def _scores(self, obs: Observation) -> np.ndarray:
        W = self.W[self._key(obs)]
        c = obs.context
        if c.shape[0] == 1:
            return W @ np.append(c[0], 1.0)
        return np.array([W[0] @ np.append(c[i], 1.0) for i in range(c.shape[0])])

    def _probs(self, obs: Observation) -> np.ndarray:
        if self._key(obs) not in self.W:
            return np.ones(obs.n_actions) / obs.n_actions
        return masked_softmax(self._scores(obs), obs.action_mask)

    def _update(self, obs: Observation, grad: np.ndarray, scale: float) -> float:
        W = self._w(obs)
        c = obs.context
        before = W.copy()
        if c.shape[0] == 1:
            W += self.lr * scale * np.outer(grad, np.append(c[0], 1.0))
        else:
            for i in range(c.shape[0]):
                W[0] += self.lr * scale * grad[i] * np.append(c[i], 1.0)
        self.n_updates += 1
        self.compute.param_updates += 1
        return float(np.linalg.norm(W - before))

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        if feedback.mode == "none":
            return None
        p = self._probs(obs)
        if feedback.mode == "full" and feedback.correct_action is not None:
            target = np.eye(obs.n_actions)[feedback.correct_action]
        else:
            g = np.zeros_like(p)
            g[action] = 1.0
            target = p + (feedback.reward - 0.5) * (g - p) * 2
        delta = self._update(obs, target - p, 1.0)
        loss = float(-np.log(max(p[action], 1e-9)))
        return LearningUpdateRecord(
            self.agent_id,
            obs.tick,
            self.policy_class,
            obs.family,
            feedback.source,
            feedback.mode,
            1,
            loss,
            delta,
            self.n_updates,
        )

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None:
        fb = demo.feedback
        obs = demo.observation
        p = self._probs(obs)
        if fb.mode == "full" and fb.correct_action is not None:
            target = np.eye(obs.n_actions)[fb.correct_action]
        elif fb.mode == "bandit" and fb.reward >= 0.5:
            target = np.eye(obs.n_actions)[demo.action]
        else:
            return None
        delta = self._update(obs, target - p, 0.5 * demo.trust)
        return LearningUpdateRecord(
            self.agent_id,
            obs.tick,
            self.policy_class,
            obs.family,
            "peer",
            fb.mode,
            1,
            0.0,
            delta,
            self.n_updates,
        )

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "n_updates": self.n_updates,
            "norms": {k: float(np.linalg.norm(v)) for k, v in sorted(self.W.items())},
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "W": {k: v.tolist() for k, v in sorted(self.W.items())},
            "n_updates": self.n_updates,
            "compute": self.compute.as_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.W = {k: np.array(v, dtype=np.float64) for k, v in state["W"].items()}
        self.n_updates = int(state["n_updates"])


def linear_factory(init: AgentInit) -> Policy:
    return LinearTestLearner(init)  # type: ignore[return-value]
