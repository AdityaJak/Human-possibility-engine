"""Comparison baselines for the developmental learner (ADR-0004).

Three policy classes, all implementing the frozen ``hpe.contracts.runtime.Policy`` protocol and
registered at import time:

* ``static_persona`` — the "written biography" agent. Fixed, trait-derived heuristic parameters
  that are never updated. Because task rules are novel and seed-generated, a non-learner can do
  no better than chance *in expectation*; individual personas vary around chance.
* ``memory_only`` — the "memory-based chatbot" analogue. A bounded episodic store of
  (context, action, outcome) records plus a FIXED retrieval rule (similarity-weighted kNN voting).
  Behaviour changes with experience only through what is stored; no parameter is ever fitted.
* ``plain_rl`` — the "simple RL baseline" (spec §15.3). Independent per-family online linear
  softmax policies on raw perceived features, REINFORCE with a running reward baseline under bandit
  feedback and cross-entropy under instructional feedback. No shared representation, no age
  schedule, no replay, no curriculum, no social learning.

Hyperparameters were chosen on tuning task seeds 1001-1003 (never on evaluation seeds): REINFORCE
with lr=0.05 gave the best mean held-out accuracy among {REINFORCE, reward regression} x
lr {0.02, 0.05, 0.1, 0.2, 0.4} (see experiments/README.md). This is a deliberate
effort to make the strongest *fair* baseline, not a strawman.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np

from hpe.contracts import registry
from hpe.contracts.runtime import (
    AgentInit,
    BasePolicy,
    DecisionKind,
    Demonstration,
    Feedback,
    LearningUpdateRecord,
    Observation,
    Policy,
    PolicyFactory,
    masked_softmax,
)
from hpe.core.hashing import stable_hash
from hpe.core.rng import derive_seed, stream

TRAIT_KEYS = ("curiosity", "diligence", "sociability", "risk_tolerance", "patience")


def _key(obs_kind: str, family: str) -> str:
    return f"{obs_kind}:{family}"


def _is_slot_task(context: np.ndarray, n_actions: int) -> bool:
    """Pair families present one symbol per action (choose a slot): score slots with shared weights."""
    return context.ndim == 2 and context.shape[0] == n_actions and n_actions > 1


def _flat_with_bias(context: np.ndarray) -> np.ndarray:
    return np.append(np.asarray(context, dtype=np.float64).reshape(-1), 1.0)


# =========================================================================================
# Static persona
# =========================================================================================


class StaticPersonaPolicy(BasePolicy):
    """Fixed heuristic derived from traits and a persona seed. Never learns.

    Task decisions: a fixed random linear readout of perceived features (``persona salience``),
    plus a trait bias (risk-tolerant personas lean towards "approach" on ``signal``). Activity
    decisions: fixed trait propensities (diligence -> study/work, sociability -> socialize,
    patience -> rest), modulated by the observed need state through fixed weights.
    """

    policy_class: ClassVar[str] = "static_persona"
    version = "1.0.0"

    def __init__(self, init: AgentInit, params: dict[str, Any] | None = None) -> None:
        super().__init__(init)
        self.params = dict(params or {})
        self.traits = {k: float(init.traits.get(k, 0.5)) for k in TRAIT_KEYS}
        self.persona_seed = derive_seed(init.seed, "static_persona", init.agent_id)
        self.decisiveness = 1.0 + 2.0 * self.traits["diligence"]
        self._cache: dict[tuple[str, int, int], np.ndarray] = {}

    # Pure function of (persona seed, key, shape): cached, but not learnable state.
    def _weights(self, key: str, rows: int, cols: int) -> np.ndarray:
        ck = (key, rows, cols)
        w = self._cache.get(ck)
        if w is None:
            g = stream(self.persona_seed, "weights", key, rows, cols)
            w = g.normal(0.0, 0.6, size=(rows, cols))
            self._cache[ck] = w
        return w

    def _activity_bias(self, obs: Observation) -> np.ndarray:
        t = self.traits
        names = list(obs.meta.get("actions", self.init.activity_actions))[: obs.n_actions]
        table = {
            "rest": 0.4 + t["patience"] - 0.5 * t["diligence"],
            "study": 0.2 + t["diligence"] + 0.5 * t["curiosity"],
            "work": 0.2 + 1.2 * t["diligence"],
            "socialize": 0.2 + 1.3 * t["sociability"],
        }
        bias = np.array([table.get(n, 0.5) for n in names], dtype=np.float64)
        if len(bias) < obs.n_actions:
            bias = np.concatenate([bias, np.full(obs.n_actions - len(bias), 0.5)])
        return bias

    def _logits(self, obs: Observation) -> np.ndarray:
        key = _key(obs.kind, obs.family)
        ctx = np.asarray(obs.context, dtype=np.float64)
        if obs.kind == "task" and _is_slot_task(ctx, obs.n_actions):
            w = self._weights(key, 1, ctx.shape[1])[0]
            return ctx @ w
        x = _flat_with_bias(ctx)
        logits = self._weights(key, obs.n_actions, x.shape[0]) @ x
        if obs.kind == "activity":
            logits = 0.5 * logits + 2.0 * self._activity_bias(obs)
        elif obs.kind == "task" and obs.family == "signal" and obs.n_actions == 2:
            logits = logits + np.array([0.0, self.traits["risk_tolerance"] - 0.5])
        elif obs.kind == "social" and obs.n_actions == 2:
            logits = logits + np.array([0.0, 1.5 * (self.traits["sociability"] - 0.3)])
        return logits

    def _probs(self, obs: Observation) -> np.ndarray:
        return masked_softmax(self._logits(obs), obs.action_mask, 1.0 / self.decisiveness)

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray:
        self.compute.decisions += 0
        key = _key(kind, family)
        ctx = np.asarray(contexts, dtype=np.float64)
        if kind == "task" and _is_slot_task(ctx[0], n_actions):
            w = self._weights(key, 1, ctx.shape[2])[0]
            logits = ctx @ w
        elif kind == "task":
            flat = np.concatenate([ctx.reshape(ctx.shape[0], -1), np.ones((ctx.shape[0], 1))], axis=1)
            logits = flat @ self._weights(key, n_actions, flat.shape[1]).T
            if family == "signal" and n_actions == 2:
                logits = logits + np.array([0.0, self.traits["risk_tolerance"] - 0.5])
        else:
            return super().action_probabilities_batch(family, kind, contexts, n_actions, age_years)
        return masked_softmax(logits, None, 1.0 / self.decisiveness)

    def _explain(self, obs: Observation, probs: np.ndarray, action: int) -> dict[str, Any]:
        dominant = max(self.traits, key=lambda k: self.traits[k])
        return {
            "rule": "fixed persona heuristic (never updated)",
            "dominant_trait": dominant,
            "confidence": float(probs[action]),
        }

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        return None

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "learns": False,
            "persona_seed": self.persona_seed,
            "traits": dict(self.traits),
            "param_hash": stable_hash({"seed": self.persona_seed, "traits": self.traits}),
            "compute": self.compute.as_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "persona_seed": self.persona_seed,
            "traits": dict(self.traits),
            "compute": self.compute.as_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.persona_seed = int(state["persona_seed"])
        self.traits = {k: float(v) for k, v in state["traits"].items()}
        self.decisiveness = 1.0 + 2.0 * self.traits["diligence"]
        self._cache = {}


# =========================================================================================
# Memory only
# =========================================================================================


class _EpisodeStore:
    """Bounded FIFO store of (context, action, vote-weight) records for one decision key."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.contexts: deque[list[float]] = deque(maxlen=capacity)
        self.actions: deque[int] = deque(maxlen=capacity)
        self.values: deque[float] = deque(maxlen=capacity)
        self.weights: deque[float] = deque(maxlen=capacity)

    def add(self, context: np.ndarray, action: int, value: float, weight: float) -> None:
        self.contexts.append([float(v) for v in np.asarray(context, dtype=np.float64).reshape(-1)])
        self.actions.append(int(action))
        self.values.append(float(value))
        self.weights.append(float(weight))

    def __len__(self) -> int:
        return len(self.actions)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.contexts, dtype=np.float64),
            np.asarray(self.actions, dtype=np.int64),
            np.asarray(self.values, dtype=np.float64),
            np.asarray(self.weights, dtype=np.float64),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "contexts": [list(c) for c in self.contexts],
            "actions": list(self.actions),
            "values": list(self.values),
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _EpisodeStore:
        s = cls(int(d["capacity"]))
        for c, a, v, w in zip(d["contexts"], d["actions"], d["values"], d["weights"], strict=True):
            s.contexts.append([float(x) for x in c])
            s.actions.append(int(a))
            s.values.append(float(v))
            s.weights.append(float(w))
        return s


class MemoryOnlyPolicy(BasePolicy):
    """Episodic recall with a fixed retrieval rule; no fitted parameters.

    Storage (per decision key, bounded FIFO, capacity = ``base_params['memory_capacity']``):

    * bandit feedback: (context, chosen action, value = reward - 0.5)
    * full feedback: (context, correct action, value = +0.5) — "I was told the answer"
    * demonstrations: same rules, weight = demonstrator trust
    * activity/social/practice: (context, action, value = reward - running mean reward)

    Retrieval: cosine similarity to every stored context; each record votes for its action with
    weight ``trust * exp(beta * (sim - 1)) * value`` (a fixed similarity kernel, beta = 6). For pair tasks the swapped
    presentation of every memory is also considered (a remembered choice of the right-hand symbol is
    evidence for the same symbol on the left). Votes pass through a fixed softmax. With no relevant
    memory the policy falls back to uniform choice.
    """

    policy_class: ClassVar[str] = "memory_only"
    version = "1.0.0"

    BETA = 6.0
    VOTE_GAIN = 3.0

    def __init__(self, init: AgentInit, params: dict[str, Any] | None = None) -> None:
        super().__init__(init)
        self.params = dict(params or {})
        self.capacity = int(self.params.get("memory_capacity", init.base_params.get("memory_capacity", 256)))
        self.beta = float(self.params.get("beta", self.BETA))
        self.stores: dict[str, _EpisodeStore] = {}
        self.reward_mean: dict[str, float] = {}
        self.recent: dict[str, list[float]] = {}

    def _store(self, key: str) -> _EpisodeStore:
        s = self.stores.get(key)
        if s is None:
            s = _EpisodeStore(self.capacity)
            self.stores[key] = s
        return s

    # ---- retrieval ---------------------------------------------------------------------
    def _votes(self, key: str, contexts: np.ndarray, n_actions: int, slot: bool) -> np.ndarray:
        """contexts: (B, n_slots, F). Returns vote matrix (B, n_actions)."""
        B = contexts.shape[0]
        votes = np.zeros((B, n_actions))
        store = self.stores.get(key)
        if store is None or len(store) == 0:
            return votes
        M, acts, vals, wts = store.arrays()
        q = contexts.reshape(B, -1)
        cands = [(M, acts)]
        if slot and n_actions == 2:
            F = contexts.shape[2]
            swapped = np.concatenate([M[:, F:], M[:, :F]], axis=1)
            cands.append((swapped, 1 - acts))
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        for mem, act in cands:
            mn = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-9)
            sim = qn @ mn.T  # (B, N): kernel over ALL stored episodes (no arbitrary top-k tie-breaks)
            w = np.exp(self.beta * (sim - 1.0)) * (wts * vals)[None, :]
            votes += w @ np.eye(n_actions)[act]
        self.compute.flops_estimate += float(B * len(store) * q.shape[1] * len(cands))
        return votes

    def _probs(self, obs: Observation) -> np.ndarray:
        ctx = np.asarray(obs.context, dtype=np.float64)[None, ...]
        slot = obs.kind == "task" and _is_slot_task(ctx[0], obs.n_actions)
        v = self._votes(_key(obs.kind, obs.family), ctx, obs.n_actions, slot)[0]
        return masked_softmax(self.VOTE_GAIN * v, obs.action_mask)

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray:
        ctx = np.asarray(contexts, dtype=np.float64)
        slot = kind == "task" and _is_slot_task(ctx[0], n_actions)
        v = self._votes(_key(kind, family), ctx, n_actions, slot)
        return masked_softmax(self.VOTE_GAIN * v, None)

    def _explain(self, obs: Observation, probs: np.ndarray, action: int) -> dict[str, Any]:
        store = self.stores.get(_key(obs.kind, obs.family))
        return {
            "rule": "similarity-weighted recall of stored episodes (fixed rule)",
            "memories_available": 0 if store is None else len(store),
            "confidence": float(probs[action]),
        }

    # ---- storage (the only way behaviour changes) --------------------------------------
    def _record(self, obs: Observation, action: int, feedback: Feedback, weight: float) -> bool:
        key = _key(obs.kind, obs.family)
        if feedback.mode == "none":
            return False
        if obs.kind == "task":
            if feedback.mode == "full" and feedback.correct_action is not None:
                self._store(key).add(obs.context, int(feedback.correct_action), 0.5, weight)
                r = 1.0 if action == feedback.correct_action else 0.0
            else:
                r = float(feedback.reward)
                self._store(key).add(obs.context, action, r - 0.5, weight)
            hist = self.recent.setdefault(obs.family, [])
            hist.append(r)
            del hist[:-50]
        else:
            r = float(feedback.reward)
            if not np.isfinite(r):
                return False
            m = self.reward_mean.get(key, r)
            self._store(key).add(obs.context, action, r - m, weight)
            self.reward_mean[key] = m + 0.1 * (r - m)
        self.compute.memory_items = sum(len(s) for s in self.stores.values())
        return True

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        stored = self._record(obs, action, feedback, 1.0)
        if not stored:
            return None
        return LearningUpdateRecord(
            agent_id=self.agent_id,
            tick=obs.tick,
            policy_class=self.policy_class,
            family=obs.family,
            source=feedback.source,
            feedback_mode=feedback.mode,
            n_samples=1,
            loss=float("nan"),
            param_delta_norm=0.0,  # no parameters exist; only the episodic store grew
            n_updates_total=0,
        )

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None:
        stored = self._record(demo.observation, demo.action, demo.feedback, float(np.clip(demo.trust, 0, 1)))
        if not stored:
            return None
        return LearningUpdateRecord(
            agent_id=self.agent_id,
            tick=demo.observation.tick,
            policy_class=self.policy_class,
            family=demo.observation.family,
            source=demo.feedback.source,
            feedback_mode=demo.feedback.mode,
            n_samples=1,
            loss=float("nan"),
            param_delta_norm=0.0,
            n_updates_total=0,
        )

    def self_estimates(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for fam in sorted(self.recent):
            h = self.recent[fam]
            if h:
                m = float(np.mean(h))
                out[fam] = {"mean": m, "sd": float(np.sqrt(m * (1 - m) / len(h))), "n": float(len(h))}
        return out

    def recent_memories(self, n: int = 10) -> list[dict[str, Any]]:
        out = []
        for key in sorted(self.stores):
            s = self.stores[key]
            for a, v, w in list(zip(s.actions, s.values, s.weights, strict=True))[-n:]:
                out.append({"key": key, "action": a, "value": v, "trust": w})
        return out[-n:]

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "learns_parameters": False,
            "memory_capacity": self.capacity,
            "memory_items": {k: len(s) for k, s in sorted(self.stores.items())},
            "memory_hash": stable_hash({k: s.to_dict() for k, s in sorted(self.stores.items())}),
            "compute": self.compute.as_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "capacity": self.capacity,
            "beta": self.beta,
            "stores": {k: s.to_dict() for k, s in sorted(self.stores.items())},
            "reward_mean": dict(sorted(self.reward_mean.items())),
            "recent": {k: list(v) for k, v in sorted(self.recent.items())},
            "compute": self.compute.as_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.capacity = int(state["capacity"])
        self.beta = float(state["beta"])
        self.stores = {k: _EpisodeStore.from_dict(v) for k, v in state["stores"].items()}
        self.reward_mean = {k: float(v) for k, v in state["reward_mean"].items()}
        self.recent = {k: [float(x) for x in v] for k, v in state["recent"].items()}


# =========================================================================================
# Plain RL
# =========================================================================================


class PlainRLPolicy(BasePolicy):
    """Independent per-key online linear softmax policies (REINFORCE / cross-entropy)."""

    policy_class: ClassVar[str] = "plain_rl"
    version = "1.0.0"

    DEFAULT_LR = 0.05
    BASELINE_RATE = 0.05

    def __init__(self, init: AgentInit, params: dict[str, Any] | None = None) -> None:
        super().__init__(init)
        self.params = dict(params or {})
        self.lr = float(self.params.get("lr", self.DEFAULT_LR)) * float(
            init.base_params.get("learning_rate_scale", 1.0)
        )
        self.weights: dict[str, np.ndarray] = {}
        self.baselines: dict[str, float] = {}
        self.n_updates: dict[str, int] = {}
        self.recent: dict[str, list[float]] = {}

    def _shape(self, ctx: np.ndarray, n_actions: int, kind: str) -> tuple[int, ...]:
        if kind == "task" and _is_slot_task(ctx, n_actions):
            return (ctx.shape[1],)
        return (n_actions, ctx.size + 1)

    def _w(self, key: str, shape: tuple[int, ...]) -> np.ndarray:
        w = self.weights.get(key)
        if w is None or w.shape != shape:
            return np.zeros(shape)
        return w

    def _logits(self, obs: Observation) -> np.ndarray:
        ctx = np.asarray(obs.context, dtype=np.float64)
        shape = self._shape(ctx, obs.n_actions, obs.kind)
        w = self._w(_key(obs.kind, obs.family), shape)
        if len(shape) == 1:
            return ctx @ w
        return w @ _flat_with_bias(ctx)

    def _probs(self, obs: Observation) -> np.ndarray:
        return masked_softmax(self._logits(obs), obs.action_mask)

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray:
        ctx = np.asarray(contexts, dtype=np.float64)
        shape = self._shape(ctx[0], n_actions, kind)
        w = self._w(_key(kind, family), shape)
        if len(shape) == 1:
            logits = ctx @ w
        else:
            flat = np.concatenate([ctx.reshape(ctx.shape[0], -1), np.ones((ctx.shape[0], 1))], axis=1)
            logits = flat @ w.T
        return masked_softmax(logits, None)

    def _explain(self, obs: Observation, probs: np.ndarray, action: int) -> dict[str, Any]:
        key = _key(obs.kind, obs.family)
        return {
            "rule": "independent linear softmax policy (online REINFORCE / cross-entropy)",
            "updates_in_family": self.n_updates.get(key, 0),
            "confidence": float(probs[action]),
        }

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        if feedback.mode == "none":
            return None
        key = _key(obs.kind, obs.family)
        ctx = np.asarray(obs.context, dtype=np.float64)
        n = obs.n_actions
        shape = self._shape(ctx, n, obs.kind)
        w = self._w(key, shape).copy()
        p = self._probs(obs)
        if feedback.mode == "full" and feedback.correct_action is not None:
            target = np.zeros(n)
            target[int(feedback.correct_action)] = 1.0
            g = target - p
            loss = float(-np.log(max(p[int(feedback.correct_action)], 1e-9)))
            r = 1.0 if action == feedback.correct_action else 0.0
        else:
            r = float(feedback.reward)
            if not np.isfinite(r):
                return None
            b = self.baselines.get(key, 0.5 if obs.kind == "task" else r)
            adv = r - b
            self.baselines[key] = b + self.BASELINE_RATE * (r - b)
            g = adv * (np.eye(n)[action] - p)
            loss = float(-adv * np.log(max(p[action], 1e-9)))
        if obs.kind == "task":
            hist = self.recent.setdefault(obs.family, [])
            hist.append(r)
            del hist[:-50]
        if len(shape) == 1:
            delta = self.lr * (g @ ctx)
        else:
            delta = self.lr * np.outer(g, _flat_with_bias(ctx))
        self.weights[key] = w + delta
        self.n_updates[key] = self.n_updates.get(key, 0) + 1
        self.compute.param_updates += 1
        self.compute.flops_estimate += float(4 * delta.size)
        return LearningUpdateRecord(
            agent_id=self.agent_id,
            tick=obs.tick,
            policy_class=self.policy_class,
            family=obs.family,
            source=feedback.source,
            feedback_mode=feedback.mode,
            n_samples=1,
            loss=loss,
            param_delta_norm=float(np.linalg.norm(delta)),
            n_updates_total=int(sum(self.n_updates.values())),
        )

    def self_estimates(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for fam in sorted(self.recent):
            h = self.recent[fam]
            if h:
                m = float(np.mean(h))
                out[fam] = {"mean": m, "sd": float(np.sqrt(m * (1 - m) / len(h))), "n": float(len(h))}
        return out

    def parameter_summary(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "learns_parameters": True,
            "lr": self.lr,
            "weight_norms": {k: float(np.linalg.norm(v)) for k, v in sorted(self.weights.items())},
            "n_updates": dict(sorted(self.n_updates.items())),
            "param_hash": stable_hash({k: v for k, v in sorted(self.weights.items())}),
            "compute": self.compute.as_dict(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "lr": self.lr,
            "weights": {
                k: {"shape": list(v.shape), "data": v.reshape(-1).tolist()}
                for k, v in sorted(self.weights.items())
            },
            "baselines": dict(sorted(self.baselines.items())),
            "n_updates": dict(sorted(self.n_updates.items())),
            "recent": {k: list(v) for k, v in sorted(self.recent.items())},
            "compute": self.compute.as_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.lr = float(state["lr"])
        self.weights = {
            k: np.asarray(v["data"], dtype=np.float64).reshape(tuple(v["shape"]))
            for k, v in state["weights"].items()
        }
        self.baselines = {k: float(v) for k, v in state["baselines"].items()}
        self.n_updates = {k: int(v) for k, v in state["n_updates"].items()}
        self.recent = {k: [float(x) for x in v] for k, v in state["recent"].items()}


# =========================================================================================
# Registration
# =========================================================================================


def _builder(cls: type[BasePolicy]) -> Any:
    def build(params: dict[str, Any]) -> PolicyFactory:
        frozen = dict(params)

        def factory(init: AgentInit) -> Policy:
            return cls(init, frozen)  # type: ignore[call-arg,return-value]

        return factory

    return build


BASELINE_CLASSES: dict[str, type[BasePolicy]] = {
    "static_persona": StaticPersonaPolicy,
    "memory_only": MemoryOnlyPolicy,
    "plain_rl": PlainRLPolicy,
}

for _name, _cls in BASELINE_CLASSES.items():
    registry.register(_name, _builder(_cls))
