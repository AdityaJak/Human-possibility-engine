"""Candidate A — hybrid parametric + exemplar learner (``policy_class = "hybrid_exemplar"``).

Two predictors share one input:

* **parametric pathway** — the unchanged Phase I ``DevelopmentalLearner`` (shared encoder, per-family
  heads, replay, mastery, bandits for activity/social/practice decisions);
* **exemplar pathway** — an ``ExemplarStore`` per task family with a learned diagonal similarity
  metric, recency weighting and (on ordinal families) a graded action-distance kernel.

An **arbitration gate** per family mixes them: ``p = g p_ex + (1 - g) p_par`` with
``g = sigmoid(w . [1, log(1 + retrieval mass), max p_par, max p_ex, recent advantage])``, learned by
gradient ascent on the log-likelihood of observed outcomes. Controls: fixed 50/50 gate, exemplar-only,
parametric-only, no ordinal structure (``hybrid_config.py``).

Ordinal/metric output structure: on families declared ordinal (config ``ordinal_families``,
``Observation.meta["ordinal"]`` or ``AgentInit.families[f]["ordinal"]``), both pathways' distributions
are smoothed by a Gaussian kernel over action indices, so neighbouring levels of a scale share
probability mass. Consolidation replays stored exemplars through the parametric pathway; forgetting
prunes exemplars the parametric pathway already captures.

The learner never reads task internals, held-out sets or other agents' state; assessment and greedy
decisions do not mutate learnable state (contract conformance).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from hpe.contracts.runtime import (
    AgentInit,
    BasePolicy,
    DecisionKind,
    Demonstration,
    Feedback,
    LearningUpdateRecord,
    Observation,
    Policy,
)
from hpe.core.hashing import stable_hash
from hpe.development.config import DevConfig
from hpe.development.exemplar import ExemplarStore, ordinal_kernel
from hpe.development.hybrid_config import HybridConfig
from hpe.development.learner import DevelopmentalLearner

N_GATE_FEATURES = 5
_U_CLIP = 4.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class HybridExemplarLearner(BasePolicy):
    policy_class: ClassVar[str] = "hybrid_exemplar"
    version = "0.1.0"

    def __init__(self, init: AgentInit, config: HybridConfig | None = None) -> None:
        super().__init__(init)
        self.cfg = config or HybridConfig()
        self._build(init)

    # ================================================================== construction
    def _build(self, init: AgentInit) -> None:
        cfg = self.cfg
        inner_params = dict(cfg.inner)
        if not cfg.use_learning:
            inner_params["use_learning"] = False
        self.inner = DevelopmentalLearner(init, DevConfig.from_params(inner_params))
        self.families = list(self.inner.families)
        self.fam_index = dict(self.inner.fam_index)
        n_f = len(self.families)
        self.n_slots = list(self.inner.n_slots)
        self.n_actions = list(self.inner.n_actions)
        self.d = self.inner.d
        self.space_dim = self.inner.k if cfg.exemplar_space == "latent" else self.d
        self.stores = [
            ExemplarStore(cfg.exemplar_capacity, self.n_slots[i], self.d, self.n_actions[i])
            for i in range(n_f)
        ]
        self.logw = [np.zeros(self.space_dim) for _ in range(n_f)]
        self.gate_w = [np.array([cfg.gate_init_bias] + [0.0] * (N_GATE_FEATURES - 1)) for _ in range(n_f)]
        self.ema_adv: list[float] = [0.0] * n_f
        self.n_task_updates: list[int] = [0] * n_f
        self.consolidate_ptr: list[int] = [0] * n_f
        self.now = 0
        self.ordinal = [self._declared_ordinal(f, init) for f in self.families]
        self.ky: list[np.ndarray | None] = [
            ordinal_kernel(self.n_actions[i], cfg.ordinal_spread)
            if (cfg.use_ordinal and self.ordinal[i] and self.n_slots[i] == 1)
            else None
            for i in range(n_f)
        ]
        self.gate_delta_sq = 0.0

    def _declared_ordinal(self, family: str, init: AgentInit) -> bool:
        if family in self.cfg.ordinal_families:
            return True
        meta = init.families.get(family, {})
        return bool(meta.get("ordinal", False)) if isinstance(meta, dict) else False

    # ================================================================== helpers
    def _proj(self, fi: int) -> np.ndarray | None:
        if self.cfg.exemplar_space != "latent":
            return None
        return self.inner.E[self.inner.enc_of[fi]]

    def _query(self, fi: int, ctx: np.ndarray) -> np.ndarray:
        """Query features in the kernel space, shape (1, n_slots, space_dim)."""
        c = self.inner._ctx(ctx)
        proj = self._proj(fi)
        q = c @ proj.T if proj is not None else c
        return q[None, :, :]

    def _weights(self, fi: int) -> np.ndarray:
        if not self.cfg.use_learned_kernel:
            return np.ones(self.space_dim)
        return np.exp(self.logw[fi])

    def _smooth(self, fi: int, p: np.ndarray) -> np.ndarray:
        ky = self.ky[fi]
        if ky is None or p.shape[-1] != ky.shape[0]:
            return p
        out = p @ ky.T
        s = out.sum(axis=-1, keepdims=True) if out.ndim > 1 else out.sum()
        return out / np.maximum(s, 1e-12)

    def _gate_features(self, mass: float, p_par: np.ndarray, p_ex: np.ndarray, fi: int) -> np.ndarray:
        return np.array([1.0, math.log1p(mass), float(p_par.max()), float(p_ex.max()), self.ema_adv[fi]])

    def _gate(self, fi: int, phi: np.ndarray, store_size: int) -> float:
        cfg = self.cfg
        if not cfg.use_exemplar or store_size == 0:
            return 0.0 if cfg.use_parametric else 1.0
        if not cfg.use_parametric:
            return 1.0
        if not cfg.use_learned_gate:
            return float(cfg.fixed_gate)
        return _sigmoid(float(self.gate_w[fi] @ phi))

    def _components(
        self, obs: Observation, fi: int
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
        """(p_par, p_ex, mass, K, query, gate) for one observation; no state is mutated."""
        cfg = self.cfg
        p_par = self._smooth(fi, np.asarray(self.inner.action_probabilities(obs), dtype=np.float64))
        store = self.stores[fi]
        q = self._query(fi, obs.context)
        if cfg.use_exemplar and store.size > 0:
            p_ex_b, mass_b, K = store.evidence(
                q,
                self._weights(fi),
                cfg.kernel_width,
                self.now,
                cfg.recency_decay,
                self.ky[fi],
                cfg.kappa,
                self._proj(fi),
            )
            p_ex, mass = p_ex_b[0], float(mass_b[0])
        else:
            p_ex, mass, K = np.full(obs.n_actions, 1.0 / obs.n_actions), 0.0, np.zeros((1, 0))
        g = self._gate(fi, self._gate_features(mass, p_par, p_ex, fi), store.size)
        return p_par, p_ex, mass, K, q, g

    def _usable(self, obs: Observation) -> int | None:
        fi = self.fam_index.get(obs.family)
        if fi is None or (self.n_slots[fi] > 1 and obs.n_actions != self.n_slots[fi]):
            return None
        if self.n_slots[fi] == 1 and obs.n_actions != self.n_actions[fi]:
            return None
        return fi

    # ================================================================== probabilities
    def _probs(self, obs: Observation) -> np.ndarray:
        if obs.kind != "task":
            return np.asarray(self.inner.action_probabilities(obs), dtype=np.float64)
        fi = self._usable(obs)
        if fi is None:
            return np.full(obs.n_actions, 1.0 / obs.n_actions)
        p_par, p_ex, _, _, _, g = self._components(obs, fi)
        return g * p_ex + (1.0 - g) * p_par

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray:
        fi = self.fam_index.get(family)
        if kind != "task" or fi is None:
            return super().action_probabilities_batch(family, kind, contexts, n_actions, age_years)
        p_par = self._smooth(
            fi, self.inner.action_probabilities_batch(family, kind, contexts, n_actions, age_years)
        )
        cfg = self.cfg
        store = self.stores[fi]
        if not cfg.use_exemplar or store.size == 0 or p_par.shape[1] != self.n_actions[fi]:
            g0 = 0.0 if (cfg.use_parametric or store.size == 0) else 1.0
            if g0 == 0.0:
                return p_par
        c = np.asarray(contexts, dtype=np.float64)
        if c.ndim == 2:
            c = c[:, None, :]
        if c.shape[2] != self.d:
            pad = np.zeros((c.shape[0], c.shape[1], self.d))
            m = min(self.d, c.shape[2])
            pad[:, :, :m] = c[:, :, :m]
            c = pad
        proj = self._proj(fi)
        q = c @ proj.T if proj is not None else c
        if store.size == 0:
            p_ex = np.full_like(p_par, 1.0 / n_actions)
            mass = np.zeros(c.shape[0])
        else:
            p_ex, mass, _ = store.evidence(
                q,
                self._weights(fi),
                cfg.kernel_width,
                self.now,
                cfg.recency_decay,
                self.ky[fi],
                cfg.kappa,
                proj,
            )
        if not cfg.use_parametric:
            return p_ex
        if not cfg.use_learned_gate:
            g = np.full(c.shape[0], cfg.fixed_gate)
        else:
            phi = np.stack(
                [
                    np.ones(c.shape[0]),
                    np.log1p(mass),
                    p_par.max(axis=1),
                    p_ex.max(axis=1),
                    np.full(c.shape[0], self.ema_adv[fi]),
                ],
                axis=1,
            )
            g = 1.0 / (1.0 + np.exp(-np.clip(phi @ self.gate_w[fi], -30, 30)))
        g = np.where(mass > 0, g, 0.0)
        return g[:, None] * p_ex + (1.0 - g[:, None]) * p_par

    def _explain(self, obs: Observation, probs: np.ndarray, action: int) -> dict[str, Any]:
        if obs.kind != "task" or (fi := self._usable(obs)) is None:
            return {"pathway": "inner", "chosen": int(action)}
        p_par, p_ex, mass, _, _, g = self._components(obs, fi)
        return {
            "rule": "gated mixture of exemplar retrieval and parametric prediction",
            "exemplar_share": round(g, 4),
            "retrieval_mass": round(mass, 4),
            "n_exemplars": int(self.stores[fi].size),
            "p_parametric": [round(float(v), 4) for v in p_par],
            "p_exemplar": [round(float(v), 4) for v in p_ex],
            "ordinal": bool(self.ky[fi] is not None),
            "chosen": int(action),
        }

    # ================================================================== learning
    def _event_prob(
        self, p: np.ndarray, action: int, mode: str, correct: int | None, reward: float
    ) -> tuple[float, int, bool]:
        """Probability of the observed event under ``p``; returns (prob, index, positive)."""
        if mode == "full" and correct is not None and 0 <= correct < len(p):
            return float(p[correct]), int(correct), True
        if reward >= 0.5:
            return float(p[action]), int(action), True
        return float(1.0 - p[action]), int(action), False

    def _evid_grad(self, fi: int, p_ex: np.ndarray, idx: int, positive: bool, mass: float) -> np.ndarray:
        """d log p_ex[event] / d evid, using p_ex and the ordinal kernel (S ∝ Ky evid + kappa)."""
        ky = self.ky[fi]
        A = len(p_ex)
        Ky = ky if ky is not None else np.eye(A)
        # S_a = sum_b Ky[a,b] evid_b + kappa ; p = S / sum(S). Use p and sum(S) ≈ mass*rowsums + A*kappa.
        total = max(mass + A * self.cfg.kappa, 1e-9)
        colsum = Ky.sum(axis=0)
        if positive:
            return Ky[idx, :] / max(p_ex[idx] * total, 1e-12) - colsum / total
        # event = not idx: f = 1 - p[idx]
        f = max(1.0 - p_ex[idx], 1e-12)
        return (-(Ky[idx, :] / total) + p_ex[idx] * colsum / total) / f

    def _consolidate(self, fi: int, obs: Observation) -> int:
        store = self.stores[fi]
        if store.size == 0:
            return 0
        n = 0
        for _ in range(self.cfg.n_consolidate):
            i = self.consolidate_ptr[fi] % store.size
            self.consolidate_ptr[fi] += 7
            x = store.X[i].astype(np.float32)
            o = Observation(self.agent_id, obs.tick, obs.age_years, "task", obs.family, x, obs.n_actions)
            if store.mode[i] == 1:
                ra = np.zeros(obs.n_actions)
                ra[int(store.label[i])] = 1.0
                fb = Feedback(
                    mode="full",
                    reward=float(store.action[i] == store.label[i]),
                    correct_action=int(store.label[i]),
                    rewards_all=ra,
                    source="self",
                )
            else:
                fb = Feedback(mode="bandit", reward=float(store.reward[i]), source="self")
            self.inner.learn(o, int(store.action[i]), fb)
            n += 1
        return n

    def _prune(self, fi: int, obs: Observation) -> int:
        store = self.stores[fi]
        if store.size < 8:
            return 0
        X = store.X[: store.size].astype(np.float64)
        p_par = self._smooth(
            fi, self.inner.action_probabilities_batch(obs.family, "task", X, obs.n_actions, obs.age_years)
        )
        capture = store.parametric_capture(p_par)
        n_drop = int(self.cfg.prune_frac * store.size)
        if n_drop <= 0:
            return 0
        order = np.argsort(-capture)
        drop = order[:n_drop]
        drop = drop[capture[drop] > 2.0 / obs.n_actions]
        store.remove(drop)
        return int(drop.size)

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        cfg = self.cfg
        if obs.kind != "task":
            return self.inner.learn(obs, action, feedback)
        fi = self._usable(obs)
        if fi is None or not cfg.use_learning:
            return None
        mode = feedback.mode if self.inner.cfg.use_feedback else "none"
        p_par, p_ex, mass, K, q, g = self._components(obs, fi)
        rec = self.inner.learn(obs, action, feedback) if cfg.use_parametric else None
        if mode == "none":
            return rec
        reward = (
            float(feedback.reward) if feedback.reward is not None and math.isfinite(feedback.reward) else 0.0
        )
        correct = feedback.correct_action if mode == "full" else None
        l_ex, idx, positive = self._event_prob(p_ex, int(action), mode, correct, reward)
        l_par, _, _ = self._event_prob(p_par, int(action), mode, correct, reward)
        l_mix = max(g * l_ex + (1.0 - g) * l_par, 1e-12)
        delta_sq = 0.0
        store = self.stores[fi]
        if cfg.use_exemplar and cfg.use_parametric and store.size > 0:
            adv = math.log(max(l_ex, 1e-12)) - math.log(max(l_par, 1e-12))
            if cfg.use_learned_gate:
                phi = self._gate_features(mass, p_par, p_ex, fi)
                dg = (l_ex - l_par) / l_mix * g * (1.0 - g)
                step = cfg.gate_lr * dg * phi
                self.gate_w[fi] += step
                delta_sq += float(step @ step)
            self.ema_adv[fi] += cfg.advantage_ema * (adv - self.ema_adv[fi])
        if cfg.use_exemplar and cfg.use_learned_kernel and store.size > 0:
            G = self._evid_grad(fi, p_ex, idx, positive, mass)
            gw = store.kernel_gradient(q, self._weights(fi), cfg.kernel_width, K, G, self._proj(fi))
            step = cfg.kernel_lr * gw * np.exp(self.logw[fi])
            step = np.clip(step, -0.5, 0.5)
            self.logw[fi] = np.clip(self.logw[fi] + step, -_U_CLIP, _U_CLIP)
            delta_sq += float(step @ step)
        if cfg.use_exemplar:
            store.add(self.inner._ctx(obs.context).astype(np.float32), correct, int(action), reward, self.now)
            self.compute.memory_items = sum(s.size for s in self.stores)
        self.now += 1
        self.n_task_updates[fi] += 1
        n_extra = 0
        if cfg.use_consolidation and cfg.use_parametric and cfg.use_exemplar and cfg.consolidate_every > 0:
            if self.n_task_updates[fi] % cfg.consolidate_every == 0:
                n_extra += self._consolidate(fi, obs)
        if cfg.use_forgetting and cfg.use_exemplar and cfg.use_parametric and cfg.prune_every > 0:
            if self.n_task_updates[fi] % cfg.prune_every == 0:
                self._prune(fi, obs)
        self.compute.param_updates += 1
        inner_norm = rec.param_delta_norm if rec is not None else 0.0
        return LearningUpdateRecord(
            agent_id=self.agent_id,
            tick=obs.tick,
            policy_class=self.policy_class,
            family=obs.family,
            source=feedback.source,
            feedback_mode=feedback.mode,
            n_samples=1 + n_extra + (rec.n_samples - 1 if rec is not None else 0),
            loss=-math.log(l_mix),
            param_delta_norm=math.sqrt(inner_norm * inner_norm + delta_sq),
            n_updates_total=int(sum(self.n_task_updates)),
        )

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None:
        if not self.cfg.use_learning:
            return None
        return self.inner.learn_from_demonstration(demo)

    def tick_lifecycle(self, age_years: float, dt_years: float) -> None:
        self.inner.tick_lifecycle(age_years, dt_years)

    def self_estimates(self) -> dict[str, dict[str, float]]:
        return self.inner.self_estimates()

    # ================================================================== introspection
    def _param_hash(self) -> str:
        return stable_hash(
            {
                "gate": [g.tolist() for g in self.gate_w],
                "logw": [w.tolist() for w in self.logw],
                "inner": self.inner._param_hash(),
            }
        )

    def parameter_summary(self) -> dict[str, Any]:
        inner = self.inner.parameter_summary()
        fams: dict[str, dict[str, Any]] = {}
        for f, fi in self.fam_index.items():
            fams[f] = {
                "n_exemplars": int(self.stores[fi].size),
                "gate_weights": [round(float(v), 4) for v in self.gate_w[fi]],
                "gate_at_rest": round(_sigmoid(float(self.gate_w[fi][0])), 4),
                "recent_exemplar_advantage": round(self.ema_adv[fi], 4),
                "kernel_weights": [round(float(v), 3) for v in self._weights(fi)],
                "ordinal": bool(self.ky[fi] is not None),
                "n_task_updates": int(self.n_task_updates[fi]),
            }
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "implementation_class": type(self).__name__,
            "flags": {
                k: getattr(self.cfg, k)
                for k in (
                    "use_learning",
                    "use_exemplar",
                    "use_parametric",
                    "use_learned_gate",
                    "use_learned_kernel",
                    "use_ordinal",
                    "use_consolidation",
                    "use_forgetting",
                )
            },
            "exemplar_space": self.cfg.exemplar_space,
            "param_hash": self._param_hash(),
            "families": fams,
            "inner": inner,
            "compute": self.compute.as_dict(),
        }

    # ================================================================== persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "compute": self.compute.as_dict(),
            "config": self.cfg.as_dict(),
            "families": list(self.families),
            "inner": self.inner.state_dict(),
            "stores": [s.state_dict() for s in self.stores],
            "logw": [w.tolist() for w in self.logw],
            "gate_w": [g.tolist() for g in self.gate_w],
            "ema_adv": list(self.ema_adv),
            "n_task_updates": list(self.n_task_updates),
            "consolidate_ptr": list(self.consolidate_ptr),
            "now": int(self.now),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.cfg = HybridConfig.from_params(state["config"])
        self._build(self.init)
        if list(state["families"]) != self.families:
            raise ValueError("state_dict families do not match this agent's AgentInit")
        self.inner.load_state_dict(state["inner"])
        for s, d in zip(self.stores, state["stores"], strict=True):
            s.load_state_dict(d)
        self.logw = [np.asarray(w, dtype=np.float64) for w in state["logw"]]
        self.gate_w = [np.asarray(g, dtype=np.float64) for g in state["gate_w"]]
        self.ema_adv = [float(v) for v in state["ema_adv"]]
        self.n_task_updates = [int(v) for v in state["n_task_updates"]]
        self.consolidate_ptr = [int(v) for v in state["consolidate_ptr"]]
        self.now = int(state["now"])
        c = state.get("compute", {})
        self.compute.memory_items = int(c.get("memory_items", sum(s.size for s in self.stores)))

    def clone(self) -> Policy:
        other = HybridExemplarLearner(self.init, self.cfg)
        other.load_state_dict(self.state_dict())
        return other  # type: ignore[return-value]
