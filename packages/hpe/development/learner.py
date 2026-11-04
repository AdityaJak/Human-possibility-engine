"""The developmental learner (proposed policy class, ``policy_class = "developmental"``).

What it is
----------
An inspectable online learner whose competence comes only from experience:

* **Task model** — a symbol encoder ``E`` (``k_hat x d``) maps each perceived symbol ``x`` to a
  latent code ``z = E x``. Single-symbol families (``sort``, ``signal``) read out class logits
  ``H_f z + b_f``; pair families (``compare``, ``barter``) score each slot ``u_f . z_i`` and pick
  via softmax over slots. With ``use_shared_representation=True`` (default) one encoder is
  shared by every family; with ``False`` each family has an independent specialist encoder.
  All heads start at exactly zero, so an unpractised family is scored at chance.
* **Learning** — online SGD on a likelihood model in which the softmax over logits is the
  agent's belief about which action is correct:

  - full (instructional) feedback: cross-entropy toward the revealed correct action;
  - bandit feedback, reward 1: the chosen action is (probably) correct → ``-log p_a``;
  - bandit feedback, reward 0: the chosen action is (probably) wrong → ``-log(1 - p_a)``
    (a proper likelihood for the observed reward; no information is invented);
  - no feedback: no parameter update.

* **Self-model** — Beta mastery posterior per family (decaying), used for exploration
  temperature and curriculum. It is a belief, never a measurement of skill.
* **Episodic replay** — bounded memory of the agent's own labelled episodes, replayed to
  consolidate (a deterministic strided cursor; no hidden randomness).
* **Age schedules** (scripted assumptions, see ``schedules.py``) — plasticity, working-memory
  capacity, late-life forgetting acceleration.
* **Forgetting** — Ornstein–Uhlenbeck drift of unpractised families' parameters back toward
  their initial values (seeded noise; RNG state is part of ``state_dict``).
* **Curriculum** — chooses what to practise by expected learning progress.
* **Social learning** — trust-weighted learning from demonstrations, with skepticism toward
  demonstrations that contradict confident beliefs.
* **Activity / trust-game choice** — discounted Bayesian linear contextual bandits.

* **Consolidation** — per-family step size shrinks with effective (weight-summed) experience;
  the shared encoder's with the total experience of all families using it, so a new family's
  head learns fast on top of a stable representation.

Hyperparameter provenance
-------------------------
Defaults in ``DevConfig`` were chosen by a small manual sweep (6 configurations) on task
seeds 7 and 21 only, against a scratch linear reference learner. Reported experiments must
use other task seeds so that held-out results are not tuned on.

What it is not
--------------
No language model, no pretrained knowledge, no access to task rules, held-out items or other
agents' state. Simulated age is a state variable, not a claim about human development.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from hpe.contracts.runtime import (
    ACTIVITY_CONTEXT_FEATURES,
    SOCIAL_CONTEXT_FEATURES,
    AgentInit,
    BasePolicy,
    Decision,
    DecisionKind,
    Demonstration,
    Feedback,
    LearningUpdateRecord,
    Observation,
    Policy,
)
from hpe.core.hashing import stable_hash
from hpe.core.rng import stream
from hpe.development.bandit import LinearBandit
from hpe.development.config import DevConfig
from hpe.development.memory import CODE_MODES, ReplayMemory
from hpe.development.schedules import forgetting_multiplier, plasticity, working_memory
from hpe.tasks.symbols import TRAINING_FAMILIES

SD_MAX = math.sqrt(1.0 / 12.0)  # sd of Beta(1, 1): maximal uncertainty
ENCODER_STATIONARY_SD = 0.1


def _softmax_list(vals: list[float], mask: np.ndarray | None) -> list[float]:
    """Softmax for one small logit vector in pure Python (numpy call overhead dominates at n<=4)."""
    if mask is not None:
        keep = mask.tolist()
        m = max(v for v, k in zip(vals, keep, strict=True) if k)
        e = [math.exp(v - m) if k else 0.0 for v, k in zip(vals, keep, strict=True)]
    else:
        m = max(vals)
        e = [math.exp(v - m) for v in vals]
    s = sum(e)
    return [v / s for v in e]


def _softmax(logits: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return np.array(_softmax_list(logits.tolist(), mask))


class DevelopmentalLearner(BasePolicy):
    policy_class: ClassVar[str] = "developmental"
    version = "0.1.0"

    def __init__(self, init: AgentInit, config: DevConfig | None = None) -> None:
        super().__init__(init)
        self.cfg = config or DevConfig()
        self._build(init)

    # ================================================================== construction
    def _build(self, init: AgentInit) -> None:
        cfg = self.cfg
        task_fams = {f: m for f, m in init.families.items()}
        self.families: list[str] = list(task_fams)
        self.fam_index: dict[str, int] = {f: i for i, f in enumerate(self.families)}
        self.n_slots = [int(task_fams[f].get("n_slots", 1)) for f in self.families]
        self.n_actions = [int(task_fams[f].get("n_actions", 2)) for f in self.families]
        self.is_pair = [s > 1 for s in self.n_slots]
        self.d = max([int(task_fams[f].get("n_features", 12)) for f in self.families] or [12])
        k = int(cfg.latent_dim)
        self.k = k
        n_f = len(self.families)

        rng = stream(init.seed, "development", init.agent_id, "init")
        n_enc = 1 if cfg.use_shared_representation else n_f
        self.enc_of = [0] * n_f if cfg.use_shared_representation else list(range(n_f))
        self.E = [rng.normal(0.0, cfg.encoder_init_scale, size=(k, self.d)) for _ in range(max(1, n_enc))]
        self.E0 = [e.copy() for e in self.E]
        self.H = [
            np.zeros((1, k)) if self.is_pair[i] else np.zeros((self.n_actions[i], k)) for i in range(n_f)
        ]
        self.hb = [np.zeros(0) if self.is_pair[i] else np.zeros(self.n_actions[i]) for i in range(n_f)]

        # scalar bookkeeping is kept in Python lists: per-element numpy access is slower
        self.alpha: list[float] = [1.0] * n_f
        self.beta: list[float] = [1.0] * n_f
        self.n_feedback: list[float] = [0.0] * n_f
        self.ema_fast: list[float] = [0.5] * n_f
        self.ema_slow: list[float] = [0.5] * n_f
        self.n_updates: list[int] = [0] * n_f
        self.practiced: list[bool] = [False] * n_f
        self.last_practice_age: list[float] = [-1.0] * n_f
        self.total_updates = 0
        # effective (weight-summed) experience per family drives consolidation decay; a
        # down-weighted demonstration counts for less than a fully attended own episode
        self.experience: list[float] = [0.0] * n_f
        self.enc_users: list[list[int]] = [
            [fi for fi in range(n_f) if self.enc_of[fi] == e] for e in range(len(self.E))
        ]
        self.age = 0.0

        self.practice_fams = [f for f in TRAINING_FAMILIES if f in self.fam_index]
        self.practice_value: list[float] = [0.0] * len(self.practice_fams)

        base_cap = int(init.base_params.get("memory_capacity", 256))
        self.memory = ReplayMemory(base_cap, max(self.n_slots or [1]), self.d)

        self.activity_actions = list(init.activity_actions)
        self.activity = LinearBandit(
            max(1, len(self.activity_actions)),
            len(ACTIVITY_CONTEXT_FEATURES),
            discount=cfg.bandit_discount,
            prior=cfg.bandit_prior_precision,
        )
        self.social = LinearBandit(
            2, len(SOCIAL_CONTEXT_FEATURES), discount=cfg.bandit_discount, prior=cfg.bandit_prior_precision
        )
        self._rng = stream(init.seed, "development", init.agent_id, "forgetting")

        traits = init.traits
        self.kappa = max(
            0.0,
            cfg.exploration_base
            + 2.0 * float(init.base_params.get("exploration", 0.1))
            + 0.5 * (float(traits.get("curiosity", 0.5)) - 0.5),
        )
        self.lr_scale = float(init.base_params.get("learning_rate_scale", 1.0))
        self.social_scale = 0.5 + float(traits.get("sociability", 0.5))
        self.source_credibility: dict[str, float] = {}

        self._version = 0
        self._cache_obs: Observation | None = None
        self._cache_version = -1
        self._cache_ctx: np.ndarray | None = None
        self._cache_z: np.ndarray | None = None
        self._cache_logits: np.ndarray | None = None

    # ================================================================== helpers
    def _mastery(self, fi: int) -> tuple[float, float]:
        a, b = self.alpha[fi], self.beta[fi]
        s = a + b
        return a / s, math.sqrt(a * b / (s * s * (s + 1.0)))

    def _temperature(self, fi: int) -> float:
        _, sd = self._mastery(fi)
        return 1.0 + self.kappa * sd / SD_MAX

    def _ctx(self, context: np.ndarray) -> np.ndarray:
        c = context if context.dtype == np.float64 else context.astype(np.float64)
        if c.ndim == 1:
            c = c[None, :]
        if c.shape[1] != self.d:
            out = np.zeros((c.shape[0], self.d))
            m = min(self.d, c.shape[1])
            out[:, :m] = c[:, :m]
            c = out
        return c

    def _forward(self, fi: int, ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (latent codes with shape (n_slots, k), logits with shape (n_actions,))."""
        E = self.E[self.enc_of[fi]]
        if self.is_pair[fi]:
            z = ctx @ E.T
            return z, z @ self.H[fi][0]
        z1 = E @ ctx[0]
        return z1[None, :], self.H[fi] @ z1 + self.hb[fi]

    def _task_forward_cached(self, obs: Observation, fi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache_obs is obs and self._cache_version == self._version and self._cache_z is not None:
            assert self._cache_logits is not None and self._cache_ctx is not None
            return self._cache_ctx, self._cache_z, self._cache_logits
        ctx = self._ctx(obs.context)
        z, logits = self._forward(fi, ctx)
        self._cache_obs, self._cache_version = obs, self._version
        self._cache_ctx, self._cache_z, self._cache_logits = ctx, z, logits
        return ctx, z, logits

    def _lr(self, age: float, fi: int) -> tuple[float, float]:
        """Step sizes for family ``fi``: age plasticity x consolidation (experience) decay.

        The head decays with that family's own practice count; the encoder decays with the
        total practice of every family sharing it, so a newly encountered family learns its
        head quickly on top of a stable, already-consolidated shared representation.
        """
        cfg = self.cfg
        p = plasticity(age, cfg.use_plasticity_schedule) * self.lr_scale
        if cfg.lr_decay_tau > 0.0:
            n_head = self.experience[fi]
            n_enc = sum(self.experience[j] for j in self.enc_users[self.enc_of[fi]])
            d_head = (1.0 + n_head / cfg.lr_decay_tau) ** -cfg.lr_decay_power
            d_enc = (1.0 + n_enc / cfg.lr_decay_tau) ** -cfg.lr_decay_power
            return cfg.lr_head * p * d_head, cfg.lr_encoder * p * d_enc
        return cfg.lr_head * p, cfg.lr_encoder * p

    def _effective_capacity(self, age: float) -> int:
        return max(1, int(self.memory.capacity * working_memory(age, self.cfg.use_plasticity_schedule)))

    # ================================================================== probabilities
    def _probs(self, obs: Observation) -> np.ndarray:
        mask = obs.action_mask
        kind = obs.kind
        if kind == "task":
            fi = self.fam_index.get(obs.family)
            if fi is None or (self.is_pair[fi] and obs.n_actions != self.n_slots[fi]):
                return _uniform(obs)
            _, _, logits = self._task_forward_cached(obs, fi)
            if logits.shape[0] != obs.n_actions:
                return _uniform(obs)
            return _softmax(logits / self._temperature(fi), mask)
        if kind == "activity":
            if obs.n_actions != self.activity.n_actions:
                return _uniform(obs)
            lg = self.activity.logits(obs.context, self.cfg.bandit_ucb, self.cfg.bandit_temperature)
            return _softmax(lg, mask)
        if kind == "social":
            if obs.n_actions != self.social.n_actions:
                return _uniform(obs)
            lg = self.social.logits(obs.context, self.cfg.bandit_ucb, self.cfg.bandit_temperature)
            return _softmax(lg, mask)
        if kind == "practice":
            if not self.cfg.use_curriculum or obs.n_actions != len(self.practice_fams):
                return _uniform(obs)
            return _softmax(self._curriculum_scores() / self.cfg.curriculum_temperature, mask)
        return _uniform(obs)

    def action_probabilities(self, obs: Observation) -> np.ndarray:
        # _probs always returns a normalised, masked distribution (softmax or uniform)
        return self._probs(obs)

    def action_probabilities_batch(
        self, family: str, kind: DecisionKind, contexts: np.ndarray, n_actions: int, age_years: float
    ) -> np.ndarray:
        fi = self.fam_index.get(family)
        if kind != "task" or fi is None:
            return super().action_probabilities_batch(family, kind, contexts, n_actions, age_years)
        c = np.asarray(contexts, dtype=np.float64)
        if c.ndim == 2:
            c = c[:, None, :]
        if c.shape[2] != self.d:
            pad = np.zeros((c.shape[0], c.shape[1], self.d))
            m = min(self.d, c.shape[2])
            pad[:, :, :m] = c[:, :, :m]
            c = pad
        if self.is_pair[fi] and n_actions != c.shape[1]:
            return np.full((c.shape[0], n_actions), 1.0 / n_actions)
        z = c @ self.E[self.enc_of[fi]].T  # (n, slots, k)
        if self.is_pair[fi]:
            logits = z @ self.H[fi][0]  # (n, slots)
        else:
            logits = z[:, 0, :] @ self.H[fi].T + self.hb[fi]
        if logits.shape[1] != n_actions:
            return np.full((c.shape[0], n_actions), 1.0 / n_actions)
        lg = logits / self._temperature(fi)
        lg = lg - lg.max(axis=1, keepdims=True)
        e = np.exp(lg)
        out: np.ndarray = e / e.sum(axis=1, keepdims=True)
        return out

    def _curriculum_scores(self) -> np.ndarray:
        cfg = self.cfg
        scores = np.empty(len(self.practice_fams))
        for j, f in enumerate(self.practice_fams):
            fi = self.fam_index[f]
            mean, sd = self._mastery(fi)
            progress = abs(float(self.ema_fast[fi] - self.ema_slow[fi]))
            scores[j] = sd + 2.0 * progress + cfg.curriculum_weakness_bonus * (1.0 - mean)
        return scores

    # ================================================================== decide / explain
    def decide(self, obs: Observation, rng: np.random.Generator, *, explore: bool = True) -> Decision:
        p = self.action_probabilities(obs)
        self.compute.decisions += 1
        if explore:
            pl = p.tolist()
            u = rng.random() * sum(pl)
            action = max(i for i, v in enumerate(pl) if v > 0.0)  # fallback: last available action
            acc = 0.0
            for i, v in enumerate(pl):
                acc += v
                if v > 0.0 and u < acc:
                    action = i
                    break
        else:
            action = int(np.argmax(p))
        return Decision(action, p, self.policy_class, self.version, self._explain(obs, p, action))

    def _explain(self, obs: Observation, probs: np.ndarray, action: int) -> dict[str, Any]:
        kind = obs.kind
        if kind == "task":
            fi = self.fam_index.get(obs.family)
            if fi is None:
                return {"rule": "unknown family: uniform"}
            mean, sd = self._mastery(fi)
            if not obs.meta.get("explain"):
                return {"mastery_mean": mean}
            ctx = self._ctx(obs.context)
            z, logits = self._forward(fi, ctx)
            if self.is_pair[fi]:
                contrib = self.H[fi][0] * z[action]
            else:
                contrib = self.H[fi][action] * z[0]
            top = np.argsort(-np.abs(contrib))[:2]
            return {
                "rule": "softmax over learned latent readout",
                "family": obs.family,
                "scores": [float(v) for v in logits],
                "probs": [float(v) for v in probs],
                "temperature": self._temperature(fi),
                "top_latent": [{"dim": int(j), "contribution": float(contrib[j])} for j in top],
                "mastery_mean": mean,
                "mastery_sd": sd,
                "shared_representation": self.cfg.use_shared_representation,
            }
        if kind in ("activity", "social"):
            bandit = self.activity if kind == "activity" else self.social
            if obs.n_actions != bandit.n_actions:
                return {"rule": "uniform (unexpected action space)"}
            mean_v, sd_v = bandit.values(obs.context)
            names = self.activity_actions if kind == "activity" else ["withhold", "cooperate"]
            return {
                "rule": "optimistic softmax over Bayesian linear reward model",
                "actions": list(names),
                "estimated_value": [float(v) for v in mean_v],
                "uncertainty": [float(v) for v in sd_v],
                "chosen": names[action] if action < len(names) else action,
                "n_observations": [int(v) for v in bandit.n],
            }
        if kind == "practice":
            if not self.cfg.use_curriculum or obs.n_actions != len(self.practice_fams):
                return {"rule": "uniform practice choice"}
            return {
                "rule": "learning-progress curriculum",
                "families": list(self.practice_fams),
                "scores": [float(v) for v in self._curriculum_scores()],
                "chosen": self.practice_fams[action] if action < len(self.practice_fams) else action,
            }
        return {}

    # ================================================================== learning (tasks)
    def _logit_grad(
        self, p: list[float], mode: str, action: int, reward: float, correct: int | None
    ) -> tuple[list[float], float, int]:
        """Return (descent direction in logit space, loss, success indicator).

        ``p`` is the (untempered) belief over which action is correct.
        """
        n = len(p)
        if mode == "full" and correct is not None and 0 <= correct < n:
            g = [-v for v in p]
            g[correct] += 1.0
            return g, -math.log(max(p[correct], 1e-12)), int(action == correct)
        if reward >= 0.5:
            g = [-v for v in p]
            g[action] += 1.0
            return g, -math.log(max(p[action], 1e-12)), 1
        pa = p[action]
        q = max(1.0 - pa, self.cfg.bandit_negative_floor)
        r = pa / q
        g = [v * r for v in p]
        g[action] -= r
        return g, -math.log(q), 0

    def _apply(
        self,
        fi: int,
        ctx: np.ndarray,
        z: np.ndarray,
        g_list: list[float],
        lr: float,
        lr_e: float,
        want_norm: bool = False,
    ) -> float:
        """Apply a logit-space descent direction. Returns the squared update norm if requested.

        Single-symbol families: logits = H z + b, z = E x.
        Pair families:          logit_i = u . z_i, z_i = E x_i.
        """
        E = self.E[self.enc_of[fi]]
        sq = 0.0
        if self.is_pair[fi]:
            u = self.H[fi][0]
            gl = np.array([v * lr for v in g_list])
            du = gl @ z  # lr * sum_i g_i z_i       (k,)
            if lr_e > 0.0:
                gx = np.array([v * lr_e for v in g_list]) @ ctx  # lr_e * sum_i g_i x_i   (d,)
                if want_norm:
                    uu = float(u @ u)
                    sq += uu * float(gx @ gx)
                E += u[:, None] * gx
            u += du
            if want_norm:
                sq += float(du @ du)
            return sq
        H = self.H[fi]
        x = ctx[0]
        lg = np.array([v * lr for v in g_list])
        if lr_e > 0.0:
            hg = np.array([v * lr_e for v in g_list]) @ H  # lr_e * H^T g             (k,)
            if want_norm:
                sq += float(hg @ hg) * float(x @ x)
            E += hg[:, None] * x
        zz = z[0]
        H += lg[:, None] * zz
        self.hb[fi] += lg
        if want_norm:
            gg = sum(v * v for v in g_list) * lr * lr
            sq += gg * float(zz @ zz) + gg
        return sq

    def _replay(self, age: float) -> int:
        mem = self.memory
        n_done = 0
        for idx in mem.next_replay_indices(self.cfg.n_replay):
            fi = int(mem.fam[idx])
            ns = self.n_slots[fi]
            ctx = mem.ctx[idx, :ns].astype(np.float64)
            z, logits = self._forward(fi, ctx)
            p = _softmax_list(logits.tolist(), None)
            mode = CODE_MODES[int(mem.mode[idx])]
            label = int(mem.label[idx])
            g, _, _ = self._logit_grad(
                p, mode, int(mem.action[idx]), float(mem.reward[idx]), label if label >= 0 else None
            )
            w = self.cfg.replay_weight
            lr, lr_e = self._lr(age, fi)
            self._apply(fi, ctx, z, g, lr * w, lr_e * w)
            n_done += 1
        return n_done

    def _update_mastery(self, fi: int, success: int) -> None:
        gamma = self.cfg.mastery_decay
        self.alpha[fi] = 1.0 + gamma * (self.alpha[fi] - 1.0) + success
        self.beta[fi] = 1.0 + gamma * (self.beta[fi] - 1.0) + (1 - success)
        self.n_feedback[fi] += 1
        self.ema_fast[fi] += 0.2 * (success - self.ema_fast[fi])
        self.ema_slow[fi] += 0.05 * (success - self.ema_slow[fi])

    def _learn_task(
        self, obs: Observation, action: int, fb: Feedback, mode: str
    ) -> LearningUpdateRecord | None:
        fi = self.fam_index.get(obs.family)
        if fi is None or (self.is_pair[fi] and obs.n_actions != self.n_slots[fi]):
            return None
        age = float(obs.age_years)
        self.age = age
        self.practiced[fi] = True
        self.last_practice_age[fi] = age
        if mode == "none":
            self._version += 1
            return None
        reward = float(fb.reward) if fb.reward is not None and math.isfinite(fb.reward) else 0.0
        ctx, z, logits = self._task_forward_cached(obs, fi)
        p = _softmax_list(logits.tolist(), obs.action_mask)
        correct = fb.correct_action if mode == "full" else None
        g, loss, success = self._logit_grad(p, mode, int(action), reward, correct)
        lr, lr_e = self._lr(age, fi)
        sq = self._apply(fi, ctx, z, g, lr, lr_e, want_norm=True)
        self._update_mastery(fi, success)
        n_replayed = 0
        if self.cfg.use_memory:
            if self.cfg.replay_every > 0 and self.total_updates % self.cfg.replay_every == 0:
                n_replayed = self._replay(age)
            self.memory.add(self._effective_capacity(age), fi, ctx, int(action), mode, reward, correct)
        self.n_updates[fi] += 1
        self.experience[fi] += 1.0
        self.total_updates += 1
        self._version += 1
        self.compute.param_updates += 1 + n_replayed
        self.compute.flops_estimate += (1 + n_replayed) * 4.0 * self.k * (self.d + max(self.n_actions[fi], 2))
        self.compute.memory_items = self.memory.size
        return LearningUpdateRecord(
            agent_id=self.agent_id,
            tick=obs.tick,
            policy_class=self.policy_class,
            family=obs.family,
            source=fb.source,
            feedback_mode=fb.mode,
            n_samples=1 + n_replayed,
            loss=loss,
            param_delta_norm=math.sqrt(sq),
            n_updates_total=self.total_updates,
        )

    # ================================================================== learning (contract)
    def learn(self, obs: Observation, action: int, feedback: Feedback) -> LearningUpdateRecord | None:
        self.compute.learn_calls += 1
        if not self.cfg.use_learning:
            return None
        mode = feedback.mode if self.cfg.use_feedback else "none"
        kind = obs.kind
        if kind == "task":
            return self._learn_task(obs, action, feedback, mode)
        if mode == "none" or feedback.reward is None or not math.isfinite(feedback.reward):
            return None
        r = float(feedback.reward)
        if kind in ("activity", "social"):
            bandit = self.activity if kind == "activity" else self.social
            if obs.n_actions != bandit.n_actions or not (0 <= action < bandit.n_actions):
                return None
            delta = bandit.update(int(action), obs.context, r)
            self._version += 1
            self.compute.param_updates += 1
            fam = "activity" if kind == "activity" else obs.family
            return LearningUpdateRecord(
                self.agent_id,
                obs.tick,
                self.policy_class,
                fam,
                feedback.source,
                feedback.mode,
                1,
                0.0,
                delta,
                self.total_updates,
            )
        if kind == "practice":
            if not (0 <= action < len(self.practice_value)):
                return None
            before = self.practice_value[action]
            self.practice_value[action] = before + 0.1 * (r - before)
            self._version += 1
            return LearningUpdateRecord(
                self.agent_id,
                obs.tick,
                self.policy_class,
                "practice",
                feedback.source,
                feedback.mode,
                1,
                0.0,
                abs(self.practice_value[action] - before),
                self.total_updates,
            )
        return None

    def learn_from_demonstration(self, demo: Demonstration) -> LearningUpdateRecord | None:
        cfg = self.cfg
        if not (cfg.use_learning and cfg.use_social_learning):
            return None
        obs = demo.observation
        if obs.kind != "task":
            return None
        fi = self.fam_index.get(obs.family)
        if fi is None or (self.is_pair[fi] and obs.n_actions != self.n_slots[fi]):
            return None
        fb = demo.feedback
        age = float(obs.age_years)
        ctx = self._ctx(obs.context)
        z, logits = self._forward(fi, ctx)
        p = _softmax_list(logits.tolist(), obs.action_mask)
        n = len(p)
        weight = float(np.clip(demo.trust, 0.0, 1.0)) * cfg.social_weight * self.social_scale
        reward = float(fb.reward) if fb.reward is not None and math.isfinite(fb.reward) else float("nan")
        if (
            cfg.use_feedback
            and fb.mode == "full"
            and fb.correct_action is not None
            and 0 <= fb.correct_action < n
        ):
            mode, act, correct, proposed, positive = (
                "full",
                int(demo.action),
                int(fb.correct_action),
                int(fb.correct_action),
                True,
            )
        elif cfg.use_feedback and fb.mode == "bandit" and math.isfinite(reward):
            mode, act, correct, proposed, positive = (
                "bandit",
                int(demo.action),
                None,
                int(demo.action),
                reward >= 0.5,
            )
        else:  # unlabeled imitation: assume the demonstrated action is correct
            mode, act, correct, proposed, positive = (
                "full",
                int(demo.action),
                int(demo.action),
                int(demo.action),
                True,
            )
            weight *= cfg.unlabeled_imitation_weight
            reward = 1.0
        skepticism = 1.0
        if cfg.use_skepticism:
            mean, sd = self._mastery(fi)
            chance = 1.0 / n
            competence = min(1.0, max(0.0, (mean - chance) / (1.0 - chance)))
            certainty = max(0.0, 1.0 - sd / SD_MAX)
            conf = competence * certainty
            pmax = max(p)
            if positive:
                agree = p[proposed] / pmax if pmax > 0 else 1.0
            else:  # demonstration claims `proposed` is wrong; disagreement if we believe it is right
                agree = 1.0 - p[proposed] if p.index(pmax) == proposed else 1.0
            # source monitoring: credibility of this demonstrator, judged against confident beliefs
            own_choice = p.index(pmax)
            consistent = (own_choice == proposed) if positive else (own_choice != proposed)
            cred = self.source_credibility.get(demo.demonstrator_id, 1.0)
            cred += cfg.credibility_rate * conf * ((1.0 if consistent else 0.0) - cred)
            self._remember_source(demo.demonstrator_id, cred)
            skepticism = max(cfg.skepticism_floor, (1.0 - conf * (1.0 - agree)) * cred)
            weight *= skepticism
        if weight <= 0.0:
            return None
        g, loss, _ = self._logit_grad(p, mode, act, reward if math.isfinite(reward) else 0.0, correct)
        lr, lr_e = self._lr(age, fi)
        sq = self._apply(fi, ctx, z, g, lr * weight, lr_e * weight, want_norm=True)
        self.practiced[fi] = True
        self.age = age
        self.n_updates[fi] += 1
        self.experience[fi] += weight
        self.total_updates += 1
        self._version += 1
        self.compute.param_updates += 1
        return LearningUpdateRecord(
            agent_id=self.agent_id,
            tick=obs.tick,
            policy_class=self.policy_class,
            family=obs.family,
            source=fb.source if fb.source in ("peer", "tutor", "school") else "peer",
            feedback_mode=fb.mode,
            n_samples=1,
            loss=loss,
            param_delta_norm=math.sqrt(sq),
            n_updates_total=self.total_updates,
        )

    def _remember_source(self, source_id: str, cred: float) -> None:
        """Bounded, insertion-ordered (deterministic) credibility table."""
        table = self.source_credibility
        if source_id in table:
            del table[source_id]
        table[source_id] = cred
        while len(table) > self.cfg.max_tracked_sources:
            del table[next(iter(table))]

    # ================================================================== lifecycle
    def tick_lifecycle(self, age_years: float, dt_years: float) -> None:
        cfg = self.cfg
        self.age = float(age_years)
        if not cfg.use_learning:
            return
        dt = max(0.0, float(dt_years))
        changed = False
        if cfg.use_forgetting and dt > 0.0:
            lam = cfg.forgetting_rate_per_year * forgetting_multiplier(age_years, cfg.use_plasticity_schedule)
            decay = math.exp(-lam * dt)
            noise_sd = cfg.forgetting_noise * math.sqrt(max(0.0, 1.0 - decay * decay))
            any_practiced = any(self.practiced)
            for fi in range(len(self.families)):
                if self.practiced[fi] or self.n_updates[fi] == 0:
                    continue
                self.H[fi] *= decay
                self.H[fi] += self._rng.normal(0.0, noise_sd, size=self.H[fi].shape)
                if self.hb[fi].size:
                    self.hb[fi] *= decay
                    self.hb[fi] += self._rng.normal(0.0, noise_sd, size=self.hb[fi].shape)
                changed = True
            if self.total_updates > 0:
                lam_e = lam * cfg.encoder_forgetting_fraction
                decay_e = math.exp(-lam_e * dt)
                noise_e = ENCODER_STATIONARY_SD * math.sqrt(max(0.0, 1.0 - decay_e * decay_e))
                for enc in range(len(self.E)):
                    users = [fi for fi in range(len(self.families)) if self.enc_of[fi] == enc]
                    if any_practiced and any(self.practiced[fi] for fi in users):
                        continue
                    if not any(self.n_updates[fi] > 0 for fi in users):
                        continue
                    self.E[enc] = self.E0[enc] + decay_e * (self.E[enc] - self.E0[enc])
                    self.E[enc] += self._rng.normal(0.0, noise_e, size=self.E[enc].shape)
                    changed = True
        if dt > 0.0:
            md = math.exp(-cfg.mastery_decay_per_year * dt)
            for fi in range(len(self.families)):
                if not self.practiced[fi] and self.n_feedback[fi] > 0:
                    self.alpha[fi] = 1.0 + md * (self.alpha[fi] - 1.0)
                    self.beta[fi] = 1.0 + md * (self.beta[fi] - 1.0)
                    changed = True
        before = self.memory.size
        self.memory.shrink(self._effective_capacity(age_years))
        changed = changed or self.memory.size != before or any(self.practiced)
        self.practiced = [False] * len(self.families)
        if changed:
            self._version += 1

    # ================================================================== introspection
    def self_estimates(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for f, fi in self.fam_index.items():
            mean, sd = self._mastery(fi)
            out[f] = {
                "mean": mean,
                "sd": sd,
                "alpha": self.alpha[fi],
                "beta": self.beta[fi],
                "n_feedback": self.n_feedback[fi],
            }
        return out

    def _param_hash(self) -> str:
        return stable_hash({"E": self.E, "H": self.H, "hb": self.hb}, length=10)

    def parameter_summary(self) -> dict[str, Any]:
        per_family: dict[str, dict[str, Any]] = {}
        for f, fi in self.fam_index.items():
            mean, sd = self._mastery(fi)
            per_family[f] = {
                "head_norm": float(np.linalg.norm(self.H[fi]) + np.linalg.norm(self.hb[fi])),
                "n_updates": int(self.n_updates[fi]),
                "mastery_mean": mean,
                "mastery_sd": sd,
                "last_practice_age": self.last_practice_age[fi],
            }
        return {
            "policy_class": self.policy_class,
            "version": self.version,
            "flags": self.cfg.flags(),
            "param_hash": self._param_hash(),
            "encoder_norm": [float(np.linalg.norm(e)) for e in self.E],
            "encoder_drift_from_init": [
                float(np.linalg.norm(e - e0)) for e, e0 in zip(self.E, self.E0, strict=True)
            ],
            "shared_representation": self.cfg.use_shared_representation,
            "families": per_family,
            "total_updates": int(self.total_updates),
            "plasticity": plasticity(self.age, self.cfg.use_plasticity_schedule) * self.lr_scale,
            "memory_items": int(self.memory.size),
            "memory_capacity_effective": self._effective_capacity(self.age) if self.cfg.use_memory else 0,
            "activity_observations": [int(v) for v in self.activity.n],
            "age": self.age,
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
            "enc_of": list(self.enc_of),
            "E": [e.tolist() for e in self.E],
            "E0": [e.tolist() for e in self.E0],
            "H": [h.tolist() for h in self.H],
            "hb": [b.tolist() for b in self.hb],
            "alpha": list(self.alpha),
            "beta": list(self.beta),
            "n_feedback": list(self.n_feedback),
            "ema_fast": list(self.ema_fast),
            "ema_slow": list(self.ema_slow),
            "n_updates": list(self.n_updates),
            "practiced": list(self.practiced),
            "last_practice_age": list(self.last_practice_age),
            "total_updates": int(self.total_updates),
            "experience": list(self.experience),
            "age": self.age,
            "practice_value": list(self.practice_value),
            "memory": self.memory.state_dict(),
            "activity": self.activity.state_dict(),
            "social": self.social.state_dict(),
            "rng": self._rng.bit_generator.state,
            "kappa": self.kappa,
            "lr_scale": self.lr_scale,
            "social_scale": self.social_scale,
            "source_credibility": [[k, v] for k, v in self.source_credibility.items()],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.cfg = DevConfig.from_params(state["config"])
        self._build(self.init)
        if list(state["families"]) != self.families:
            raise ValueError("state_dict families do not match this agent's AgentInit")
        self.enc_of = [int(v) for v in state["enc_of"]]
        self.E = [np.asarray(e, dtype=np.float64) for e in state["E"]]
        self.E0 = [np.asarray(e, dtype=np.float64) for e in state["E0"]]
        self.H = [np.asarray(h, dtype=np.float64).reshape(self.H[i].shape) for i, h in enumerate(state["H"])]
        self.hb = [
            np.asarray(b, dtype=np.float64).reshape(self.hb[i].shape) for i, b in enumerate(state["hb"])
        ]
        self.alpha = [float(v) for v in state["alpha"]]
        self.beta = [float(v) for v in state["beta"]]
        self.n_feedback = [float(v) for v in state["n_feedback"]]
        self.ema_fast = [float(v) for v in state["ema_fast"]]
        self.ema_slow = [float(v) for v in state["ema_slow"]]
        self.n_updates = [int(v) for v in state["n_updates"]]
        self.practiced = [bool(v) for v in state["practiced"]]
        self.last_practice_age = [float(v) for v in state["last_practice_age"]]
        self.total_updates = int(state["total_updates"])
        self.experience = [float(v) for v in state["experience"]]
        self.enc_users = [
            [fi for fi in range(len(self.families)) if self.enc_of[fi] == e] for e in range(len(self.E))
        ]
        self.age = float(state["age"])
        self.practice_value = [float(v) for v in state["practice_value"]]
        self.memory.load_state_dict(state["memory"])
        self.activity.load_state_dict(state["activity"])
        self.social.load_state_dict(state["social"])
        self._rng.bit_generator.state = state["rng"]
        self.kappa = float(state["kappa"])
        self.lr_scale = float(state["lr_scale"])
        self.social_scale = float(state["social_scale"])
        self.source_credibility = {str(k): float(v) for k, v in state["source_credibility"]}
        c = state.get("compute", {})
        self.compute.memory_items = int(c.get("memory_items", self.memory.size))
        self._version += 1

    def clone(self) -> Policy:
        other = DevelopmentalLearner(self.init, self.cfg)
        other.load_state_dict(self.state_dict())
        return other  # type: ignore[return-value]


def _uniform(obs: Observation) -> np.ndarray:
    m = obs.mask().astype(np.float64)
    s = m.sum()
    if s <= 0:
        return np.full(obs.n_actions, 1.0 / max(1, obs.n_actions))
    out: np.ndarray = m / s
    return out
