"""Phase II encoder and stronger reference models (all drive the same ``Policy`` protocol).

Encoder ``encode_generic`` turns ``GenericTrial`` records into (Observation, human action, Feedback,
forced) steps. Phase I families (MCPL, IGT) keep the Phase I encoding (``models.encode``).

Reference models (not simulator components; standard families from the literature):

* ``ref_exemplar``   — exemplar model with attention-weighted similarity (GCM: Nosofsky, 1986) and
                       error-driven attention learning (ALCOVE: Kruschke, 1992), recency decay, and a
                       softmax over similarity-weighted outcome evidence. Works on any context features.
* ``ref_pvl_delta``  — Prospect Valence Learning with delta rule (Ahn, Busemeyer, Wagenmakers & Stout,
                       2008; Ahn et al., 2014): utility u = x^a (gains), -l|x|^a (losses), delta update,
                       softmax sensitivity 3^c - 1. Context-free; for bandit-style families.
* ``ref_bayes_rule`` — Bayesian linear rule learner (Bayesian LMS / Kalman learner; Speekenbrink &
                       Shanks, 2010): Gaussian posterior over linear weights on the context, predictive
                       distribution binned onto the ordinal/binary response scale. Full-feedback families.
* ``ref_gru``        — small GRU sequence model (Cho et al., 2014) fitted by teacher forcing on the
                       fitting participants with early stopping on a held-out fifth of them; frozen at
                       evaluation. ``gru.py`` holds the numpy implementation.
* ``ref_bias``       — pooled choice frequencies (Phase I; generic).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from scipy.stats import norm

from hpe.contracts.runtime import AgentInit, Feedback, Observation, Policy, masked_softmax
from hpe.evaluation.human import gru
from hpe.evaluation.human.models import (
    IGT_ACTIONS,
    MCPL_ACTIONS,
    DeltaRuleModel,
    FixedDistributionModel,
    InstanceBasedModel,
    PopulationStaticModel,
    _RefPolicy,
    encode,
    pooled_choice_probs,
    population_table,
)
from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, PARADIGM_ACTIONS, GenericTrial

Step2 = tuple[Observation, int, Feedback, bool]  # observation, human action, feedback, forced

PARADIGM_N_ACTIONS: dict[str, int] = {"mcpl": MCPL_ACTIONS, "igt": IGT_ACTIONS, **PARADIGM_ACTIONS}
FAMILIES_META2: dict[str, dict[str, int]] = {
    par: {"n_slots": 1, "n_actions": k, "n_features": N_FEATURES} for par, k in PARADIGM_N_ACTIONS.items()
}
REFERENCE_MODELS2 = (
    "ref_bias",
    "ref_population_static",
    "ref_delta_rule",
    "ref_instance_based",
    "ref_exemplar",
    "ref_pvl_delta",
    "ref_bayes_rule",
    "ref_gru",
)
FULL_FEEDBACK_PARADIGMS = ("mcpl", "wpt", "garcia", "described")


# ----------------------------------------------------------------------------- encoding
def encode_generic(p: Participant, *, age_years: float = 25.0) -> list[Step2]:
    steps: list[Step2] = []
    prev_block = None
    for t, tr in enumerate(p.trials):
        assert isinstance(tr, GenericTrial)
        obs = Observation(
            p.key,
            t,
            age_years,
            "task",
            p.paradigm,
            np.asarray([tr.features], dtype=np.float32),
            tr.n_actions,
            meta={"block": tr.block, "block_start": tr.block != prev_block, "phase": tr.phase, **tr.meta},
        )
        prev_block = tr.block
        if tr.mode == "none":
            fb = Feedback(mode="none", source="human_data")
        elif tr.mode == "bandit":
            fb = Feedback(mode="bandit", reward=float(tr.reward or 0.0), source="human_data")
        else:
            ra = np.asarray(
                tr.rewards_all if tr.rewards_all is not None else np.eye(tr.n_actions)[tr.correct or 0],
                dtype=np.float64,
            )
            fb = Feedback(
                mode="full",
                reward=float(tr.reward or 0.0),
                correct_action=tr.correct,
                rewards_all=ra,
                source="human_data",
            )
        steps.append((obs, tr.choice, fb, tr.forced))
    return steps


def encode_any(p: Participant, *, igt_scale: float = 500.0, age_years: float = 25.0) -> list[Step2]:
    if p.paradigm in ("mcpl", "igt"):
        return [(o, a, f, False) for (o, a, f) in encode(p, scale=igt_scale, age_years=age_years)]
    return encode_generic(p, age_years=age_years)


def _target_of(obs: Observation, fb: Feedback) -> int | None:
    """Correct action for full feedback; MCPL carries it in the feedback, generic families too."""
    return fb.correct_action if fb.mode == "full" else None


# ----------------------------------------------------------------------------- exemplar (GCM/ALCOVE)
class ExemplarModel(_RefPolicy):
    policy_class: ClassVar[str] = "ref_exemplar"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init, params)
        self.c = float(params.get("c", 1.0))
        self.beta = float(params.get("beta", 2.0))
        self.decay = float(params.get("decay", 1.0))
        self.lr_att = float(params.get("lr_att", 0.0))
        self.kappa = float(params.get("kappa", 0.1))
        self.max_items = int(params.get("max_items", 600))
        self.att: dict[str, np.ndarray] = {}
        self.X: dict[str, list[np.ndarray]] = {}
        self.E: dict[str, list[np.ndarray]] = {}
        self.W: dict[str, list[float]] = {}

    def _arrays(self, fam: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if fam not in self.att:
            self.att[fam] = np.full(N_FEATURES, 1.0 / N_FEATURES)
        if not self.X.get(fam):
            return self.att[fam], np.zeros((0, N_FEATURES)), np.zeros((0, k)), np.zeros(0)
        return self.att[fam], np.asarray(self.X[fam]), np.asarray(self.E[fam]), np.asarray(self.W[fam])

    def _evidence(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        att, X, E, W = self._arrays(obs.family, obs.n_actions)
        if X.shape[0] == 0:
            return np.zeros(obs.n_actions), X, np.zeros(0), np.zeros((0, N_FEATURES))
        D = np.abs(obs.context[0].astype(np.float64)[None, :] - X)  # (n, F)
        S = np.exp(-self.c * (D @ att)) * W  # (n,)
        ev = (S[:, None] * E).sum(axis=0) / (S.sum() + self.kappa)
        return ev, X, S, D

    def _probs(self, obs: Observation) -> np.ndarray:
        ev, _, _, _ = self._evidence(obs)
        return masked_softmax(self.beta * ev, obs.mask())

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> None:
        self.compute.learn_calls += 1
        fam, k = obs.family, obs.n_actions
        self._arrays(fam, k)
        if feedback.mode == "none":
            return None
        if feedback.mode == "full" and feedback.rewards_all is not None:
            e = np.asarray(feedback.rewards_all, dtype=np.float64) - 0.5
            target = feedback.correct_action
        else:
            e = np.zeros(k)
            e[action] = float(feedback.reward) - 0.5
            target = None
        if self.lr_att > 0 and target is not None:
            ev, X, S, D = self._evidence(obs)
            if X.shape[0]:
                p = masked_softmax(self.beta * ev, None)
                g = p.copy()
                g[target] -= 1.0
                E = np.asarray(self.E[fam])
                denom = S.sum() + self.kappa
                dS = -self.c * D * S[:, None]  # (n, F)
                d_ev = (E.T @ dS) / denom  # (k, F)
                grad = self.beta * (g @ d_ev)
                att = np.clip(self.att[fam] - self.lr_att * grad, 1e-3, None)
                self.att[fam] = att / att.sum()
        # recency decay and insertion
        if self.decay < 1.0 and self.W.get(fam):
            self.W[fam] = [w * self.decay for w in self.W[fam]]
        self.X.setdefault(fam, []).append(obs.context[0].astype(np.float64).copy())
        self.E.setdefault(fam, []).append(e)
        self.W.setdefault(fam, []).append(1.0)
        if len(self.X[fam]) > self.max_items:
            for d in (self.X, self.E, self.W):
                d[fam].pop(0)
        self.compute.memory_items += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d.update(
            att={f: a.tolist() for f, a in self.att.items()}, n_items={f: len(x) for f, x in self.X.items()}
        )
        return d


# ----------------------------------------------------------------------------- PVL-Delta
class PVLDeltaModel(_RefPolicy):
    policy_class: ClassVar[str] = "ref_pvl_delta"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init, params)
        self.alpha = float(params.get("alpha", 0.5))
        self.lam = float(params.get("lam", 1.0))
        self.A = float(params.get("A", 0.2))
        self.theta = 3.0 ** float(params.get("c", 1.0)) - 1.0
        self.reset_on_block = bool(params.get("reset_on_block", True))
        self.ev: dict[str, np.ndarray] = {}

    def _v(self, obs: Observation) -> np.ndarray:
        if obs.family not in self.ev:
            self.ev[obs.family] = np.zeros(obs.n_actions)
        return self.ev[obs.family]

    def _probs(self, obs: Observation) -> np.ndarray:
        if self.reset_on_block and obs.meta.get("block_start") and obs.meta.get("block", 0) > 0:
            self.ev[obs.family] = np.zeros(obs.n_actions)  # new game: values start over
        return masked_softmax(self.theta * self._v(obs), obs.mask())

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> None:
        self.compute.learn_calls += 1
        if feedback.mode == "none":
            return None
        x = 2.0 * float(feedback.reward) - 1.0  # normalised reward -> signed outcome in [-1, 1]
        u = x**self.alpha if x >= 0 else -self.lam * (abs(x) ** self.alpha)
        v = self._v(obs)
        v[action] += self.A * (u - v[action])
        self.compute.param_updates += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["ev"] = {f: v.tolist() for f, v in self.ev.items()}
        return d


# ----------------------------------------------------------------------------- Bayesian rule learner
class BayesRuleModel(_RefPolicy):
    """Bayesian linear regression on the context; predictive Gaussian binned onto K ordered actions."""

    policy_class: ClassVar[str] = "ref_bayes_rule"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init, params)
        self.tau2 = float(params.get("tau2", 1.0))
        self.sigma = float(params.get("sigma", 1.0))
        self.temp = float(params.get("temp", 1.0))
        self.Lam: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}

    def _post(self, fam: str) -> tuple[np.ndarray, np.ndarray]:
        if fam not in self.Lam:
            self.Lam[fam] = np.eye(N_FEATURES + 1) / self.tau2
            self.b[fam] = np.zeros(N_FEATURES + 1)
        return self.Lam[fam], self.b[fam]

    @staticmethod
    def _x(obs: Observation) -> np.ndarray:
        return np.append(obs.context[0].astype(np.float64), 1.0)

    def _probs(self, obs: Observation) -> np.ndarray:
        Lam, b = self._post(obs.family)
        cov = np.linalg.inv(Lam)
        m = cov @ b
        x = self._x(obs)
        k = obs.n_actions
        scale = (k - 1) / 2.0  # targets live on 0..k-1; prior centred on the middle
        mu = scale + x @ m
        s = float(np.sqrt(x @ cov @ x + self.sigma**2)) * self.temp
        edges = np.arange(k + 1) - 0.5
        cdf = norm.cdf((edges - mu) / s)
        p = np.diff(cdf)
        p[0] += cdf[0]
        p[-1] += 1.0 - cdf[-1]
        p = np.clip(p, 1e-9, None)
        return p / p.sum()

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> None:
        self.compute.learn_calls += 1
        target = _target_of(obs, feedback)
        if target is None:
            return None
        Lam, b = self._post(obs.family)
        x = self._x(obs)
        y = float(target) - (obs.n_actions - 1) / 2.0
        Lam += np.outer(x, x) / self.sigma**2
        b += x * y / self.sigma**2
        self.compute.param_updates += 1
        return None

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["b"] = {f: v.tolist() for f, v in self.b.items()}
        return d


# ----------------------------------------------------------------------------- GRU policy wrapper
def gru_input_dim(k: int) -> int:
    return N_FEATURES + 2 * k + 2


def gru_input(
    x: np.ndarray, k: int, prev_action: int | None, prev_reward: float, prev_correct: int | None, had_fb: bool
) -> np.ndarray:
    v = np.zeros(gru_input_dim(k))
    v[:N_FEATURES] = x
    if prev_action is not None:
        v[N_FEATURES + prev_action] = 1.0
    v[N_FEATURES + k] = prev_reward
    if prev_correct is not None:
        v[N_FEATURES + k + 1 + prev_correct] = 1.0
    v[N_FEATURES + 2 * k + 1] = 1.0 if had_fb else 0.0
    return v


def sequences_for(parts: list[Participant], *, igt_scale: float = 500.0) -> list[gru.Sequence]:
    seqs = []
    for p in parts:
        steps = encode_any(p, igt_scale=igt_scale)
        k = steps[0][0].n_actions
        xs, acts, mask = [], [], []
        prev_a: int | None = None
        prev_r, had = 0.0, False
        prev_c: int | None = None
        for obs, a, fb, forced in steps:
            xs.append(gru_input(obs.context[0].astype(np.float64), k, prev_a, prev_r, prev_c, had))
            acts.append(a)
            mask.append(not forced)
            prev_a, had = a, fb.mode != "none"
            prev_r = float(fb.reward) if fb.mode != "none" else 0.0
            prev_c = fb.correct_action if fb.mode == "full" else None
        seqs.append((np.asarray(xs), np.asarray(acts), np.asarray(mask, dtype=bool)))
    return seqs


class GRUPolicy(_RefPolicy):
    policy_class: ClassVar[str] = "ref_gru"

    def __init__(self, init: AgentInit, params: dict[str, Any]) -> None:
        super().__init__(init, params)
        self.P = gru.GRUParams.from_json(params["weights"])
        self.k = int(params["k"])
        self.h = np.zeros(self.P.bz.shape[0])
        self.prev_a: int | None = None
        self.prev_r, self.had = 0.0, False
        self.prev_c: int | None = None
        self._cached: np.ndarray | None = None

    def _probs(self, obs: Observation) -> np.ndarray:
        x = gru_input(
            obs.context[0].astype(np.float64), self.k, self.prev_a, self.prev_r, self.prev_c, self.had
        )
        _, p, _ = gru.step(self.P, x, self.h)
        return p

    def learn(self, obs: Observation, action: int, feedback: Feedback) -> None:
        self.compute.learn_calls += 1
        x = gru_input(
            obs.context[0].astype(np.float64), self.k, self.prev_a, self.prev_r, self.prev_c, self.had
        )
        self.h, _, _ = gru.step(self.P, x, self.h)
        self.prev_a, self.had = action, feedback.mode != "none"
        self.prev_r = float(feedback.reward) if feedback.mode != "none" else 0.0
        self.prev_c = feedback.correct_action if feedback.mode == "full" else None
        return None

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d.update(h=self.h.tolist(), prev_a=self.prev_a)
        d["params"] = {k: v for k, v in self.params.items() if k != "weights"}
        return d


def fit_gru(
    train: list[Participant], params: dict[str, Any], paradigm: str, *, seed: int, igt_scale: float = 500.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on 80% of the fitting participants, early-stop on the other 20%. Returns (weights params, log)."""
    k = PARADIGM_N_ACTIONS[paradigm]
    seqs = sequences_for(train, igt_scale=igt_scale)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(seqs))
    n_val = max(1, len(seqs) // 5)
    val = [seqs[i] for i in order[:n_val]]
    fit = [seqs[i] for i in order[n_val:]] or val
    hidden = int(params.get("hidden", 8))
    P, log = gru.train_gru(
        fit,
        val,
        n_in=gru_input_dim(k),
        n_hidden=hidden,
        n_out=k,
        lr=float(params.get("lr", 0.01)),
        max_epochs=int(params.get("max_epochs", 15)),
        patience=int(params.get("patience", 4)),
        window=int(params.get("window", 64)),
        seed=seed,
        max_steps_per_epoch=params.get("max_steps_per_epoch"),
    )
    return {"weights": P.to_json(), "k": k, "hidden": hidden, "param_count": P.count()}, log


# ----------------------------------------------------------------------------- factories
FitCache = dict[str, Any]


def reference_factory2(
    name: str,
    params: dict[str, Any],
    train: list[Participant],
    paradigm: str,
    *,
    seed: int = 0,
    cache: FitCache | None = None,
) -> Callable[[AgentInit], Policy]:
    """Per-participant factory for a reference model; models needing fitting are fitted on ``train`` only."""
    k = PARADIGM_N_ACTIONS[paradigm]
    if name == "ref_bias":
        probs = pooled_choice_probs(train, k, params.get("smoothing", 1.0)).tolist()
        p2 = {**params, "probs": {paradigm: probs}}
        return lambda init: FixedDistributionModel(init, p2)  # type: ignore[return-value]
    if name == "ref_population_static":
        if paradigm != "mcpl":
            raise ValueError("ref_population_static is defined for MCPL only")
        p2 = {**params, "table": population_table(train, params.get("smoothing", 1.0)), "fallback": {}}
        return lambda init: PopulationStaticModel(init, p2)  # type: ignore[return-value]
    if name == "ref_delta_rule":
        if paradigm not in ("mcpl", "igt"):
            raise ValueError("ref_delta_rule (Phase I) is defined for MCPL and IGT only")
        return lambda init: DeltaRuleModel(init, params)  # type: ignore[return-value]
    if name == "ref_instance_based":
        if paradigm not in ("mcpl", "igt"):
            raise ValueError("ref_instance_based (Phase I) is defined for MCPL and IGT only")
        return lambda init: InstanceBasedModel(init, params)  # type: ignore[return-value]
    if name == "ref_exemplar":
        return lambda init: ExemplarModel(init, params)  # type: ignore[return-value]
    if name == "ref_pvl_delta":
        return lambda init: PVLDeltaModel(init, params)  # type: ignore[return-value]
    if name == "ref_bayes_rule":
        if paradigm not in FULL_FEEDBACK_PARADIGMS:
            raise ValueError("ref_bayes_rule needs full (corrective) feedback")
        return lambda init: BayesRuleModel(init, params)  # type: ignore[return-value]
    if name == "ref_gru":
        key = f"gru:{paradigm}:{sorted(params.items())}:{seed}"
        if cache is not None and key in cache:
            weights = cache[key]
        else:
            weights, log = fit_gru(
                train, params, paradigm, seed=seed, igt_scale=float(params.get("igt_reward_scale", 500.0))
            )
            weights["fit_log"] = log
            if cache is not None:
                cache[key] = weights
        wp = {**params, **{kk: v for kk, v in weights.items() if kk != "fit_log"}}
        return lambda init: GRUPolicy(init, wp)  # type: ignore[return-value]
    raise KeyError(name)


def parameter_count(name: str, params: dict[str, Any], paradigm: str) -> int | None:
    """Free parameters fitted per model (hyper-parameters count 1 each; GRU weights counted exactly)."""
    k = PARADIGM_N_ACTIONS[paradigm]
    return {
        "ref_bias": k - 1,
        "ref_population_static": 25 * (k - 1),
        "ref_delta_rule": 2,
        "ref_instance_based": 3,
        "ref_exemplar": 4 + N_FEATURES,
        "ref_pvl_delta": 4,
        "ref_bayes_rule": 2 + N_FEATURES + 1,
        "ref_gru": gru.param_count(gru_input_dim(k), int(params.get("hidden", 8)), k),
    }.get(name)
