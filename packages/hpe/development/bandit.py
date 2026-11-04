"""Discounted Bayesian linear contextual bandit (activity choice and trust game).

Each action keeps a ridge/Bayesian linear-regression posterior over reward given context:

    A_a <- g * A_a + (1 - g) * lambda * I + x x^T        b_a <- g * b_a + r x
    theta_a = A_a^-1 b_a        sigma_a(x) = s_r * sqrt(x^T A_a^-1 x)

Discounting (g < 1) lets the posterior track non-stationary rewards (needs change with age);
the (1 - g) * lambda * I term keeps the prior from vanishing. Action probabilities are a
deterministic optimistic softmax over ``theta_a . x + ucb * sigma_a(x)`` with a temperature
proportional to the running reward scale, so exploration is driven by posterior uncertainty
and the distribution is a pure function of state (required by the Policy contract).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LinearBandit:
    __slots__ = (
        "A",
        "A_inv",
        "b",
        "discount",
        "n",
        "n_actions",
        "n_features",
        "prior",
        "r_mean",
        "r_sq",
        "theta",
    )

    def __init__(self, n_actions: int, n_features: int, *, discount: float, prior: float) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.discount = float(discount)
        self.prior = float(prior)
        self.A = np.stack([np.eye(n_features) * prior for _ in range(n_actions)])
        self.b = np.zeros((n_actions, n_features))
        self.n = np.zeros(n_actions)
        self.r_mean = 0.0
        self.r_sq = 0.0
        self.A_inv = np.empty_like(self.A)
        self.theta = np.empty_like(self.b)
        self._refresh_all()

    # ------------------------------------------------------------------ derived quantities
    def _refresh(self, a: int) -> None:
        self.A_inv[a] = np.linalg.inv(self.A[a])
        self.theta[a] = self.A_inv[a] @ self.b[a]

    def _refresh_all(self) -> None:
        for a in range(self.n_actions):
            self._refresh(a)

    @property
    def reward_scale(self) -> float:
        n = float(self.n.sum())
        if n < 2:
            return 1.0
        var = max(0.0, self.r_sq - self.r_mean * self.r_mean)
        return max(0.05, float(np.sqrt(var)))

    def _fit_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] == self.n_features:
            return x
        out = np.zeros(self.n_features)
        m = min(self.n_features, x.shape[0])
        out[:m] = x[:m]
        return out

    def values(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean reward and (scaled) posterior sd for every action."""
        x = self._fit_x(x)
        mean = self.theta @ x
        var = np.einsum("i,aij,j->a", x, self.A_inv, x)
        sd = self.reward_scale * np.sqrt(np.maximum(var, 0.0))
        return mean, sd

    def logits(self, x: np.ndarray, ucb: float, temperature: float) -> np.ndarray:
        mean, sd = self.values(x)
        tau = max(1e-3, temperature * self.reward_scale)
        out: np.ndarray = (mean + ucb * sd) / tau
        return out

    # ------------------------------------------------------------------ learning
    def update(self, a: int, x: np.ndarray, r: float) -> float:
        """Return the norm of the change in theta_a."""
        x = self._fit_x(x)
        g = self.discount
        before = self.theta[a].copy()
        self.A[a] = g * self.A[a] + (1.0 - g) * self.prior * np.eye(self.n_features) + np.outer(x, x)
        self.b[a] = g * self.b[a] + r * x
        self.n[a] += 1
        k = float(self.n.sum())
        self.r_mean += (r - self.r_mean) / min(k, 200.0)
        self.r_sq += (r * r - self.r_sq) / min(k, 200.0)
        self._refresh(a)
        return float(np.linalg.norm(self.theta[a] - before))

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "n_actions": self.n_actions,
            "n_features": self.n_features,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "n": self.n.tolist(),
            "r_mean": self.r_mean,
            "r_sq": self.r_sq,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.n_actions = int(d["n_actions"])
        self.n_features = int(d["n_features"])
        self.A = np.asarray(d["A"], dtype=np.float64)
        self.b = np.asarray(d["b"], dtype=np.float64)
        self.n = np.asarray(d["n"], dtype=np.float64)
        self.r_mean = float(d["r_mean"])
        self.r_sq = float(d["r_sq"])
        self.A_inv = np.empty_like(self.A)
        self.theta = np.empty_like(self.b)
        self._refresh_all()
