"""Gate E continual-learning comparators: a small value network trained online while it acts.

All three learners share one network and one policy and differ only in how online updates are protected:

* ``ft`` — online fine-tuning, no protection: one SGD step per trial on the newest transition;
* ``ewc`` — elastic weight consolidation (Kirkpatrick et al., 2017) in its task-agnostic online form
  (Schwarz et al., 2018): every ``period`` trials the diagonal empirical Fisher of the transitions since
  the last consolidation is added to the running Fisher and the anchor is moved to the current weights;
  every step adds ``lam · F ⊙ (θ − θ*)`` to the gradient. Context boundaries are **not** given (no learner
  in Gate E receives a context cue); ``boundaries="oracle"`` is a privileged diagnostic only;
* ``replay`` — experience replay: each step uses the newest transition plus ``replay_batch`` transitions
  drawn uniformly from a reservoir buffer of this sequence's past transitions (capacity ``buffer``).

Network: input = the last ``window`` trials, each as [one-hot(action), one-hot(action)·reward] (zero-padded
at the start of a sequence), tanh hidden layer, one value per option; loss ½(Q_a − r)²; softmax policy with
inverse temperature ``beta``. Initial weights are either a seeded random init (``init="scratch"``) or a
network pretrained offline to predict rewards on meta-train development sequences (``init="pretrained"``;
the same seeds F4 is meta-trained on, so both get the same information). Pure numpy, deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hpe.core.rng import stream
from hpe.population.continual.suite import K_OPTIONS, DeltaRuleAgent, SwitchingSequence, rollout


def feature_dim(window: int, K: int = K_OPTIONS) -> int:
    return window * 2 * K


class History:
    """Rolling window of (action, reward) encoded as the network input."""

    def __init__(self, window: int, K: int) -> None:
        self.window, self.K = window, K
        self.buf = np.zeros((window, 2 * K))

    def features(self) -> np.ndarray:
        return self.buf.reshape(-1).copy()

    def push(self, a: int, r: float) -> None:
        self.buf = np.roll(self.buf, 1, axis=0)
        self.buf[0] = 0.0
        self.buf[0, a] = 1.0
        self.buf[0, self.K + a] = r


@dataclass
class MLP:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @classmethod
    def init(cls, d: int, h: int, k: int, rng: np.random.Generator) -> MLP:
        return cls(
            rng.normal(0, 1 / np.sqrt(d), (h, d)),
            np.zeros(h),
            rng.normal(0, 1 / np.sqrt(h), (k, h)),
            np.full(k, 0.5),
        )

    def copy(self) -> MLP:
        return MLP(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def flat(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_flat(self, v: np.ndarray) -> None:
        i = 0
        for name in ("W1", "b1", "W2", "b2"):
            p = getattr(self, name)
            n = p.size
            setattr(self, name, v[i : i + n].reshape(p.shape).copy())
            i += n

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def q(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.W2 @ np.tanh(self.W1 @ x + self.b1) + self.b2)

    def grads(self, X: np.ndarray, A: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Per-sample flat gradients of ½(Q_a − r)², shape [n, P]."""
        H = np.tanh(X @ self.W1.T + self.b1)  # [n, h]
        Q = H @ self.W2.T + self.b2  # [n, k]
        n, k = Q.shape
        err = Q[np.arange(n), A] - R  # [n]
        dQ = np.zeros((n, k))
        dQ[np.arange(n), A] = err
        gW2 = dQ[:, :, None] * H[:, None, :]  # [n, k, h]
        gb2 = dQ
        dH = (dQ @ self.W2) * (1.0 - H**2)  # [n, h]
        gW1 = dH[:, :, None] * X[:, None, :]  # [n, h, d]
        gb1 = dH
        return np.concatenate([gW1.reshape(n, -1), gb1, gW2.reshape(n, -1), gb2], axis=1)


def softmax_choice(q: np.ndarray, beta: float, rng: np.random.Generator) -> int:
    z = beta * (q - q.max())
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(q.size, p=p))


class OnlineNetAgent:
    """Shared implementation of ``ft``, ``ewc`` and ``replay`` (see module docstring)."""

    def __init__(
        self,
        method: str,
        init_net: MLP,
        *,
        lr: float = 0.1,
        beta: float = 10.0,
        window: int = 10,
        lam: float = 0.0,
        period: int = 20,
        gamma: float = 1.0,
        buffer: int = 300,
        replay_batch: int = 8,
        boundaries: Sequence[int] | None = None,
        K: int = K_OPTIONS,
    ) -> None:
        if method not in ("ft", "ewc", "replay"):
            raise ValueError(method)
        self.method = method
        self.name = method
        self.init_net = init_net
        self.lr, self.beta, self.window = float(lr), float(beta), int(window)
        self.lam, self.period, self.gamma = float(lam), int(period), float(gamma)
        self.buffer, self.replay_batch = int(buffer), int(replay_batch)
        self.boundaries = set(boundaries) if boundaries is not None else None
        self.K = K
        self.rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.net = self.init_net.copy()
        self.hist = History(self.window, self.K)
        self.t = 0
        self.x_t: np.ndarray | None = None
        P = self.net.n_params()
        self.fisher = np.zeros(P)
        self.anchor = self.net.flat()
        self.since: list[tuple[np.ndarray, int, float]] = []
        self.buf_x: list[np.ndarray] = []
        self.buf_a: list[int] = []
        self.buf_r: list[float] = []
        self.n_seen = 0

    def act(self) -> int:
        assert self.rng is not None
        self.x_t = self.hist.features()
        return softmax_choice(self.net.q(self.x_t), self.beta, self.rng)

    def _consolidate(self) -> None:
        if self.since:
            X = np.stack([s[0] for s in self.since])
            A = np.array([s[1] for s in self.since])
            R = np.array([s[2] for s in self.since])
            g = self.net.grads(X, A, R)
            self.fisher = self.gamma * self.fisher + (g**2).mean(axis=0)
        self.anchor = self.net.flat()
        self.since = []

    def observe(self, action: int, reward: float) -> None:
        assert self.rng is not None and self.x_t is not None
        x = self.x_t
        if self.method == "ewc":
            due = (
                (self.t in self.boundaries)
                if self.boundaries is not None
                else (self.t > 0 and self.t % self.period == 0)
            )
            if due:
                self._consolidate()
        X, A, R = x[None, :], np.array([action]), np.array([reward])
        if self.method == "replay" and self.buf_x and self.replay_batch > 0:
            m = min(self.replay_batch, len(self.buf_x))
            idx = self.rng.integers(0, len(self.buf_x), size=m)
            X = np.vstack([X, np.stack([self.buf_x[i] for i in idx])])
            A = np.concatenate([A, np.array([self.buf_a[i] for i in idx])])
            R = np.concatenate([R, np.array([self.buf_r[i] for i in idx])])
        g = self.net.grads(X, A, R).mean(axis=0)
        theta = self.net.flat()
        if self.method == "ewc" and self.lam > 0:
            g = g + self.lam * self.fisher * (theta - self.anchor)
        gn = float(np.linalg.norm(g))
        if gn > 5.0:  # gradient clipping keeps large lam stable
            g *= 5.0 / gn
        self.net.set_flat(theta - self.lr * g)
        if self.method == "ewc":
            self.since.append((x, action, reward))
        if self.method == "replay":
            self.n_seen += 1
            if len(self.buf_x) < self.buffer:
                self.buf_x.append(x)
                self.buf_a.append(action)
                self.buf_r.append(reward)
            else:  # reservoir sampling
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.buffer:
                    self.buf_x[j], self.buf_a[j], self.buf_r[j] = x, action, reward
        self.hist.push(action, reward)
        self.t += 1


def scratch_net(seed: int, window: int = 10, hidden: int = 32, K: int = K_OPTIONS) -> MLP:
    return MLP.init(feature_dim(window, K), hidden, K, stream(seed, "gate_e_net_init", window, hidden))


def pretrain_net(
    seqs: Sequence[SwitchingSequence],
    *,
    seed: int,
    window: int = 10,
    hidden: int = 32,
    epochs: int = 20,
    lr: float = 0.05,
    batch: int = 32,
) -> tuple[MLP, dict[str, Any]]:
    """Offline reward-prediction pretraining on meta-train sequences (behaviour: delta-rule softmax agent)."""
    X, A, R = [], [], []
    for seq in seqs:
        traj = rollout(DeltaRuleAgent(0.3, 10.0), seq, stream(seed, "gate_e_pretrain_behaviour", seq.seed))
        h = History(window, seq.K)
        for a, r in zip(traj.actions, traj.rewards, strict=True):
            X.append(h.features())
            A.append(int(a))
            R.append(float(r))
            h.push(int(a), float(r))
    Xa, Aa, Ra = np.stack(X), np.array(A), np.array(R)
    net = scratch_net(seed, window, hidden)
    rng = stream(seed, "gate_e_pretrain_batches")
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(Xa))
        for i in range(0, len(order), batch):
            j = order[i : i + batch]
            g = net.grads(Xa[j], Aa[j], Ra[j]).mean(axis=0)
            net.set_flat(net.flat() - lr * g)
        q = (np.tanh(Xa @ net.W1.T + net.b1) @ net.W2.T + net.b2)[np.arange(len(Xa)), Aa]
        losses.append(float(np.mean((q - Ra) ** 2)))
    return net, {"n_transitions": len(Xa), "mse_by_epoch": [round(v, 5) for v in losses]}
