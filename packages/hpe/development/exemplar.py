"""Exemplar store with a learnable similarity kernel (Candidate A, exemplar pathway).

One store per task family. Each exemplar is a context (``n_slots`` x ``d``), the information the
world revealed about it (a label under instructional feedback, or an action and reward under outcome
feedback), and its insertion time (for recency weighting).

Evidence for a query ``x``::

    K_i   = exp(-sum_j w_j (x_j - x_ij)^2 / (2 h^2))      learned diagonal metric (w = exp(u))
    rec_i = decay ** (now - t_i)
    evid  = sum_i K_i rec_i T_i                            T_i: per-exemplar target over actions
    S     = Ky @ evid + kappa                              Ky: ordinal action kernel (identity if not ordinal)
    p_ex  = S / sum(S)

Targets: label -> one-hot; rewarded outcome (r >= 0.5) -> r * one-hot(action); unrewarded outcome ->
(1 - r) spread over the other actions (negative evidence). Pair families (two slots, choose the better
slot) also use the slot-swapped exemplar with the complementary label, so evidence is order-invariant.

The kernel weights are learned by gradient ascent on log p_ex[y] (GCM-style attention learning).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ordinal_kernel(n_actions: int, spread: float) -> np.ndarray:
    a = np.arange(n_actions, dtype=np.float64)
    k = np.exp(-((a[:, None] - a[None, :]) ** 2) / (2.0 * spread * spread))
    return k / k.sum(axis=1, keepdims=True)


class ExemplarStore:
    __slots__ = (
        "T",
        "X",
        "action",
        "capacity",
        "d",
        "label",
        "mode",
        "n_actions",
        "n_slots",
        "ptr",
        "reward",
        "size",
        "t",
    )

    def __init__(self, capacity: int, n_slots: int, d: int, n_actions: int) -> None:
        self.capacity = max(1, int(capacity))
        self.n_slots = int(n_slots)
        self.d = int(d)
        self.n_actions = int(n_actions)
        self.X = np.zeros((self.capacity, self.n_slots, self.d), dtype=np.float32)
        self.T = np.zeros((self.capacity, self.n_actions), dtype=np.float32)
        self.label = np.full(self.capacity, -1, dtype=np.int16)
        self.action = np.zeros(self.capacity, dtype=np.int16)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.mode = np.zeros(self.capacity, dtype=np.int8)  # 1 = label (full), 0 = outcome (bandit)
        self.t = np.zeros(self.capacity, dtype=np.int64)
        self.size = 0
        self.ptr = 0

    # ------------------------------------------------------------------ writing
    def add(self, x: np.ndarray, label: int | None, action: int, reward: float, now: int) -> int:
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = self.ptr % self.capacity
            self.ptr += 1
        self.X[idx] = x[: self.n_slots]
        tgt = np.zeros(self.n_actions, dtype=np.float32)
        if label is not None and 0 <= label < self.n_actions:
            tgt[label] = 1.0
            self.mode[idx] = 1
        else:
            r = float(np.clip(reward, 0.0, 1.0))
            if r >= 0.5:
                tgt[action] = r
            elif self.n_actions > 1:
                tgt[:] = (1.0 - r) / (self.n_actions - 1)
                tgt[action] = 0.0
            self.mode[idx] = 0
        self.T[idx] = tgt
        self.label[idx] = -1 if label is None else int(label)
        self.action[idx] = int(action)
        self.reward[idx] = float(reward)
        self.t[idx] = int(now)
        return idx

    def remove(self, indices: np.ndarray) -> None:
        """Compact the store, dropping ``indices`` (forgetting)."""
        if self.size == 0 or indices.size == 0:
            return
        keep = np.ones(self.size, dtype=bool)
        keep[indices[indices < self.size]] = False
        n = int(keep.sum())
        for name in ("X", "T", "label", "action", "reward", "mode", "t"):
            arr = getattr(self, name)
            arr[:n] = arr[: self.size][keep]
        self.size = n
        self.ptr = 0

    # ------------------------------------------------------------------ reading
    def _stored(self, proj: np.ndarray | None, swap: bool = False) -> np.ndarray:
        X = self.X[: self.size].astype(np.float64)
        if swap and self.n_slots == 2:
            X = X[:, ::-1, :]
        return X @ proj.T if proj is not None else X

    def _sqdist(
        self, q: np.ndarray, w: np.ndarray, swap: bool = False, proj: np.ndarray | None = None
    ) -> np.ndarray:
        """Weighted squared distances (B, N) between queries (B, n_slots, k) and stored contexts.

        ``proj`` (k, d) maps stored raw contexts into the query space (latent-space kernel); the
        query must already be in that space.
        """
        X = self._stored(proj, swap)
        diff = q[:, None, :, :] - X[None, :, :, :]  # (B, N, S, k)
        return np.einsum("bnsd,d->bn", diff * diff, w)

    def evidence(
        self,
        q: np.ndarray,
        w: np.ndarray,
        h: float,
        now: int,
        decay: float,
        ky: np.ndarray | None,
        kappa: float,
        proj: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_ex (B, A), retrieval mass (B,), per-exemplar weights K*rec (B, N))."""
        B = q.shape[0]
        A = self.n_actions
        if self.size == 0:
            return np.full((B, A), 1.0 / A), np.zeros(B), np.zeros((B, 0))
        rec = decay ** (now - self.t[: self.size]).astype(np.float64) if decay < 1.0 else np.ones(self.size)
        K = np.exp(-self._sqdist(q, w, proj=proj) / (2.0 * h * h)) * rec[None, :]
        T = self.T[: self.size].astype(np.float64)
        evid = K @ T
        if self.n_slots == 2:
            Ks = np.exp(-self._sqdist(q, w, swap=True, proj=proj) / (2.0 * h * h)) * rec[None, :]
            evid += Ks @ T[:, ::-1]
            K = K + Ks
        S = (evid @ ky.T if ky is not None else evid) + kappa
        p = S / S.sum(axis=1, keepdims=True)
        return p, K.sum(axis=1), K

    def kernel_gradient(
        self,
        q: np.ndarray,
        w: np.ndarray,
        h: float,
        K: np.ndarray,
        G: np.ndarray,
        proj: np.ndarray | None = None,
    ) -> np.ndarray:
        """d log p_ex[event] / d w for one query (B = 1); ``G`` = d log p[event] / d evid (A,)."""
        if self.size == 0:
            return np.zeros_like(w)
        T = self.T[: self.size].astype(np.float64)
        c = (T @ G) * K[0]  # (N,) coefficient per exemplar
        X = self._stored(proj)
        diff = q[0][None, :, :] - X  # (N, S, k)
        D = (diff * diff).sum(axis=1)  # (N, k)
        g = -(c @ D) / (2.0 * h * h)
        if self.n_slots == 2:
            diff_s = q[0][None, :, :] - self._stored(proj, swap=True)
            D_s = (diff_s * diff_s).sum(axis=1)
            g = g - (c @ D_s) / (2.0 * h * h)
        return g

    def parametric_capture(self, p_par: np.ndarray) -> np.ndarray:
        """Per-exemplar probability the parametric pathway assigns to the exemplar's target (N,)."""
        if self.size == 0:
            return np.zeros(0)
        T = self.T[: self.size].astype(np.float64)
        tn = T / np.maximum(T.sum(axis=1, keepdims=True), 1e-12)
        return (p_par * tn).sum(axis=1)

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        s = self.size
        return {
            "capacity": self.capacity,
            "n_slots": self.n_slots,
            "d": self.d,
            "n_actions": self.n_actions,
            "size": s,
            "ptr": self.ptr,
            "X": self.X[:s].tolist(),
            "T": self.T[:s].tolist(),
            "label": self.label[:s].tolist(),
            "action": self.action[:s].tolist(),
            "reward": self.reward[:s].tolist(),
            "mode": self.mode[:s].tolist(),
            "t": self.t[:s].tolist(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.__init__(int(d["capacity"]), int(d["n_slots"]), int(d["d"]), int(d["n_actions"]))  # type: ignore[misc]
        s = int(d["size"])
        self.size = s
        self.ptr = int(d["ptr"])
        if s:
            self.X[:s] = np.asarray(d["X"], dtype=np.float32).reshape(s, self.n_slots, self.d)
            self.T[:s] = np.asarray(d["T"], dtype=np.float32).reshape(s, self.n_actions)
            self.label[:s] = np.asarray(d["label"], dtype=np.int16)
            self.action[:s] = np.asarray(d["action"], dtype=np.int16)
            self.reward[:s] = np.asarray(d["reward"], dtype=np.float32)
            self.mode[:s] = np.asarray(d["mode"], dtype=np.int8)
            self.t[:s] = np.asarray(d["t"], dtype=np.int64)
