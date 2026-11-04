"""Small GRU sequence model in numpy (manual backprop) — the "modern sequence-model baseline".

Reference: Cho et al. (2014), "Learning Phrase Representations using RNN Encoder-Decoder for
Statistical Machine Translation" (GRU). Trained with Adam (Kingma & Ba, 2015) by teacher forcing on
human choice sequences; truncated BPTT in windows with the hidden state carried across windows.

Input at step t: [context features (F), one-hot previous action (K), previous reward (1),
one-hot previous correct action (K), previous-feedback-present flag (1)]. Output: K logits.
Parameter count = 3 * H * (I + H + 1) + K * (H + 1); kept <= 2,000 by the grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass(slots=True)
class GRUParams:
    Wz: np.ndarray
    Wr: np.ndarray
    Wh: np.ndarray
    bz: np.ndarray
    br: np.ndarray
    bh: np.ndarray
    Wo: np.ndarray
    bo: np.ndarray

    def as_list(self) -> list[np.ndarray]:
        return [self.Wz, self.Wr, self.Wh, self.bz, self.br, self.bh, self.Wo, self.bo]

    def count(self) -> int:
        return int(sum(p.size for p in self.as_list()))

    def copy(self) -> GRUParams:
        return GRUParams(*[p.copy() for p in self.as_list()])

    def to_json(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("Wz", "Wr", "Wh", "bz", "br", "bh", "Wo", "bo")}

    @classmethod
    def from_json(cls, d: dict) -> GRUParams:
        return cls(
            *[np.asarray(d[k], dtype=np.float64) for k in ("Wz", "Wr", "Wh", "bz", "br", "bh", "Wo", "bo")]
        )


def init_params(n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator) -> GRUParams:
    s = 1.0 / np.sqrt(n_in + n_hidden)
    g = lambda: rng.uniform(-s, s, size=(n_hidden, n_in + n_hidden))  # noqa: E731
    return GRUParams(
        g(),
        g(),
        g(),
        np.zeros(n_hidden),
        np.zeros(n_hidden),
        np.zeros(n_hidden),
        rng.uniform(-s, s, size=(n_out, n_hidden)),
        np.zeros(n_out),
    )


def param_count(n_in: int, n_hidden: int, n_out: int) -> int:
    return 3 * n_hidden * (n_in + n_hidden + 1) + n_out * (n_hidden + 1)


@dataclass(slots=True)
class _Cache:
    xh: np.ndarray
    z: np.ndarray
    r: np.ndarray
    hc: np.ndarray
    h_prev: np.ndarray
    h: np.ndarray
    p: np.ndarray


def step(P: GRUParams, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, _Cache]:
    xh = np.concatenate([x, h_prev])
    z = _sigmoid(P.Wz @ xh + P.bz)
    r = _sigmoid(P.Wr @ xh + P.br)
    xrh = np.concatenate([x, r * h_prev])
    hc = np.tanh(P.Wh @ xrh + P.bh)
    h = (1 - z) * h_prev + z * hc
    logits = P.Wo @ h + P.bo
    logits = logits - logits.max()
    e = np.exp(logits)
    p = e / e.sum()
    return h, p, _Cache(xh, z, r, hc, h_prev, h, p)


def _zeros_like(P: GRUParams) -> GRUParams:
    return GRUParams(*[np.zeros_like(p) for p in P.as_list()])


def forward_backward(
    P: GRUParams, xs: np.ndarray, actions: np.ndarray, mask: np.ndarray, h0: np.ndarray
) -> tuple[float, GRUParams, np.ndarray]:
    """Loss = mean over masked steps of -log p(action). Returns (loss, grads, final hidden)."""
    H = h0.shape[0]
    n_in = xs.shape[1]
    caches: list[_Cache] = []
    h = h0
    loss = 0.0
    n_eff = max(1, int(mask.sum()))
    for t in range(xs.shape[0]):
        h, p, c = step(P, xs[t], h)
        caches.append(c)
        if mask[t]:
            loss -= np.log(max(p[actions[t]], 1e-12))
    loss /= n_eff
    G = _zeros_like(P)
    dh_next = np.zeros(H)
    for t in range(xs.shape[0] - 1, -1, -1):
        c = caches[t]
        dlogits = np.zeros_like(c.p)
        if mask[t]:
            dlogits = c.p.copy()
            dlogits[actions[t]] -= 1.0
            dlogits /= n_eff
        G.Wo += np.outer(dlogits, c.h)
        G.bo += dlogits
        dh = P.Wo.T @ dlogits + dh_next
        dz = dh * (c.hc - c.h_prev)
        dhc = dh * c.z
        dh_prev = dh * (1 - c.z)
        dhc_pre = dhc * (1 - c.hc**2)
        xrh = np.concatenate([xs[t], c.r * c.h_prev])
        G.Wh += np.outer(dhc_pre, xrh)
        G.bh += dhc_pre
        dxrh = P.Wh.T @ dhc_pre
        drh = dxrh[n_in:]
        dr = drh * c.h_prev
        dh_prev += drh * c.r
        dz_pre = dz * c.z * (1 - c.z)
        dr_pre = dr * c.r * (1 - c.r)
        G.Wz += np.outer(dz_pre, c.xh)
        G.bz += dz_pre
        G.Wr += np.outer(dr_pre, c.xh)
        G.br += dr_pre
        dxh = P.Wz.T @ dz_pre + P.Wr.T @ dr_pre
        dh_prev += dxh[n_in:]
        dh_next = dh_prev
    return float(loss), G, h


@dataclass(slots=True)
class Adam:
    lr: float
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: list[np.ndarray] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)

    def update(self, P: GRUParams, G: GRUParams, clip: float = 5.0) -> None:
        ps, gs = P.as_list(), G.as_list()
        norm = float(np.sqrt(sum(float((g**2).sum()) for g in gs)))
        if norm > clip:
            gs = [g * (clip / norm) for g in gs]
        if not self.m:
            self.m = [np.zeros_like(p) for p in ps]
            self.v = [np.zeros_like(p) for p in ps]
        self.t += 1
        for i, (p, g) in enumerate(zip(ps, gs, strict=True)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


Sequence = tuple[np.ndarray, np.ndarray, np.ndarray]  # inputs (T, I), actions (T,), mask (T,)


def sequence_nll(P: GRUParams, seq: Sequence) -> float:
    xs, actions, mask = seq
    h = np.zeros(P.bz.shape[0])
    tot, n = 0.0, 0
    for t in range(xs.shape[0]):
        h, p, _ = step(P, xs[t], h)
        if mask[t]:
            tot -= np.log(max(p[actions[t]], 1e-12))
            n += 1
    return tot / max(1, n)


def train_gru(
    train: list[Sequence],
    val: list[Sequence],
    *,
    n_in: int,
    n_hidden: int,
    n_out: int,
    lr: float,
    max_epochs: int,
    patience: int,
    window: int,
    seed: int,
    max_steps_per_epoch: int | None = None,
) -> tuple[GRUParams, dict]:
    """Adam + truncated BPTT, early stopping on ``val`` mean NLL. Returns best params and a log."""
    rng = np.random.default_rng(seed)
    P = init_params(n_in, n_hidden, n_out, rng)
    opt = Adam(lr)
    best, best_val, bad = P.copy(), float("inf"), 0
    log: dict = {"epochs": [], "param_count": P.count()}
    for epoch in range(max_epochs):
        order = rng.permutation(len(train))
        steps_done = 0
        tr_loss, tr_n = 0.0, 0
        for i in order:
            xs, actions, mask = train[i]
            h = np.zeros(n_hidden)
            for s in range(0, xs.shape[0], window):
                loss, G, h = forward_backward(
                    P, xs[s : s + window], actions[s : s + window], mask[s : s + window], h
                )
                opt.update(P, G)
                tr_loss += loss
                tr_n += 1
                steps_done += min(window, xs.shape[0] - s)
            if max_steps_per_epoch and steps_done >= max_steps_per_epoch:
                break
        val_nll = float(np.mean([sequence_nll(P, s) for s in val])) if val else float(tr_loss / max(1, tr_n))
        log["epochs"].append({"epoch": epoch, "train_loss": tr_loss / max(1, tr_n), "val_nll": val_nll})
        if val_nll < best_val - 1e-4:
            best_val, best, bad = val_nll, P.copy(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    log["best_val_nll"] = best_val
    return best, log
