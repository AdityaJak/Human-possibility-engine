"""Torch substrate shared by comparators and candidates: batching, masked log-softmax, a generic
early-stopping trainer, and a population RNN module. Requires the optional ``research`` extra."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.core.rng import derive_seed, stream
from hpe.population.core.encoding import EncodedSeq


def setup_torch(seed: int, threads: int, *key: str) -> int:
    s = derive_seed(seed, "torch", *key) % (2**31 - 1)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, int(threads)))
    return s


class Batch:
    __slots__ = ("B", "T", "a", "avail", "keys", "mask", "x")

    def __init__(self, seqs: Sequence[EncodedSeq]) -> None:
        self.B = len(seqs)
        self.T = max(s.T for s in seqs)
        k = seqs[0].k_max
        d = seqs[0].x.shape[1]
        x = np.zeros((self.B, self.T, d), dtype=np.float32)
        a = np.zeros((self.B, self.T), dtype=np.int64)
        av = np.zeros((self.B, self.T, k), dtype=bool)
        av[:, :, 0] = True  # padded steps: keep log-softmax finite; they are masked out of every loss
        m = np.zeros((self.B, self.T), dtype=bool)
        for i, s in enumerate(seqs):
            x[i, : s.T] = s.x
            a[i, : s.T] = s.actions
            av[i, : s.T] = s.avail
            m[i, : s.T] = s.scored
        self.x = torch.from_numpy(x)
        self.a = torch.from_numpy(a)
        self.avail = torch.from_numpy(av)
        self.mask = torch.from_numpy(m)
        self.keys = [s.key for s in seqs]


def masked_log_softmax(logits: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~avail, float("-inf")), dim=-1)


def nll_sum(logp: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, int]:
    """Sum of -log p(chosen) over scored steps, and the number of scored steps."""
    lp = logp.gather(-1, batch.a.unsqueeze(-1)).squeeze(-1)
    lp = torch.where(batch.mask, lp, torch.zeros_like(lp))
    return -lp.sum(), int(batch.mask.sum())


def split_train_val(
    seqs: Sequence[EncodedSeq], seed: int, key: str, frac: float = 0.2
) -> tuple[list[EncodedSeq], list[EncodedSeq]]:
    rng = stream(seed, "phase3_val_split", key)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * frac)))
    val = [seqs[i] for i in sorted(order[:n_val])]
    fit = [seqs[i] for i in sorted(order[n_val:])] or val
    return fit, val


LossFn = Callable[[nn.Module, Batch], tuple[torch.Tensor, int]]


def train(
    model: nn.Module,
    loss_fn: LossFn,
    seqs: Sequence[EncodedSeq],
    *,
    seed: int,
    key: str,
    lr: float = 0.01,
    max_epochs: int = 40,
    patience: int = 5,
    batch_size: int = 16,
    budget_s: float = 2700.0,
    weight_decay: float = 0.0,
    clip: float = 1.0,
    val_loss_fn: LossFn | None = None,
) -> dict[str, Any]:
    """Adam with early stopping on a seeded 20 % participant split of ``seqs``; restores the best weights."""
    fit, val = split_train_val(seqs, seed, key)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = stream(seed, "phase3_batches", key)
    vfn = val_loss_fn or loss_fn
    val_batches = [Batch(val[i : i + batch_size]) for i in range(0, len(val), batch_size)]

    def val_nll() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in val_batches:
                s, k = vfn(model, b)
                tot += float(s)
                n += k
        model.train()
        return tot / max(n, 1)

    best = val_nll()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.process_time()
    epochs, bad, history = 0, 0, [best]
    stopped = "max_epochs"
    for _ep in range(max_epochs):
        order = rng.permutation(len(fit))
        for i in range(0, len(fit), batch_size):
            b = Batch([fit[j] for j in order[i : i + batch_size]])
            s, k = loss_fn(model, b)
            loss = s / max(k, 1)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        epochs += 1
        v = val_nll()
        history.append(v)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            stopped = "early_stop"
            break
        if time.process_time() - t0 > budget_s:
            stopped = "budget"
            break
    model.load_state_dict(best_state)
    model.eval()
    return {
        "epochs": epochs,
        "best_val_nll": best,
        "val_history": [round(h, 5) for h in history],
        "stopped": stopped,
        "n_fit": len(fit),
        "n_val": len(val),
        "cpu_s": round(time.process_time() - t0, 2),
    }


class PopRNN(nn.Module):
    """Population sequence model: RNN over encoding v3 (+ optional extra inputs) -> K logits."""

    def __init__(
        self, d_in: int, hidden: int, k: int, cell: str = "gru", layers: int = 1, extra_in: int = 0
    ) -> None:
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.cell = cell
        self.rnn = rnn_cls(d_in + extra_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, k)

    def forward(
        self, x: torch.Tensor, avail: torch.Tensor, extra: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = x if extra is None else torch.cat([x, extra], dim=-1)
        h, _ = self.rnn(inp)
        return masked_log_softmax(self.head(h), avail), h


def n_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def logp_numpy(logp: torch.Tensor, seq: EncodedSeq) -> np.ndarray:
    out = logp.detach().double().numpy()[: seq.T]
    out = np.where(seq.avail, out, -np.inf)
    return out
