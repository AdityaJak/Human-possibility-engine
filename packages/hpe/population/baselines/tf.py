"""``pop_tf`` — a small causal transformer over encoding v3 (population-only comparator; thesis §4 item 3).

Architecture: linear input projection to ``d_model``; ``layers`` pre-LayerNorm blocks of multi-head
self-attention (``heads``) + a 2·``d_model`` GELU feed-forward, each with a residual; a final LayerNorm and a
linear head to ``K`` logits. Attention is restricted to a **sliding causal window**: step t attends to steps
``t − W + 1 … t`` only (a band mask; ``-inf`` elsewhere), with a **learned relative-position bias** per head for
offsets ``0 … W − 1`` (the window's positions are learned; there are no absolute positions, so behaviour does
not depend on where the window sits in the sequence). Row t therefore depends on ``x[:t+1]`` only. Trained on
dev with the core trainer (``nets.train``; batch 8, ≤ 150 epochs, plateau lr halving, early stopping).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq
from hpe.population.core.interface import ModelSpec, register_model


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.h, self.dk, self.window = heads, d // heads, window
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.rel = nn.Parameter(torch.zeros(heads, window))
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        bias = self.rel[:, offset.clamp(0, self.window - 1)]  # [h, T, T]
        att = (att + bias.unsqueeze(0)).masked_fill(~band, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


class CausalTF(nn.Module):
    def __init__(
        self, d_in: int, k: int, d_model: int, layers: int, heads: int, window: int, dropout: float
    ) -> None:
        super().__init__()
        self.window = window
        self.inp = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, heads, window, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, k)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, avail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[1]
        idx = torch.arange(T)
        offset = idx[:, None] - idx[None, :]  # t - s
        band = (offset >= 0) & (offset < self.window)
        h = self.drop(self.inp(x))
        for blk in self.blocks:
            h = blk(h, offset, band)
        h = self.ln(h)
        return nets.masked_log_softmax(self.head(h), avail), h


class PopTF:
    name = "pop_tf"
    adaptive = False

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.net = self._make()

    def _make(self) -> CausalTF:
        c = self.config
        return CausalTF(
            self.d_in,
            self.k_max,
            int(c.get("d_model", 32)),
            int(c.get("layers", 2)),
            int(c.get("heads", 4)),
            int(c.get("window", 32)),
            float(c.get("dropout", 0.0)),
        )

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        nets.setup_torch(seed, threads, self.name, repr(sorted(self.config.items())))
        self.net = self._make()

        def loss(m: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            logp, _ = m(b.x, b.avail)
            return nets.nll_sum(logp, b)

        return nets.train(
            self.net,
            loss,
            train,
            seed=seed,
            key=self.name,
            lr=float(self.config.get("lr", 0.003)),
            max_epochs=int(self.config.get("max_epochs", 150)),
            patience=int(self.config.get("patience", 10)),
            batch_size=int(self.config.get("batch_size", 8)),
            budget_s=budget_s,
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    @torch.no_grad()
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        self.net.eval()
        b = nets.Batch([seq])
        logp, _ = self.net(b.x, b.avail)
        return nets.logp_numpy(logp[0], seq)

    def param_count(self) -> int:
        return nets.n_params(self.net)

    def describe(self) -> dict[str, Any]:
        return {"population_params": self.param_count(), "per_person_params": 0, "config": self.config}


def _build(config: dict[str, Any], k: int, d: int) -> PopTF:
    return PopTF(config, k, d)


register_model(
    ModelSpec(
        "pop_tf",
        _build,
        "baselines",
        "comparator",
        {"window": [32, 128], "lr": [0.001, 0.003], "dropout": [0.0, 0.1]},
        note="causal sliding-window transformer, d_model 32, 2 layers, 4 heads, learned relative positions",
    )
)
