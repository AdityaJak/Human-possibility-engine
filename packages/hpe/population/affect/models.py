"""Gate F models for the GBE risky-decision-and-happiness data (docs/phase3/AFFECTIVE_SOCIAL_VALIDATION.md).

All are population GRUs over encoding v3 (``nets.PopRNN``); they differ only in what they get about the
person's *reported* happiness:

* ``affect_gru_base``   — no affect input (comparator: the population GRU the mechanism must improve on).
* ``affect_gru_obs``    — the mechanism: the latest rating made before each choice, its freshness and its
  age (the :data:`gbe.N_EXTRA` appended columns).
* ``affect_gru_shuf``   — matched-complexity control: identical network and inputs, except that each rating
  is replaced by one of the person's *earlier* ratings drawn at random (:func:`past_shuffle`). Same
  parameters, same timing columns, same person-level rating distribution so far; the alignment between
  a rating and the outcomes that preceded it is destroyed. Causal by construction.
* ``affect_gru_aux`` / ``affect_gru_aux_shuf`` — latent-affect variant: the GRU sees no rating at all but
  is trained with an auxiliary head that predicts each observed rating from the hidden state (a learned
  "mood" readout, cf. Eldar & Niv 2015; Rutledge et al. 2014); the control trains the same head on
  ratings permuted within participant. At prediction time neither needs ratings.

These models measure whether *observed* affect improves choice prediction; they make no claim that any
model has or experiences affect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from hpe.core.rng import stream
from hpe.population.affect import gbe
from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq, Layout
from hpe.population.core.interface import ModelSpec, register_model

EXTRA_MODES = ("none", "observed", "past_shuffle")
SHUFFLE_SEED = 20261108


def past_shuffle(extra: np.ndarray, key: str, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Causal within-participant resampling of the rating column (column 0).

    Rating events are rows whose freshness column (1) is set. At the first event the true value is kept
    (nothing earlier exists); at event ``k >= 1`` the value of a uniformly drawn *earlier* event ``j < k``
    is fed instead and carried forward until the next event. Row ``t`` depends only on rows ``<= t``.
    """
    out = np.array(extra, dtype=np.float32, copy=True)
    rng = stream(seed, "affect_past_shuffle", key)
    past: list[float] = []
    cur = 0.0
    for t in range(out.shape[0]):
        if extra[t, 1] > 0.5:
            true = float(extra[t, 0])
            cur = past[int(rng.integers(0, len(past)))] if past else true
            past.append(true)
        out[t, 0] = cur
    return out


def permute_ratings(ratings: np.ndarray, key: str, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Permute the observed (finite) ratings across trials within one participant (auxiliary targets only)."""
    out = np.array(ratings, dtype=np.float64, copy=True)
    idx = np.flatnonzero(np.isfinite(out))
    out[idx] = out[idx][stream(seed, "affect_target_perm", key).permutation(idx.size)]
    return out


def _view(seq: EncodedSeq, d_base: int, extra_mode: str, seed: int) -> EncodedSeq:
    base = seq.x[:, :d_base]
    if extra_mode == "none":
        x = base
    else:
        extra = seq.x[:, d_base : d_base + gbe.N_EXTRA]
        if extra.shape[1] != gbe.N_EXTRA:
            raise ValueError(f"{seq.key}: expected {gbe.N_EXTRA} affect columns after the v3 layout")
        if extra_mode == "past_shuffle":
            extra = past_shuffle(extra, seq.key, seed)
        x = np.concatenate([base, extra], axis=1)
    return EncodedSeq(
        seq.key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        np.ascontiguousarray(x, dtype=np.float32),
        seq.actions,
        seq.avail,
        seq.scored,
    )


def _train_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "lr": float(config.get("lr", 0.01)),
        "max_epochs": int(config.get("max_epochs", 150)),
        "patience": int(config.get("patience", 10)),
        "batch_size": int(config.get("batch_size", 8)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }


class AffectGRU:
    """Population GRU with optional affect columns (``extra_mode`` in :data:`EXTRA_MODES`)."""

    adaptive = False

    def __init__(self, name: str, extra_mode: str, config: dict[str, Any], k_max: int, d_in: int) -> None:
        if extra_mode not in EXTRA_MODES:
            raise ValueError(f"unknown extra_mode {extra_mode!r}")
        self.name, self.extra_mode = name, extra_mode
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.n_extra = 0 if extra_mode == "none" else gbe.N_EXTRA
        self.shuffle_seed = int(config.get("shuffle_seed", SHUFFLE_SEED))
        self.net = self._build()

    def _build(self) -> nets.PopRNN:
        return nets.PopRNN(
            self.d_in,
            int(self.config.get("hidden", 32)),
            self.k_max,
            extra_in=self.n_extra,
            dropout=float(self.config.get("dropout", 0.0)),
        )

    def view(self, seq: EncodedSeq) -> EncodedSeq:
        return _view(seq, self.d_in, self.extra_mode, self.shuffle_seed)

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        nets.setup_torch(seed, threads, self.name, repr(sorted(self.config.items())))
        self.net = self._build()
        views = [self.view(s) for s in train]

        def loss(m: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            logp, _ = m(b.x, b.avail)
            return nets.nll_sum(logp, b)

        return nets.train(
            self.net, loss, views, seed=seed, key=self.name, budget_s=budget_s, **_train_kwargs(self.config)
        )

    @torch.no_grad()
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        v = self.view(seq)
        b = nets.Batch([v])
        logp, _ = self.net(b.x, b.avail)
        return nets.logp_numpy(logp[0], seq)

    def param_count(self) -> int:
        return nets.n_params(self.net)

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": 0,
            "extra_mode": self.extra_mode,
            "config": self.config,
        }


class _AuxNet(nn.Module):
    def __init__(self, d_in: int, hidden: int, k: int) -> None:
        super().__init__()
        self.core = nets.PopRNN(d_in, hidden, k)
        self.mood = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, avail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logp, h = self.core(x, avail)
        return logp, torch.sigmoid(self.mood(h)).squeeze(-1)


class AffectAuxGRU:
    """GRU without affect inputs, trained to also predict each observed rating (latent-mood readout).

    The rating made after trial ``t``'s outcome is the target of row ``t + 1`` (the first row whose input
    contains that outcome); a play's pre-play rating is the target of the play's first row; the rating after
    a play's last trial is not used. Ratings never enter as inputs, so prediction on new people needs none.
    """

    adaptive = False

    def __init__(self, name: str, permuted: bool, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.name, self.permuted = name, permuted
        self.config = dict(config)
        self.k_max, self.d_in = k_max, d_in
        self.aux_weight = float(config.get("aux_weight", 1.0))
        self.shuffle_seed = int(config.get("shuffle_seed", SHUFFLE_SEED))
        self.net = _AuxNet(d_in, int(config.get("hidden", 32)), k_max)

    @staticmethod
    def aligned_targets(seq: EncodedSeq, ratings: np.ndarray) -> np.ndarray:
        """Target for row ``r``: the rating whose information is first available at row ``r``."""
        tgt = np.full(seq.T, np.nan)
        block = seq.x[:, Layout(seq.k_max).block_start]
        for t in range(seq.T):
            if not np.isfinite(ratings[t]):
                continue
            if block[t] > 0.5:  # pre-play rating, made before this row's choice
                tgt[t] = ratings[t]
            elif t + 1 < seq.T and block[t + 1] < 0.5:
                tgt[t + 1] = ratings[t]
        return tgt / 100.0

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        nets.setup_torch(seed, threads, self.name, repr(sorted(self.config.items())))
        self.net = _AuxNet(self.d_in, int(self.config.get("hidden", 32)), self.k_max)
        views = [_view(s, self.d_in, "none", self.shuffle_seed) for s in train]
        targets: dict[str, np.ndarray] = {}
        for s in train:
            r = np.asarray(s.meta["ratings"], dtype=np.float64)
            if self.permuted:
                r = permute_ratings(r, s.key, self.shuffle_seed)
            targets[s.key] = self.aligned_targets(s, r)
        w = self.aux_weight

        def batch_targets(b: nets.Batch) -> tuple[torch.Tensor, torch.Tensor]:
            y = np.zeros((b.B, b.T), dtype=np.float32)
            m = np.zeros((b.B, b.T), dtype=bool)
            for i, k in enumerate(b.keys):
                tg = targets[k]
                ok = np.isfinite(tg)
                y[i, : tg.size][ok] = tg[ok]
                m[i, : tg.size] = ok
            return torch.from_numpy(y), torch.from_numpy(m)

        def loss(mdl: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            logp, mood = mdl(b.x, b.avail)
            s, n = nets.nll_sum(logp, b)
            y, m = batch_targets(b)
            se = torch.where(m, (mood - y) ** 2, torch.zeros_like(mood)).sum()
            # scale the squared error so that a rating SD of ~0.2 contributes O(1) nats per rated step
            return s + w * se / 0.04 * 0.5, n

        def val_loss(mdl: nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
            logp, _ = mdl(b.x, b.avail)
            return nets.nll_sum(logp, b)

        return nets.train(
            self.net,
            loss,
            views,
            seed=seed,
            key=self.name,
            budget_s=budget_s,
            val_loss_fn=val_loss,
            **_train_kwargs(self.config),
        )

    @torch.no_grad()
    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        v = _view(seq, self.d_in, "none", self.shuffle_seed)
        b = nets.Batch([v])
        logp, _ = self.net(b.x, b.avail)
        return nets.logp_numpy(logp[0], seq)

    @torch.no_grad()
    def predict_mood(self, seq: EncodedSeq) -> np.ndarray:
        v = _view(seq, self.d_in, "none", self.shuffle_seed)
        b = nets.Batch([v])
        _, mood = self.net(b.x, b.avail)
        return mood[0].double().numpy()[: seq.T]

    def param_count(self) -> int:
        return nets.n_params(self.net)

    def describe(self) -> dict[str, Any]:
        return {
            "population_params": self.param_count(),
            "per_person_params": 0,
            "aux_targets": "permuted within participant" if self.permuted else "observed ratings",
            "config": self.config,
        }


GRID: dict[str, list[Any]] = {"hidden": [16, 32], "lr": [0.01]}


def _factory(name: str, mode: str):  # type: ignore[no-untyped-def]
    def f(config: dict[str, Any], k: int, d: int) -> AffectGRU:
        return AffectGRU(name, mode, config, k, d)

    return f


def _aux_factory(name: str, permuted: bool):  # type: ignore[no-untyped-def]
    def f(config: dict[str, Any], k: int, d: int) -> AffectAuxGRU:
        return AffectAuxGRU(name, permuted, config, k, d)

    return f


register_model(
    ModelSpec(
        "affect_gru_base",
        _factory("affect_gru_base", "none"),
        "affect",
        "comparator",
        GRID,
        note="population GRU, no affect input (GBE)",
    )
)
register_model(
    ModelSpec(
        "affect_gru_obs",
        _factory("affect_gru_obs", "observed"),
        "affect",
        "candidate",
        GRID,
        note="Gate F mechanism: latest observed happiness rating before each choice",
    )
)
register_model(
    ModelSpec(
        "affect_gru_shuf",
        _factory("affect_gru_shuf", "past_shuffle"),
        "affect",
        "comparator",
        GRID,
        note="Gate F matched-complexity control: same inputs, ratings resampled from the person's earlier ratings",
    )
)
register_model(
    ModelSpec(
        "affect_gru_aux",
        _aux_factory("affect_gru_aux", False),
        "affect",
        "candidate",
        GRID,
        note="latent-mood readout trained on observed ratings; no rating input",
    )
)
register_model(
    ModelSpec(
        "affect_gru_aux_shuf",
        _aux_factory("affect_gru_aux_shuf", True),
        "affect",
        "comparator",
        GRID,
        note="control for affect_gru_aux: auxiliary targets permuted within participant",
    )
)
