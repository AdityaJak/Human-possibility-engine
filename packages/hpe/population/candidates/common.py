"""Shared helpers for the design-team candidates (F1, F2, F3).

Equal treatment: every population network here is trained with ``hpe.population.core.nets.train`` and
the *same* trainer defaults as the core ``pop_gru`` comparator (batch 8, ≤ 150 epochs, patience 10,
plateau lr halving), read from the same config keys (``lr``, ``hidden``, ``dropout``, ``batch_size``,
``max_epochs``, ``patience``, ``weight_decay``).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from hpe.core.rng import derive_seed
from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq

TRAINER_KEYS = ("lr", "max_epochs", "patience", "batch_size", "weight_decay")


def trainer_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Trainer settings with the core defaults (identical to ``core.reference.PopRNNPredictor``)."""
    return {
        "lr": float(cfg.get("lr", 0.01)),
        "max_epochs": int(cfg.get("max_epochs", 150)),
        "patience": int(cfg.get("patience", 10)),
        "batch_size": int(cfg.get("batch_size", 8)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
    }


def seq_digest(seqs: Sequence[EncodedSeq]) -> str:
    """Content hash of a training set (keys, inputs, actions, masks)."""
    h = hashlib.sha256()
    for s in seqs:
        h.update(s.key.encode())
        h.update(s.x.tobytes())
        h.update(s.actions.tobytes())
        h.update(s.scored.tobytes())
    return h.hexdigest()[:20]


# --------------------------------------------------------------------------- shared population backbone
# F3 variants (and the F3 ablations) share one backbone per (training data, backbone config) so that the
# arbitration ablations are exactly paired on the same population GRU. The cache lives for one process;
# the backbone's CPU time is recorded in every fit log that uses it (cached or not), so cost reports
# attribute the full backbone cost to every model.
_BACKBONES: dict[tuple[Any, ...], tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}


def backbone_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "hidden": int(cfg.get("hidden", 32)),
        "dropout": float(cfg.get("dropout", 0.0)),
        "backbone_seed": int(cfg.get("backbone_seed", 1)),
        **trainer_kwargs(cfg),
    }


def train_backbone(
    seqs: list[EncodedSeq], cfg: dict[str, Any], k_max: int, d_in: int, *, threads: int, budget_s: float
) -> tuple[nets.PopRNN, dict[str, Any]]:
    """Train (or fetch from the per-process cache) a ``PopRNN`` GRU exactly as ``pop_gru`` trains."""
    bc = backbone_config(cfg)
    digest = seq_digest(seqs)
    key = (digest, k_max, d_in, tuple(sorted(bc.items())))
    seed = derive_seed(bc["backbone_seed"], "candidates_backbone", digest, repr(sorted(bc.items()))) % (
        2**31 - 1
    )
    nets.setup_torch(seed, threads, "candidates_backbone")
    net = nets.PopRNN(d_in, bc["hidden"], k_max, cell="gru", dropout=bc["dropout"])
    if key in _BACKBONES:
        state, log = _BACKBONES[key]
        net.load_state_dict(state)
        net.eval()
        return net, {**log, "cached": True}
    t0 = time.process_time()

    def loss(m: torch.nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
        logp, _ = m(b.x, b.avail)
        return nets.nll_sum(logp, b)

    log = nets.train(
        net, loss, seqs, seed=seed, key="candidates_backbone", budget_s=budget_s, **trainer_kwargs(cfg)
    )
    log = {**log, "backbone_cpu_s": round(time.process_time() - t0, 2), "backbone_seed": seed}
    _BACKBONES[key] = ({k: v.detach().clone() for k, v in net.state_dict().items()}, log)
    return net, {**log, "cached": False}


def backbone_val_split(
    seqs: list[EncodedSeq], cfg: dict[str, Any]
) -> tuple[list[EncodedSeq], list[EncodedSeq]]:
    """The backbone's own early-stopping split (same seed and key as ``train_backbone``)."""
    bc = backbone_config(cfg)
    digest = seq_digest(seqs)
    seed = derive_seed(bc["backbone_seed"], "candidates_backbone", digest, repr(sorted(bc.items()))) % (
        2**31 - 1
    )
    return nets.split_train_val(seqs, seed, "candidates_backbone")


def clear_backbone_cache() -> None:
    _BACKBONES.clear()


# --------------------------------------------------------------------------- small numerics
def log_softmax_np(logits: np.ndarray, avail: np.ndarray) -> np.ndarray:
    z = np.where(avail, logits, -np.inf)
    m = np.max(z, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))
    return np.where(avail, z - lse, -np.inf)


def finalize_logp(p: np.ndarray, avail: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Probabilities [T, K] -> log-probabilities with ``-inf`` on unavailable actions, renormalised."""
    p = np.where(avail, np.maximum(p, floor), 0.0)
    p = p / p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.where(avail, np.log(p), -np.inf)
