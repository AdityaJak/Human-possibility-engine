"""Shared population GRU backbone for the conventional adapters (``ft_head``, ``ft_full``, ``bayes_head``,
``exemplar_mix``, ``reptile_gru``).

The backbone is trained inside ``fit`` **exactly like the core ``pop_gru``** (same ``nets.setup_torch`` key,
``nets.PopRNN`` module, ``nets.train`` call and defaults). Its configuration and seed are fixed hyper-parameters
of the adapter (``backbone`` and ``backbone_seed`` in the model config): the YAMLs set them to the tune-selected
``pop_gru`` configuration and the seed the runner gave that configuration, so an adapter's ``frozen`` variant is
bit-identical to the selected ``pop_gru`` (verified in ``tests/population/baselines``). Every adapter of a family
then shares one backbone; it is trained once per process and cached in memory (and on disk when
``HPE_P3_BACKBONE_CACHE`` names a directory). Its CPU cost is reported in each adapter's fit log
(``backbone_cpu_s``, ``backbone_cached``).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hpe.population.core import nets
from hpe.population.core.encoding import EncodedSeq

BACKBONE_DEFAULTS: dict[str, Any] = {
    "hidden": 32,
    "lr": 0.01,
    "dropout": 0.0,
    "max_epochs": 150,
    "patience": 10,
    "batch_size": 8,
}
CACHE_ENV = "HPE_P3_BACKBONE_CACHE"
_MEM: dict[str, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}


def data_digest(seqs: list[EncodedSeq]) -> str:
    h = hashlib.sha256()
    for s in seqs:
        h.update(f"{s.key}:{s.T}:{s.k_max}".encode())
        h.update(np.ascontiguousarray(s.x).tobytes())
        h.update(np.ascontiguousarray(s.actions).tobytes())
    return h.hexdigest()[:24]


def backbone_config(config: dict[str, Any]) -> dict[str, Any]:
    """The backbone's config dict exactly as ``pop_gru`` would have received it from the runner."""
    bb = config.get("backbone")
    return dict(BACKBONE_DEFAULTS) if bb is None else dict(bb)


def make_net(d_in: int, k_max: int, bb: dict[str, Any]) -> nets.PopRNN:
    return nets.PopRNN(
        d_in,
        int(bb.get("hidden", 32)),
        k_max,
        cell="gru",
        layers=int(bb.get("layers", 1)),
        dropout=float(bb.get("dropout", 0.0)),
    )


def _loss(m: torch.nn.Module, b: nets.Batch) -> tuple[torch.Tensor, int]:
    logp, _ = m(b.x, b.avail)
    return nets.nll_sum(logp, b)


def train_backbone(
    train: list[EncodedSeq],
    bb: dict[str, Any],
    seed: int,
    *,
    d_in: int,
    k_max: int,
    threads: int,
    budget_s: float,
) -> tuple[nets.PopRNN, dict[str, Any]]:
    """Train (or fetch from cache) the population GRU exactly as ``core.reference.PopRNNPredictor.fit`` does."""
    key = hashlib.sha256(
        repr((data_digest(train), sorted(bb.items()), int(seed), int(threads), d_in, k_max)).encode()
    ).hexdigest()[:24]
    net = make_net(d_in, k_max, bb)
    cache_dir = os.environ.get(CACHE_ENV)
    disk = Path(cache_dir) / f"backbone_{key}.pt" if cache_dir else None
    if key in _MEM:
        state, log = _MEM[key]
        net.load_state_dict(state)
        net.eval()
        nets.setup_torch(seed, threads, "pop_gru", repr(sorted(bb.items())))
        return net, {**log, "backbone_cached": "memory"}
    if disk is not None and disk.exists():
        blob = torch.load(disk, weights_only=False)
        net.load_state_dict(blob["state"])
        net.eval()
        _MEM[key] = (blob["state"], blob["log"])
        nets.setup_torch(seed, threads, "pop_gru", repr(sorted(bb.items())))
        return net, {**blob["log"], "backbone_cached": "disk"}
    t0 = time.process_time()
    nets.setup_torch(seed, threads, "pop_gru", repr(sorted(bb.items())))
    net = make_net(d_in, k_max, bb)
    log = nets.train(
        net,
        _loss,
        train,
        seed=seed,
        key="pop_gru",
        lr=float(bb.get("lr", 0.01)),
        max_epochs=int(bb.get("max_epochs", 150)),
        patience=int(bb.get("patience", 10)),
        batch_size=int(bb.get("batch_size", 8)),
        budget_s=budget_s,
        weight_decay=float(bb.get("weight_decay", 0.0)),
    )
    log = {"backbone_train": log, "backbone_cpu_s": round(time.process_time() - t0, 2), "backbone_key": key}
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    _MEM[key] = (state, log)
    if disk is not None:
        disk.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": state, "log": log}, disk)
    return net, {**log, "backbone_cached": "no"}


@torch.no_grad()
def hidden_states(net: nets.PopRNN, seq: EncodedSeq) -> tuple[np.ndarray, np.ndarray]:
    """Backbone hidden states ``[T, H]`` and population log-probabilities ``[T, K]`` (float64)."""
    b = nets.Batch([seq])
    logp, h = net(b.x, b.avail)
    return h[0].double().numpy(), nets.logp_numpy(logp[0], seq)


def masked_logsoftmax_np(logits: np.ndarray, avail: np.ndarray) -> np.ndarray:
    z = np.where(avail, logits, -np.inf)
    m = z.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))
    return np.where(avail, z - lse, -np.inf)


def update_points(T: int, m: int) -> list[int]:
    """Steps at which adapted parameters are refitted on all steps before them (every ``m`` steps)."""
    return list(range(m, T, m)) if m > 0 else []
