"""``ref_gru_p2`` — the Phase II numpy GRU reference (``hpe.evaluation.human.gru``) refitted on the Phase III dev
participants, so the Phase II reference sits in the pool on identical participants (thesis §4 item 1).

Input, rebuilt from the v3 ``EncodedSeq`` exactly as Phase II built it (``models2.gru_input``):
``[context features (12), one-hot previous action (K), previous chosen reward (1), one-hot previous correct
action (K), previous-feedback flag (1)]`` with K = the family's number of actions. "Previous correct" is
defined only after full feedback (every option's outcome observed) as the argmax of the observed outcomes
(Phase II: ``int(np.argmax(rewards_all))``; ties → first option); after bandit or no feedback it is all-zero.
Fitting: Phase II procedure (``models2.fit_gru``): an 80/20 split of the fitting participants drawn with
``numpy.random.default_rng(seed)``, Adam + truncated BPTT (``window``), early stopping on the 20 %. Phase II grid:
hidden {8, 12} × lr {0.003, 0.01}, max_epochs 15, patience 4, window 64 (horizon/described: fewer epochs and
``max_steps_per_epoch`` 60,000). Population-only (``adaptive = False``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hpe.evaluation.human import gru
from hpe.evaluation.human.models2 import gru_input_dim
from hpe.population.core.encoding import EncodedSeq, F, Layout
from hpe.population.core.interface import ModelSpec, register_model


def phase2_inputs(seq: EncodedSeq, k: int) -> np.ndarray:
    """Phase II GRU inputs ``[T, F + 2K + 2]`` from a v3 sequence (causal: row t uses only x[t])."""
    L = Layout(seq.k_max)
    T = seq.T
    out = np.zeros((T, gru_input_dim(k)))
    for t in range(T):
        x = seq.x[t].astype(np.float64)
        out[t, :F] = x[L.feat]
        if t > 0:
            pa = x[L.prev_action][:k]
            out[t, F : F + k] = pa
            fb = x[L.prev_feedback] > 0.5
            out[t, F + k] = x[L.prev_reward] if fb else 0.0
            obs = x[L.prev_observed][:k] > 0.5
            if fb and obs.all():
                out[t, F + k + 1 + int(np.argmax(x[L.prev_outcomes][:k]))] = 1.0
            out[t, F + 2 * k + 1] = 1.0 if fb else 0.0
    return out


class RefGRUP2:
    name = "ref_gru_p2"
    adaptive = False

    def __init__(self, config: dict[str, Any], k_max: int, d_in: int) -> None:
        self.config = dict(config)
        self.k_max = k_max
        self.k = int(config.get("k", 0))
        self.P: gru.GRUParams | None = None

    def fit(self, train: list[EncodedSeq], *, seed: int, threads: int, budget_s: float) -> dict[str, Any]:
        k = self.k or int(train[0].n_actions)
        self.k = k
        seqs = [(phase2_inputs(s, k), s.actions.copy(), s.scored.copy()) for s in train]
        rng = np.random.default_rng(seed)  # Phase II split procedure (models2.fit_gru)
        order = rng.permutation(len(seqs))
        n_val = max(1, len(seqs) // 5)
        val = [seqs[i] for i in order[:n_val]]
        fit = [seqs[i] for i in order[n_val:]] or val
        mspe = self.config.get("max_steps_per_epoch")
        self.P, log = gru.train_gru(
            fit,
            val,
            n_in=gru_input_dim(k),
            n_hidden=int(self.config.get("hidden", 8)),
            n_out=k,
            lr=float(self.config.get("lr", 0.01)),
            max_epochs=int(self.config.get("max_epochs", 15)),
            patience=int(self.config.get("patience", 4)),
            window=int(self.config.get("window", 64)),
            seed=seed,
            max_steps_per_epoch=int(mspe) if mspe else None,
        )
        return {
            "epochs": len(log["epochs"]),
            "best_val_nll": log["best_val_nll"],
            "val_history": [round(e["val_nll"], 5) for e in log["epochs"]],
            "n_fit": len(fit),
            "n_val": len(val),
        }

    def predict_logprobs(
        self, seq: EncodedSeq, *, mode: str = "adapt", donor: EncodedSeq | None = None
    ) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("ref_gru_p2: predict before fit")
        xs = phase2_inputs(seq, self.k)
        h = np.zeros(self.P.bz.shape[0])
        out = np.full((seq.T, seq.k_max), -np.inf)
        for t in range(seq.T):
            h, p, _ = gru.step(self.P, xs[t], h)
            with np.errstate(divide="ignore"):
                out[t, : self.k] = np.log(np.maximum(p, 1e-300))
        return np.where(seq.avail, out, -np.inf)

    def param_count(self) -> int:
        return 0 if self.P is None else self.P.count()

    def describe(self) -> dict[str, Any]:
        return {"population_params": self.param_count(), "per_person_params": 0, "config": self.config}


def _build(config: dict[str, Any], k: int, d: int) -> RefGRUP2:
    return RefGRUP2(config, k, d)


register_model(
    ModelSpec(
        "ref_gru_p2",
        _build,
        "baselines",
        "comparator",
        {"hidden": [8, 12], "lr": [0.003, 0.01]},
        note="Phase II numpy GRU refitted on Phase III dev with its Phase II grid and input",
    )
)
