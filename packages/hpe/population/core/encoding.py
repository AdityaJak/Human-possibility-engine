"""Encoding v3: one input layout for every sequence model (docs/phase3/PHASE3_RESEARCH_THESIS.md §4).

Step t input (K = k_max, F = 12 context features), all known before the choice at t is made:

    [0, F)                    current context features
    [F, F+K)                  available-action mask
    [F+K, F+2K)               previous action (one-hot)
    [F+2K, F+3K)              previous outcome of every option (0 where unobserved), outcomes in [0, 1]
    [F+3K, F+4K)              previous observed-outcome mask
    F+4K                      previous chosen outcome
    F+4K+1                    previous feedback present
    [F+4K+2, 2F+4K+2)         previous context features
    2F+4K+2                   block start
    2F+4K+3                   instructed trial (the action is given by the experimenter; not scored)
    [2F+4K+4, 2F+5K+4)        instructed action (one-hot; zeros on free trials)

Targets: the participant's action at t. Scored: free (non-instructed) trials only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, GenericTrial

F = N_FEATURES


def input_dim(k_max: int) -> int:
    return 2 * F + 5 * k_max + 4


@dataclass(frozen=True, slots=True)
class Layout:
    k: int

    @property
    def feat(self) -> slice:
        return slice(0, F)

    @property
    def avail(self) -> slice:
        return slice(F, F + self.k)

    @property
    def prev_action(self) -> slice:
        return slice(F + self.k, F + 2 * self.k)

    @property
    def prev_outcomes(self) -> slice:
        return slice(F + 2 * self.k, F + 3 * self.k)

    @property
    def prev_observed(self) -> slice:
        return slice(F + 3 * self.k, F + 4 * self.k)

    @property
    def prev_reward(self) -> int:
        return F + 4 * self.k

    @property
    def prev_feedback(self) -> int:
        return F + 4 * self.k + 1

    @property
    def prev_feat(self) -> slice:
        return slice(F + 4 * self.k + 2, 2 * F + 4 * self.k + 2)

    @property
    def block_start(self) -> int:
        return 2 * F + 4 * self.k + 2

    @property
    def instructed(self) -> int:
        return 2 * F + 4 * self.k + 3

    @property
    def instructed_action(self) -> slice:
        return slice(2 * F + 4 * self.k + 4, 2 * F + 5 * self.k + 4)


@dataclass(slots=True)
class EncodedSeq:
    key: str  # "<experiment>#<participant>"
    family: str
    k_max: int
    n_actions: int
    x: np.ndarray  # [T, D] float32
    actions: np.ndarray  # [T] int64
    avail: np.ndarray  # [T, K] bool
    scored: np.ndarray  # [T] bool
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_scored(self) -> int:
        return int(self.scored.sum())


def _outcomes(tr: GenericTrial, k: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(k)
    obs = np.zeros(k)
    if tr.mode == "bandit":
        out[tr.choice] = float(tr.reward or 0.0)
        obs[tr.choice] = 1.0
    elif tr.mode == "full":
        if tr.rewards_all is not None:
            ra = np.asarray(tr.rewards_all, dtype=np.float64)
            out[: ra.shape[0]] = ra
            obs[: ra.shape[0]] = 1.0
        else:
            out[: tr.n_actions] = np.eye(tr.n_actions)[tr.correct or 0]
            obs[: tr.n_actions] = 1.0
    return out, obs


def encode(p: Participant, family: str, k_max: int) -> EncodedSeq:
    """Encode one parsed participant (GenericTrial records) into the v3 layout."""
    L = Layout(k_max)
    T = len(p.trials)
    D = input_dim(k_max)
    x = np.zeros((T, D), dtype=np.float32)
    actions = np.zeros(T, dtype=np.int64)
    avail = np.zeros((T, k_max), dtype=bool)
    scored = np.zeros(T, dtype=bool)
    prev: GenericTrial | None = None
    prev_block: int | None = None
    for t, tr in enumerate(p.trials):
        if not isinstance(tr, GenericTrial):
            raise TypeError(f"{p.key}: encoding v3 needs GenericTrial records, got {type(tr).__name__}")
        if tr.n_actions > k_max:
            raise ValueError(f"{p.key}: {tr.n_actions} actions > k_max {k_max}")
        x[t, L.feat] = np.asarray(tr.features, dtype=np.float32)
        avail[t, : tr.n_actions] = True
        x[t, L.avail] = avail[t]
        if prev is not None:
            x[t, F + k_max + prev.choice] = 1.0
            out, obs = _outcomes(prev, k_max)
            x[t, L.prev_outcomes] = out
            x[t, L.prev_observed] = obs
            x[t, L.prev_reward] = float(prev.reward or 0.0) if prev.mode != "none" else 0.0
            x[t, L.prev_feedback] = 1.0 if prev.mode != "none" else 0.0
            x[t, L.prev_feat] = np.asarray(prev.features, dtype=np.float32)
        x[t, L.block_start] = 1.0 if tr.block != prev_block else 0.0
        if tr.forced:
            x[t, L.instructed] = 1.0
            x[t, L.instructed_action.start + tr.choice] = 1.0
        actions[t] = tr.choice
        scored[t] = not tr.forced
        prev, prev_block = tr, tr.block
    return EncodedSeq(p.key, family, k_max, p.n_actions, x, actions, avail, scored)


def truncate(seq: EncodedSeq, n: int, *, key: str | None = None) -> EncodedSeq:
    """Copy holding only the first ``n`` trials (the only data an adapter may see at row ``n``)."""
    n = max(0, min(int(n), seq.T))
    return EncodedSeq(
        key if key is not None else seq.key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        seq.x[:n].copy(),
        seq.actions[:n].copy(),
        seq.avail[:n].copy(),
        seq.scored[:n].copy(),
    )


def rekey(seq: EncodedSeq, key: str) -> EncodedSeq:
    """Copy with an opaque key: models never see participant or experiment identifiers at prediction."""
    return EncodedSeq(
        key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        seq.x.copy(),
        seq.actions.copy(),
        seq.avail.copy(),
        seq.scored.copy(),
    )
