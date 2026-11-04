"""Synthetic verification populations for the candidates (no human data).

``kind="bandit"``: each simulated person plays a K-armed restless bandit (Gaussian random-walk means,
outcomes in [0, 1]) with a softmax Q-learner: logits = β Q + κ 1[prev] + b·e_0. ``hetero=True`` draws
(β, κ, b, α) per person from a wide population distribution (individuals genuinely differ);
``hetero=False`` gives everyone the population-mean parameters (nobody differs).

``kind="cue"``: on every trial one of ``N_CUES`` cues is shown (context features); the person chooses
their personally preferred action for that cue with probability 0.85 (no outcomes). ``hetero=True``:
each person's cue→action table is drawn at random; ``hetero=False``: everyone shares one table.
A population model must learn each person's table in context; explicit individual adaptation (a latent,
an episodic memory, a person-level posterior) has something to add only in the heterogeneous case.
"""

from __future__ import annotations

import numpy as np

from hpe.core.rng import stream
from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.psych101_v2 import N_FEATURES, GenericTrial
from hpe.population.core.encoding import EncodedSeq, encode

POP_MEAN = {"alpha": 0.4, "beta": 6.0, "kappa": 0.5, "bias": 0.0}


def person_params(rng: np.random.Generator, hetero: bool) -> dict[str, float]:
    if not hetero:
        return dict(POP_MEAN)
    return {
        "alpha": float(rng.uniform(0.1, 0.9)),
        "beta": float(np.exp(rng.normal(np.log(6.0), 0.6))),
        "kappa": float(rng.normal(0.5, 1.5)),
        "bias": float(rng.normal(0.0, 1.5)),
    }


def simulate_person(rng: np.random.Generator, prm: dict[str, float], T: int, k: int, idx: int) -> Participant:
    p = Participant("toy/exp.csv", str(idx), "toy", k)
    means = rng.uniform(0.2, 0.8, size=k)
    q = np.full(k, 0.5)
    prev = -1
    zeros = (0.0,) * N_FEATURES
    for _t in range(T):
        logits = prm["beta"] * q
        if prev >= 0:
            logits[prev] += prm["kappa"]
        logits[0] += prm["bias"]
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        c = int(rng.choice(k, p=pr))
        r = float(np.clip(means[c] + rng.normal(0.0, 0.1), 0.0, 1.0))
        q[c] += prm["alpha"] * (r - q[c])
        means = np.clip(means + rng.normal(0.0, 0.04, size=k), 0.05, 0.95)
        p.trials.append(GenericTrial(c, k, zeros, "bandit", r, block=0, phase="main"))
        prev = c
    return p


N_CUES = 4
P_PREF = 0.85


def simulate_cue_person(rng: np.random.Generator, table: np.ndarray, T: int, k: int, idx: int) -> Participant:
    p = Participant("toy/cue.csv", str(idx), "toy", k)
    for _t in range(T):
        cue = int(rng.integers(0, N_CUES))
        feats = tuple(1.0 if j == cue else 0.0 for j in range(N_CUES)) + (0.0,) * (N_FEATURES - N_CUES)
        pref = int(table[cue])
        if rng.uniform() < P_PREF:
            c = pref
        else:
            c = int(rng.choice([j for j in range(k) if j != pref]))
        p.trials.append(GenericTrial(c, k, feats, "none", block=0, phase="main"))
    return p


def make_population(
    n: int,
    T: int,
    *,
    hetero: bool,
    seed: int,
    k: int = 2,
    k_max: int = 4,
    offset: int = 0,
    kind: str = "bandit",
) -> list[EncodedSeq]:
    out = []
    shared = stream(seed, "toy_shared_table").integers(0, k, size=N_CUES)
    for i in range(offset, offset + n):
        rng = stream(seed, "toy_population", kind, hetero, i)
        if kind == "cue":
            table = rng.integers(0, k, size=N_CUES) if hetero else shared
            s = encode(simulate_cue_person(rng, table, T, k, i), "toy", k_max)
            s.meta = {"table": table.tolist()}
        else:
            prm = person_params(rng, hetero)
            s = encode(simulate_person(rng, prm, T, k, i), "toy", k_max)
            s.meta = {"params": prm}
        out.append(s)
    return out
