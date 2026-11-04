"""Causality (no look-ahead) check every Phase III model must pass before its numbers are used."""

from __future__ import annotations

import numpy as np

from hpe.core.rng import stream
from hpe.population.core.encoding import EncodedSeq, Layout
from hpe.population.core.interface import SequencePredictor


def perturb_future(seq: EncodedSeq, t: int, seed: int) -> EncodedSeq:
    """Copy of ``seq`` with every action at >= t and every input at > t replaced by random values."""
    rng = stream(seed, "phase3_causal", seq.key, t)
    x = seq.x.copy()
    a = seq.actions.copy()
    L = Layout(seq.k_max)
    k_avail = seq.avail.sum(axis=1)
    a[t:] = rng.integers(0, k_avail[t:])
    if t + 1 < seq.T:
        noise = rng.uniform(0.0, 1.0, size=x[t + 1 :].shape).astype(np.float32)
        noise[:, L.avail] = x[t + 1 :, L.avail]  # keep availability coherent with the target arrays
        x[t + 1 :] = noise
    return EncodedSeq(
        seq.key, seq.family, seq.k_max, seq.n_actions, x, a, seq.avail.copy(), seq.scored.copy()
    )


def check_causal(model: SequencePredictor, seq: EncodedSeq, *, seed: int = 0, probes: int = 3) -> list[str]:
    """Return a list of violations (empty = causal) for ``mode='adapt'`` and ``mode='frozen'``."""
    problems: list[str] = []
    if seq.T < 4:
        return ["sequence too short for a causality probe"]
    ts = sorted({max(1, seq.T // 4), seq.T // 2, max(1, (3 * seq.T) // 4)})[:probes]
    for mode in ("adapt", "frozen"):
        base = model.predict_logprobs(seq, mode=mode)  # type: ignore[arg-type]
        for t in ts:
            pert = model.predict_logprobs(perturb_future(seq, t, seed), mode=mode)  # type: ignore[arg-type]
            same = np.array_equal(base[: t + 1], pert[: t + 1])
            if not same:
                a, b = base[: t + 1], pert[: t + 1]
                fin = np.isfinite(a) & np.isfinite(b)
                diff = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else float("inf")
                problems.append(
                    f"{model.name} mode={mode}: rows <= {t} changed when the future changed (max |Δ| {diff:.3g})"
                )
    return problems
