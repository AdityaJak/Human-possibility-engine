"""Scoring and statistics shared by every Phase III experiment (thesis §5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hpe.core.rng import stream
from hpe.population.core.encoding import EncodedSeq

LAPSE_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)


def apply_lapse(logp: np.ndarray, avail: np.ndarray, lam: float) -> np.ndarray:
    """``log((1-λ)p + λ/K_avail)`` on available actions; ``-inf`` elsewhere."""
    if lam == 0.0:
        return logp
    k = avail.sum(axis=1, keepdims=True).astype(np.float64)
    p = np.exp(logp) * (1.0 - lam) + lam / k
    with np.errstate(divide="ignore"):
        out = np.where(avail, np.log(p), -np.inf)
    return out


@dataclass(slots=True)
class ParticipantScore:
    key: str
    nll: float
    n: int
    top1: float
    ece: float
    quarters: tuple[float, float, float, float]


def score(seq: EncodedSeq, logp: np.ndarray, lam: float = 0.0) -> ParticipantScore:
    if logp.shape != (seq.T, seq.k_max):
        raise ValueError(f"{seq.key}: logp shape {logp.shape} != {(seq.T, seq.k_max)}")
    lp = apply_lapse(logp, seq.avail, lam)
    idx = np.flatnonzero(seq.scored)
    chosen = lp[idx, seq.actions[idx]]
    if not np.all(np.isfinite(chosen)):
        raise ValueError(f"{seq.key}: non-finite log-probability of a chosen, available action")
    nll = -chosen
    probs = np.exp(lp[idx])
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    hit = (pred == seq.actions[idx]).astype(np.float64)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    qs = np.array_split(nll, 4)
    qm = [float(q.mean()) if q.size else float("nan") for q in qs]
    quarters = (qm[0], qm[1], qm[2], qm[3])
    return ParticipantScore(
        seq.key, float(nll.mean()), int(idx.size), float(hit.mean()), float(ece), quarters
    )  # type: ignore[arg-type]


def mean_nll(scores: Sequence[ParticipantScore]) -> float:
    return float(np.mean([s.nll for s in scores])) if scores else float("nan")


def bootstrap(
    diff: np.ndarray, *, seed: int, key: tuple[str, ...], n_boot: int = 2000
) -> tuple[float, float, float, float]:
    """Mean, 95 % percentile CI, one-sided p for H0: mean >= 0 (i.e. evidence that the first model is better)."""
    rng = stream(seed, "phase3_bootstrap", *key)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_one = float(max(1.0 / n_boot, (means >= 0).mean()))
    return float(diff.mean()), float(lo), float(hi), p_one


def paired(
    a: Sequence[ParticipantScore], b: Sequence[ParticipantScore], *, seed: int, key: tuple[str, ...]
) -> dict[str, float | int]:
    """Contrast a − b on participants scored by both (negative favours a)."""
    da = {s.key: s.nll for s in a}
    db = {s.key: s.nll for s in b}
    keys = sorted(set(da) & set(db))
    if not keys:
        raise ValueError("no common participants")
    d = np.array([da[k] - db[k] for k in keys])
    est, lo, hi, p = bootstrap(d, seed=seed, key=key)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "p_one_sided": p, "sd": sd, "n": len(keys)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def passes(contrast: dict[str, float | int], margin: float, p_adj: float, alpha: float = 0.05) -> bool:
    return float(contrast["estimate"]) <= -margin and p_adj < alpha
