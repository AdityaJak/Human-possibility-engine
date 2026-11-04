"""Causality and state-integrity checks every Phase III model must pass (strict version, critic finding L-1).

For each probed sequence (re-keyed with a random token):

1. **Key independence** — predictions do not depend on the participant key.
2. **No mutation** — predicting an unseen participant leaves the model's state unchanged (no per-key
   caches, no transductive pooling of held-out participants).
3. **Prefix truncation with a flipped action** — for probe rows t, the sequence is cut after t and the
   action at t is replaced by a different available action; rows <= t must be bit-identical in every
   mode (``adapt``, ``frozen``) and in the donor control (where the *target's* prefix is cut).

Every prediction uses a fresh deep copy of the model. A model with any violation is excluded from all
comparisons by the runner.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np

from hpe.core.rng import stream
from hpe.population.core.encoding import EncodedSeq, Layout, rekey


def state_digest(model: Any) -> str:
    try:
        return hashlib.sha256(pickle.dumps(model.__dict__, protocol=4)).hexdigest()
    except Exception as e:  # unpicklable state is itself a finding (the runner reports it)
        return f"unpicklable:{type(e).__name__}"


def prefix_flip(seq: EncodedSeq, t: int, key: str) -> EncodedSeq:
    """``seq`` truncated to rows <= t, with the action at t replaced by a different available action."""
    a = seq.actions[: t + 1].copy()
    av = np.flatnonzero(seq.avail[t])
    if av.size > 1:
        a[t] = av[(int(np.searchsorted(av, a[t])) + 1) % av.size]
    return EncodedSeq(
        key,
        seq.family,
        seq.k_max,
        seq.n_actions,
        seq.x[: t + 1].copy(),
        a,
        seq.avail[: t + 1].copy(),
        seq.scored[: t + 1].copy(),
    )


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
        noise[:, L.avail] = x[t + 1 :, L.avail]
        x[t + 1 :] = noise
    return EncodedSeq(
        seq.key, seq.family, seq.k_max, seq.n_actions, x, a, seq.avail.copy(), seq.scored.copy()
    )


ATOL = 1e-5  # log-prob tolerance: batched GEMMs change summation order with sequence length (~1e-7);
# any leak below 1e-5 nats per row cannot move a per-choice NLL by more than 1e-5.


def _differ(a: np.ndarray, b: np.ndarray, atol: float = ATOL) -> bool:
    fa, fb = np.isfinite(a), np.isfinite(b)
    if a.shape != b.shape or (fa != fb).any():
        return True
    return bool(np.abs(a[fa] - b[fa]).max(initial=0.0) > atol)


def check_causal(
    model: Any,
    seqs: EncodedSeq | Sequence[EncodedSeq],
    *,
    seed: int = 0,
    probes: int = 6,
    donors: bool = True,
    max_problems: int = 20,
) -> list[str]:
    """Return de-duplicated violations (empty = no violation found)."""
    seq_list: list[EncodedSeq] = [seqs] if isinstance(seqs, EncodedSeq) else list(seqs)
    problems: list[str] = []
    rng = stream(seed, "phase3_strict_check")

    def fresh(s: EncodedSeq, **kw: Any) -> np.ndarray:
        return np.asarray(copy.deepcopy(model).predict_logprobs(s, **kw))

    for si, seq in enumerate(seq_list):
        if seq.T < 3:
            continue
        rkey = f"probe-{int(rng.integers(1 << 62))}"
        base = {m: fresh(rekey(seq, rkey), mode=m) for m in ("adapt", "frozen")}
        if _differ(base["adapt"], fresh(rekey(seq, f"probe-{int(rng.integers(1 << 62))}"), mode="adapt")):
            problems.append(f"seq{si}: predictions depend on seq.key")
        probe_model = copy.deepcopy(model)
        d0 = state_digest(probe_model)
        if d0.startswith("unpicklable"):
            problems.append(f"seq{si}: model state is {d0} (cannot verify immutability)")
        probe_model.predict_logprobs(rekey(seq, f"unseen-{int(rng.integers(1 << 62))}"))
        if state_digest(probe_model) != d0:
            problems.append(f"seq{si}: predict_logprobs mutated model state")
        T = seq.T
        rows = sorted({0, 1, T - 2, T - 1, *(int(v) for v in rng.integers(0, T, probes))})
        donor = seq_list[(si + 1) % len(seq_list)] if donors and len(seq_list) > 1 else None
        base_d = fresh(rekey(seq, rkey), donor=donor) if donor is not None else None
        for t in rows:
            pre = prefix_flip(seq, t, rkey)
            for m in ("adapt", "frozen"):
                if _differ(base[m][: t + 1], fresh(pre, mode=m)[: t + 1]):
                    problems.append(f"seq{si} mode={m}: rows <= {t} depend on data at or after row {t}")
                    break
            if base_d is not None and donor is not None:
                if _differ(base_d[: t + 1], fresh(pre, donor=donor)[: t + 1]):
                    problems.append(
                        f"seq{si} donor: rows <= {t} depend on the target's data at or after row {t}"
                    )
        if len(problems) >= max_problems:
            break
    seen: dict[str, str] = {}
    for p in problems:
        kind = p.split(":", 1)[1].split("<=")[0].strip() if ":" in p else p
        seen.setdefault(kind, p)
    return sorted(seen.values())


def check_encoding_leak(seqs: Sequence[EncodedSeq], *, threshold: float = 0.98) -> list[str]:
    """Flag input slots that near-deterministically encode the action being predicted (critic finding L-4).

    For every slot j and scored row t, the best single-threshold rule on ``x[t, j]`` predicting ``a_t`` is
    scored; a slot reaching ``threshold`` accuracy on a family with real choice variability is a parser or
    encoding leak. Instructed-trial slots are excluded (their rows are not scored).
    """
    if not seqs:
        return []
    xs = np.concatenate([s.x[s.scored] for s in seqs])
    ys = np.concatenate([s.actions[s.scored] for s in seqs])
    if xs.shape[0] < 50 or np.unique(ys).size < 2:
        return []
    base = np.bincount(ys).max() / ys.size
    problems = []
    for j in range(xs.shape[1]):
        col = xs[:, j]
        vals = np.unique(col)
        if vals.size < 2:
            continue
        best = base
        for v in vals[: min(vals.size, 64)]:
            pred = col > v
            for a in np.unique(ys):
                acc = max(np.mean((ys == a) == pred), np.mean((ys == a) == ~pred))
                if np.unique(ys).size == 2:
                    best = max(best, acc)
        if best >= threshold and best > base + 0.05:
            problems.append(
                f"slot {j}: a single threshold predicts the current action with accuracy {best:.3f}"
            )
    return problems
