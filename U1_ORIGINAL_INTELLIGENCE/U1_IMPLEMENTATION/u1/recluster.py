"""Offline re-clustering of stored worlds (dev finding D-2: additive MAP consolidation alone collapses the library).

Collapsed Gibbs sampling over the concept of every consolidated entity in every stored world, under the same
generative model the agent uses online (an infinite-relational-model-style clustering: Kemp et al. 2006/2010):
CRP(α) membership; Beta(½,½) laws per concept and probe bit and per concept pair (reaction);
Beta(1,1) appearance-sign law per concept and feature (cue reliability fixed at 1 because stored training worlds
come from one appearance regime). The final assignment is rebuilt into a ``Library``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln

from .env import D_APP, N_BITS, action_entities, decode_bits, is_pair
from .ira import Library


@dataclass
class WorldRecord:
    x: np.ndarray  # [E, D_APP]
    obs: list[tuple[int, int]]  # active (action, outcome) pairs


def _items(
    records: list[WorldRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int, int]]], list[tuple[int, int]]]:
    """Items = (world, entity) with at least one observation."""
    index: dict[tuple[int, int], int] = {}
    on, off, app = [], [], []
    partners: list[list[tuple[int, int, int]]] = []
    for w, rec in enumerate(records):
        for a, _ in rec.obs:
            for e in action_entities(a):
                if (w, e) not in index:
                    index[(w, e)] = len(index)
                    on.append(np.zeros(N_BITS))
                    off.append(np.zeros(N_BITS))
                    app.append((rec.x[e] > 0).astype(int))
                    partners.append([])
        pair_counts: dict[tuple[int, int], list[int]] = {}
        for a, o in rec.obs:
            ents = action_entities(a)
            if is_pair(a):
                pair_counts.setdefault(ents, [0, 0])[0 if o else 1] += 1
            else:
                i = index[(w, ents[0])]
                bits = decode_bits(o)
                on[i] += bits
                off[i] += 1 - bits
        for (ea, eb), (s, f) in sorted(pair_counts.items()):
            ia, ib = index[(w, ea)], index[(w, eb)]
            partners[ia].append((ib, s, f))
            partners[ib].append((ia, s, f))
    keys = sorted(index, key=lambda k: index[k])
    return np.array(on), np.array(off), np.array(app), partners, keys


def recluster(
    records: list[WorldRecord],
    rng: np.random.Generator,
    alpha: float = 1.0,
    sweeps: int = 40,
    greedy_sweeps: int = 5,
    k_init: int = 8,
    k_max: int = 40,
) -> tuple[Library, dict[tuple[int, int], int]]:
    """Returns the rebuilt library and the concept of every (world, entity) item."""
    on, off, app, partners, keys = _items(records)
    n_items = len(on)
    pooled_s = sum(o for rec in records for a, o in rec.obs if is_pair(a))
    pooled_f = sum(1 - o for rec in records for a, o in rec.obs if is_pair(a))
    rb = 0.5  # R1: symmetric Beta(1/2, 1/2) prior on pair laws (attempt 1 used the selection-biased pooled rate)
    z = rng.integers(0, k_init, size=n_items)
    bon = np.zeros((k_max, N_BITS))
    boff = np.zeros((k_max, N_BITS))
    apos = np.zeros((k_max, D_APP))
    aneg = np.zeros((k_max, D_APP))
    n = np.zeros(k_max)
    rs = np.zeros((k_max, k_max))
    rf = np.zeros((k_max, k_max))

    def add(i: int, k: int, sign: int) -> None:
        bon[k] += sign * on[i]
        boff[k] += sign * off[i]
        apos[k] += sign * app[i]
        aneg[k] += sign * (1 - app[i])
        n[k] += sign
        for j, s, f in partners[i]:
            lk = z[j]
            if j == i:
                continue
            rs[k, lk] += sign * s
            rf[k, lk] += sign * f
            if lk != k:
                rs[lk, k] += sign * s
                rf[lk, k] += sign * f

    # pairs are stored on both endpoints; add each pair once when building
    for i in range(n_items):
        k = z[i]
        bon[k] += on[i]
        boff[k] += off[i]
        apos[k] += app[i]
        aneg[k] += 1 - app[i]
        n[k] += 1
    for i in range(n_items):
        for j, s, f in partners[i]:
            if j > i:
                ki, kj = z[i], z[j]
                rs[ki, kj] += s
                rf[ki, kj] += f
                if ki != kj:
                    rs[kj, ki] += s
                    rf[kj, ki] += f

    total = sweeps + greedy_sweeps
    for sweep in range(total):
        greedy = sweep >= sweeps
        for i in rng.permutation(n_items):
            add(i, z[i], -1)
            used = np.flatnonzero(n > 0)
            free = np.flatnonzero(n == 0)
            cand = np.concatenate([used, free[:1]]) if len(free) else used
            logp = np.where(n[cand] > 0, np.log(np.maximum(n[cand], 1e-300)), np.log(alpha))
            logp = logp + (
                betaln(0.5 + bon[cand] + on[i], 0.5 + boff[cand] + off[i])
                - betaln(0.5 + bon[cand], 0.5 + boff[cand])
            ).sum(1)
            logp = logp + (
                betaln(1 + apos[cand] + app[i], 1 + aneg[cand] + 1 - app[i])
                - betaln(1 + apos[cand], 1 + aneg[cand])
            ).sum(1)
            if partners[i]:
                by_l: dict[int, list[int]] = {}
                for j, s, f in partners[i]:
                    acc = by_l.setdefault(int(z[j]), [0, 0])
                    acc[0] += s
                    acc[1] += f
                for lk, (s, f) in by_l.items():
                    a0, b0 = rb + rs[cand, lk], 1 - rb + rf[cand, lk]
                    logp = logp + betaln(a0 + s, b0 + f) - betaln(a0, b0)
            if greedy:
                k_new = int(cand[int(np.argmax(logp))])
            else:
                p = np.exp(logp - logp.max())
                k_new = int(cand[int(rng.choice(len(cand), p=p / p.sum()))])
            z[i] = k_new
            add(i, k_new, +1)
    # rebuild the library with compact concept ids, ordered by size
    used = sorted(np.flatnonzero(n > 0).tolist(), key=lambda k: (-n[k], k))
    remap = {k: c for c, k in enumerate(used)}
    lib = Library()
    for k in used:
        c = lib.add_concept()
        lib.bits[c][:, 0] = bon[k]
        lib.bits[c][:, 1] = boff[k]
        lib.app[c][:, 0] = aneg[k]
        lib.app[c][:, 1] = apos[k]
        lib.n[c] = n[k]
    for k in used:
        for lk in used:
            ck, cl = remap[k], remap[lk]
            if ck <= cl and (rs[k, lk] + rf[k, lk]) > 0:
                lib.react[(ck, cl)] = [float(rs[k, lk]), float(rf[k, lk])]
    lib.pooled = [float(pooled_s), float(pooled_f)]
    lib.worlds = len(records)
    return lib, {key: remap[int(z[i])] for i, key in enumerate(keys)}
