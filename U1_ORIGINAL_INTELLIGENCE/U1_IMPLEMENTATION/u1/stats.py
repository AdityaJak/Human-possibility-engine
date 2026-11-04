"""Preregistered analysis, revision R1 (EXPERIMENTS_AND_BASELINES.md §R1.4-§R1.5).

The unit of inference is the **universe** (critique F6): each universe has its own trained library and comparator
weights, so the per-universe mean paired difference is one replication. Tests are one-sided t-tests over the
universe means (df = #universes - 1). World-level paired bootstrap intervals are reported as descriptive detail.
Missing cells are an error, never silently dropped (critique F13).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from .env import EVAL_CONDITIONS, SHIFT_CONDITIONS

UNIVERSES = {"dev": (0,), "confirm": (4, 5, 6, 7, 8)}
N_WORLDS = {"dev": 200, "confirm": 100}
COMPARATORS = ("gru_narrow", "gru_broad", "rel_broad")
PRIMARY_MARGIN = 0.10
IID_NONINFERIORITY = -0.05
MIN_POSITIVE_UNIVERSES = 4  # of 5
MECHANISMS = [
    ("M1 cue-validity tracking", "ira_nocuetrack", ("cue_reversed",), 0.05),
    ("M2 open concept set", "ira_closed", ("novel_role",), 0.05),
    ("M3 re-identification", "ira_nochange", ("role_switch",), 0.05),
    ("M4 active identification (information gain)", "ira_noactive", EVAL_CONDITIONS, 0.03),
    ("M5 cross-world library", "ira_nolibrary", ("iid",), 0.03),
]
ACC_MARGIN = 0.05
AGENTS = (
    "ira",
    "ira_noactive",
    "ira_thompson",
    "ira_closed",
    "ira_nocuetrack",
    "ira_nochange",
    "ira_nolibrary",
    *COMPARATORS,
    "entity_bayes",
    "ucb",
    "random",
)


class IncompleteResults(RuntimeError):
    pass


def load(root: Path, phase: str, agent: str, cond: str, key: str = "norm", strict: bool = True) -> dict:
    rows: dict[tuple[int, int], float] = {}
    for u in UNIVERSES[phase]:
        p = root / f"u{u}" / phase / f"{agent}__{cond}.jsonl"
        if not p.exists():
            if strict:
                raise IncompleteResults(f"missing {p}")
            continue
        lines = [json.loads(x) for x in p.read_text().splitlines()]
        if strict and len(lines) != N_WORLDS[phase]:
            raise IncompleteResults(f"{p}: {len(lines)} worlds, expected {N_WORLDS[phase]}")
        for r in lines:
            rows[(u, int(r["seed"]))] = float(r[key])
    return rows


def contrast(a: dict, b: dict, seed: int = 0, n_boot: int = 10000) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise IncompleteResults("unpaired cells")
    d = np.array([a[k] - b[k] for k in keys])
    us = sorted({k[0] for k in keys})
    du = np.array([np.mean([a[k] - b[k] for k in keys if k[0] == u]) for u in us])
    est = float(du.mean())
    if len(du) > 1 and du.std(ddof=1) > 0:
        tstat = est / (du.std(ddof=1) / np.sqrt(len(du)))
        p = float(sps.t.sf(tstat, df=len(du) - 1))
    else:
        tstat, p = float("nan"), (0.0 if est > 0 else 1.0)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {
        "estimate": round(est, 4),
        "per_universe": {str(u): round(float(v), 4) for u, v in zip(us, du, strict=True)},
        "n_universes_positive": int((du > 0).sum()),
        "t": round(float(tstat), 3),
        "p_universe_t": p,
        "world_level_ci95": [
            round(float(np.quantile(boots, 0.025)), 4),
            round(float(np.quantile(boots, 0.975)), 4),
        ],
        "n_worlds": len(d),
    }


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def summarize(root: Path, phase: str) -> dict[str, Any]:
    strict = phase == "confirm"
    table: dict[str, dict[str, float]] = {}
    for ag in AGENTS:
        for c in EVAL_CONDITIONS:
            s = load(root, phase, ag, c, strict=False)
            if s:
                table.setdefault(ag, {})[c] = round(float(np.mean(list(s.values()))), 4)
    out: dict[str, Any] = {"phase": phase, "universes": UNIVERSES[phase], "table": table}
    ira = {c: load(root, phase, "ira", c, strict=strict) for c in EVAL_CONDITIONS}
    primary: dict[str, Any] = {}
    for comp in COMPARATORS:
        present = any((root / f"u{u}" / phase).glob(f"{comp}__*.jsonl") for u in UNIVERSES[phase])
        if not present and comp != COMPARATORS[0]:
            primary[comp] = {"status": "not run (stage 2 runs only if H1 holds against gru_narrow)"}
            continue
        try:
            cs = {
                c: contrast(ira[c], load(root, phase, comp, c, strict=strict), seed=11)
                for c in EVAL_CONDITIONS
            }
        except IncompleteResults:
            if strict:
                raise
            continue
        adj = holm([cs[c]["p_universe_t"] for c in SHIFT_CONDITIONS])
        n_pass = 0
        for c, pa in zip(SHIFT_CONDITIONS, adj, strict=True):
            r = cs[c]
            r["p_holm"] = pa
            r["passes"] = bool(
                r["estimate"] >= PRIMARY_MARGIN
                and pa < 0.05
                and r["n_universes_positive"] >= MIN_POSITIVE_UNIVERSES
            )
            n_pass += r["passes"]
        primary[comp] = {
            "contrasts": cs,
            "shift_conditions_passed": n_pass,
            "iid_noninferior": bool(cs["iid"]["estimate"] >= IID_NONINFERIORITY),
            "H1_vs_this_comparator": bool(n_pass >= 3 and cs["iid"]["estimate"] >= IID_NONINFERIORITY),
        }
    out["primary"] = primary
    out["H1_supported"] = bool(primary) and all(
        v.get("H1_vs_this_comparator", False) for v in primary.values()
    )
    out["stage2_warranted"] = bool(primary.get(COMPARATORS[0], {}).get("H1_vs_this_comparator", False))
    mech = []
    for name, comp, conds, margin in MECHANISMS:
        a_m: dict = {}
        b_m: dict = {}
        try:
            ia = [load(root, phase, "ira", c, strict=strict) for c in conds]
            cb = [load(root, phase, comp, c, strict=strict) for c in conds]
        except IncompleteResults:
            if strict:
                raise
            continue
        keys = set.intersection(*(set(x) for x in ia + cb))
        for k in keys:
            a_m[k] = float(np.mean([x[k] for x in ia]))
            b_m[k] = float(np.mean([x[k] for x in cb]))
        r = contrast(a_m, b_m, seed=12)
        r.update({"mechanism": name, "comparator": comp, "conditions": list(conds), "margin": margin})
        mech.append(r)
    for r, pa in zip(mech, holm([r["p_universe_t"] for r in mech]), strict=True):
        r["p_holm"] = pa
        r["passes"] = bool(r["estimate"] >= r["margin"] and pa < 0.05)
    out["mechanisms"] = mech
    acc_a, acc_f = {}, {}
    for u in UNIVERSES[phase]:
        p = root / f"u{u}" / phase / "accumulation.jsonl"
        if p.exists():
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            for r in rows[len(rows) // 2 :]:
                acc_a[(u, r["seed"])] = r["accumulating"]
                acc_f[(u, r["seed"])] = r["frozen"]
        elif strict:
            raise IncompleteResults(f"missing {p}")
    if acc_a:
        r = contrast(acc_a, acc_f, seed=13)
        r["passes"] = bool(r["estimate"] >= ACC_MARGIN and r["p_universe_t"] < 0.05)
        out["accumulation"] = r
    diag: dict[str, Any] = {}
    for ag in ("ira", "ira_nochange", *COMPARATORS):
        s = load(root, phase, ag, "role_switch", "norm_post_switch", strict=False)
        if s:
            diag[f"{ag} post-switch norm"] = round(float(np.mean(list(s.values()))), 4)
    for ag in ("ira", *COMPARATORS):
        for c in EVAL_CONDITIONS:
            s = load(root, phase, ag, c, "probes", strict=False)
            if s:
                diag[f"{ag} {c} probes per world"] = round(float(np.mean(list(s.values()))), 2)
    out["diagnostics"] = diag
    (root / f"summary_{phase}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return out
