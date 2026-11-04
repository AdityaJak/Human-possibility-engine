"""Preregistered analysis (EXPERIMENTS_AND_BASELINES.md §4-§5). Paired over (universe, world seed)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .env import CONDITIONS, SHIFT_CONDITIONS

EVAL_CONDITIONS = CONDITIONS[1:]
UNIVERSES = {"dev": (0,), "confirm": (1, 2, 3)}
METARL_SEEDS = ("metarl_s0", "metarl_s1")
PRIMARY_MARGIN = 0.10
IID_NONINFERIORITY = -0.05
MECHANISMS = [
    ("M1 cue-validity tracking", "ira_nocuetrack", ("cue_reversed",), 0.05),
    ("M2 open concept set", "ira_closed", ("novel_role",), 0.05),
    ("M3 re-identification", "ira_nochange", ("role_switch",), 0.05),
    ("M4 active identification", "ira_noactive", EVAL_CONDITIONS, 0.03),
    ("M5 cross-world library", "ira_nolibrary", ("iid", "recombined"), 0.03),
    ("M6 concepts vs. none", "entity_bayes", ("iid",), 0.10),
]
ACC_MARGIN = 0.05


def _load(root: Path, phase: str, agent: str, cond: str) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for u in UNIVERSES[phase]:
        p = root / f"u{u}" / phase / f"{agent}__{cond}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows[(u, int(r["seed"]))] = r
    return rows


def scores(root: Path, phase: str, agent: str, cond: str, key: str = "norm") -> dict[tuple[int, int], float]:
    if agent == "metarl":  # the comparator is the mean over independently trained seeds
        parts = [_load(root, phase, a, cond) for a in METARL_SEEDS]
        parts = [p for p in parts if p]
        if not parts:
            return {}
        keys = set.intersection(*(set(p) for p in parts))
        return {k: float(np.mean([p[k][key] for p in parts])) for k in keys}
    return {k: float(v[key]) for k, v in _load(root, phase, agent, cond).items()}


def paired(a: dict, b: dict, seed: int = 0, n_boot: int = 10000) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return {"n": len(keys)}
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else math.inf
    p_t = 0.5 * math.erfc(t / math.sqrt(2))  # one-sided, normal approximation (n >= 50)
    p_boot = float((boots <= 0).mean())
    per_u = {}
    for u in sorted({k[0] for k in keys}):
        du = np.array([a[k] - b[k] for k in keys if k[0] == u])
        per_u[str(u)] = round(float(du.mean()), 4)
    return {
        "n": len(keys),
        "estimate": round(float(d.mean()), 4),
        "ci95": [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)],
        "p_one_sided": max(p_boot, p_t),
        "sd": round(sd, 4),
        "per_universe": per_u,
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
    agents = [
        "ira",
        "ira_noactive",
        "ira_closed",
        "ira_nocuetrack",
        "ira_nochange",
        "ira_nolibrary",
        "metarl",
        *METARL_SEEDS,
        "entity_bayes",
        "ucb",
        "random",
    ]
    table: dict[str, dict[str, Any]] = {}
    for ag in agents:
        for c in EVAL_CONDITIONS:
            s = scores(root, phase, ag, c)
            if s:
                v = np.array(list(s.values()))
                table.setdefault(ag, {})[c] = {"mean": round(float(v.mean()), 4), "n": len(v)}
    out: dict[str, Any] = {"phase": phase, "universes": UNIVERSES[phase], "table": table}
    ira = {c: scores(root, phase, "ira", c) for c in EVAL_CONDITIONS}
    mrl = {c: scores(root, phase, "metarl", c) for c in EVAL_CONDITIONS}
    prim = {c: paired(ira[c], mrl[c], seed=11) for c in EVAL_CONDITIONS}
    shift_ps = [prim[c].get("p_one_sided", 1.0) for c in SHIFT_CONDITIONS]
    adj = holm(shift_ps)
    n_pass = 0
    for c, pa in zip(SHIFT_CONDITIONS, adj, strict=True):
        r = prim[c]
        r["p_holm"] = pa
        r["passes"] = bool(
            r.get("estimate", -1) >= PRIMARY_MARGIN
            and pa < 0.05
            and all(v > 0 for v in r.get("per_universe", {"x": -1}).values())
        )
        n_pass += r["passes"]
    iid_ok = prim["iid"].get("estimate", -1) >= IID_NONINFERIORITY
    out["primary"] = {
        "contrasts": prim,
        "shift_conditions_passed": n_pass,
        "iid_noninferior": iid_ok,
        "H1_supported": bool(n_pass >= 3 and iid_ok),
    }
    mech = []
    for name, comp, conds, margin in MECHANISMS:
        a_s: dict = {}
        b_s: dict = {}
        for c in conds:  # average over the listed conditions per (universe, seed)
            ia, cb = scores(root, phase, "ira", c), scores(root, phase, comp, c)
            for k in set(ia) & set(cb):
                a_s.setdefault(k, []).append(ia[k])
                b_s.setdefault(k, []).append(cb[k])
        a_m = {k: float(np.mean(v)) for k, v in a_s.items() if len(v) == len(conds)}
        b_m = {k: float(np.mean(v)) for k, v in b_s.items() if len(v) == len(conds)}
        r = paired(a_m, b_m, seed=12)
        r.update({"mechanism": name, "comparator": comp, "conditions": list(conds), "margin": margin})
        mech.append(r)
    for r, pa in zip(mech, holm([r.get("p_one_sided", 1.0) for r in mech]), strict=True):
        r["p_holm"] = pa
        r["passes"] = bool(r.get("estimate", -1) >= r["margin"] and pa < 0.05)
    out["mechanisms"] = mech
    acc_a, acc_f = {}, {}
    for u in UNIVERSES[phase]:
        p = root / f"u{u}" / phase / "accumulation.jsonl"
        if p.exists():
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            for r in rows[len(rows) // 2 :]:
                acc_a[(u, r["seed"])] = r["accumulating"]
                acc_f[(u, r["seed"])] = r["frozen"]
    if acc_a:
        r = paired(acc_a, acc_f, seed=13)
        r["passes"] = bool(r.get("estimate", -1) >= ACC_MARGIN and r.get("p_one_sided", 1) < 0.05)
        out["accumulation"] = r
    diag = {}
    for ag in ("ira", "metarl", "ira_nochange"):
        s = scores(root, phase, ag, "role_switch", "norm_post_switch")
        if s:
            diag[f"{ag} post-switch norm"] = round(float(np.mean(list(s.values()))), 4)
    for ag in ("ira", "metarl", "entity_bayes", "ucb"):
        for c in ("iid", "novel_role"):
            s = scores(root, phase, ag, c, "first_reaction")
            if s:
                diag[f"{ag} {c} steps to first reaction"] = round(float(np.mean(list(s.values()))), 2)
    out["diagnostics"] = diag
    (root / f"summary_{phase}.json").write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps(
            {
                "table": table,
                "primary": out["primary"],
                "mechanisms": mech,
                "accumulation": out.get("accumulation"),
                "diagnostics": diag,
            },
            indent=1,
            default=str,
        )
    )
    return out
