"""U1 experiment runner (preregistration: ``U1_ORIGINAL_INTELLIGENCE/EXPERIMENTS_AND_BASELINES.md``).

    python -m u1.experiment train-ira     --universe U --beta B --out DIR
    python -m u1.experiment train-metarl  --universe U --hidden H --lr LR --updates N --seed S --out DIR
    python -m u1.experiment eval          --universe U --phase dev|confirm --agents a,b --conditions c,d --out DIR
    python -m u1.experiment accumulate    --universe U --phase dev|confirm --out DIR
    python -m u1.experiment freeze        --out DIR        # hashes code + preregistration; required for confirm
    python -m u1.experiment summarize     --out DIR

Seeds: training worlds 100000+ (condition 'train'); dev evaluation 0-199; confirmatory evaluation 10000-10199;
accumulation sequences dev 30000-30049, confirmatory 20000-20049. Universe 0 is the development universe;
universes 1-3 are confirmatory and are not used before the freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import EntityBayes, RandomPairs, UCBPairs
from .env import CONDITIONS, make_universe, make_world
from .ira import IRA, IRAConfig, Library
from .play import play
from .train import train_library

ROOT = Path(__file__).resolve().parents[2]  # U1_ORIGINAL_INTELLIGENCE/
PREREG = ROOT / "EXPERIMENTS_AND_BASELINES.md"
DEV_UNIVERSES = (0,)
CONFIRM_UNIVERSES = (1, 2, 3)
SEEDS = {"dev": range(0, 200), "confirm": range(10000, 10200)}
ACC_SEEDS = {"dev": range(30000, 30050), "confirm": range(20000, 20050)}
IRA_VARIANTS = {
    "ira": {},
    "ira_noactive": {"active": False},
    "ira_closed": {"open_set": False},
    "ira_nocuetrack": {"cue_track": False},
    "ira_nochange": {"change": False},
    "ira_nolibrary": {},
}


def code_hash() -> dict[str, str]:
    files = [*sorted((ROOT / "U1_IMPLEMENTATION" / "u1").glob("*.py")), PREREG]
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def check_phase(universe: int, phase: str, out: Path) -> None:
    if phase == "dev":
        if universe not in DEV_UNIVERSES:
            sys.exit(f"universe {universe} is confirmatory; dev runs use {DEV_UNIVERSES}")
        return
    if universe not in CONFIRM_UNIVERSES:
        sys.exit(f"confirmatory runs use universes {CONFIRM_UNIVERSES}")
    freeze = out / "FREEZE.json"
    if not freeze.exists():
        sys.exit("no FREEZE.json: run `freeze` after the preregistration is final")
    frozen = json.loads(freeze.read_text())["hashes"]
    if frozen != code_hash():
        sys.exit("code or preregistration changed after the freeze; refusing to run confirmatory evaluation")


def udir(out: Path, universe: int) -> Path:
    d = out / f"u{universe}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_settings(out: Path) -> dict[str, Any]:
    p = out / "settings.json"
    return (
        json.loads(p.read_text())
        if p.exists()
        else {"beta": 1.0, "metarl_greedy": True, "metarl_hidden": 128}
    )


def ira_cfg(name: str, beta: float) -> IRAConfig:
    return IRAConfig(beta=beta, **IRA_VARIANTS[name])


def make_agent(name: str, universe: int, seed: int, out: Path, cache: dict[str, Any]) -> Any:
    st = load_settings(out)
    d = udir(out, universe)
    if name.startswith("ira"):
        if name == "ira_nolibrary":
            lib = Library()
        else:
            if "lib" not in cache:
                cache["lib"] = Library.from_json(json.loads((d / "ira_library.json").read_text()))
            lib = cache["lib"]
        return IRA(lib, ira_cfg(name, st["beta"]), np.random.default_rng([universe, seed, 606]))
    if name == "entity_bayes":
        if "lib" not in cache:
            cache["lib"] = Library.from_json(json.loads((d / "ira_library.json").read_text()))
        return EntityBayes(np.random.default_rng([universe, seed, 607]), cache["lib"].rbar())
    if name == "ucb":
        return UCBPairs()
    if name == "random":
        return RandomPairs(np.random.default_rng([universe, seed, 608]))
    if name.startswith("metarl"):
        import torch

        from .metarl import MetaRLAgent, Net

        torch.set_num_threads(1)
        tag = name.split("_", 1)[1] if "_" in name else "s0"
        key = f"net_{tag}"
        if key not in cache:
            info = json.loads((d / f"metarl_{tag}.json").read_text())
            net = Net(info["config"]["hidden"])
            net.load_state_dict(torch.load(d / f"metarl_{tag}.pt"))
            net.eval()
            cache[key] = (net, info["config"]["hidden"])
        net, hidden = cache[key]
        return MetaRLAgent(net, hidden, st["metarl_greedy"], np.random.default_rng([universe, seed, 609]))
    raise ValueError(name)


def cmd_train_ira(a: argparse.Namespace) -> None:
    if a.universe not in DEV_UNIVERSES:
        check_phase(a.universe, "confirm", Path(a.out))
    u = make_universe(a.universe)
    d = udir(Path(a.out), a.universe)
    t0 = time.process_time()
    logf = open(d / f"ira_train_beta{a.beta}.log", "w")

    def log(m: str) -> None:
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    lib, _records, norms, assign = train_library(u, IRAConfig(beta=a.beta), log=log)
    roles = {f"{w},{e}": int(make_world(u, "train", 100000 + w).roles[e]) for (w, e) in assign}
    name = "ira_library.json" if a.final else f"ira_library_beta{a.beta}.json"
    (d / name).write_text(json.dumps(lib.to_json()))
    (d / name.replace(".json", "_assign.json")).write_text(
        json.dumps({"assign": {f"{w},{e}": c for (w, e), c in sorted(assign.items())}, "true_roles": roles})
    )
    info = {
        "beta": a.beta,
        "train_norm_mean": float(np.mean(norms)),
        "cpu_s": time.process_time() - t0,
        "K": lib.k,
    }
    (d / name.replace(".json", "_info.json")).write_text(json.dumps(info, indent=1))
    log(json.dumps(info))


def cmd_train_metarl(a: argparse.Namespace) -> None:
    import torch

    from .metarl import MetaRLConfig, train_metarl

    if a.universe not in DEV_UNIVERSES:
        check_phase(a.universe, "confirm", Path(a.out))
    torch.set_num_threads(1)
    u = make_universe(a.universe)
    d = udir(Path(a.out), a.universe)
    cfg = MetaRLConfig(hidden=a.hidden, lr=a.lr, updates=a.updates, seed=a.seed)
    logf = open(d / f"metarl_{a.tag}.log", "w")

    def log(m: str) -> None:
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    t0 = time.process_time()
    net, info = train_metarl(u, cfg, log=log)
    info["cpu_s"] = time.process_time() - t0
    torch.save(net.state_dict(), d / f"metarl_{a.tag}.pt")
    (d / f"metarl_{a.tag}.json").write_text(json.dumps(info, indent=1))
    log(json.dumps(info))


def cmd_eval(a: argparse.Namespace) -> None:
    out = Path(a.out)
    check_phase(a.universe, a.phase, out)
    u = make_universe(a.universe)
    d = udir(out, a.universe) / a.phase
    d.mkdir(exist_ok=True)
    conds = a.conditions.split(",") if a.conditions else list(CONDITIONS[1:])
    for name in a.agents.split(","):
        cache: dict[str, Any] = {}
        for c in conds:
            path = d / f"{name}__{c}.jsonl"
            if path.exists() and a.resume:
                continue
            t0 = time.process_time()
            rows = []
            for s in SEEDS[a.phase]:
                ag = make_agent(name, a.universe, s, out, cache)
                m = play(ag, u, c, s)
                if isinstance(ag, IRA):
                    m["resets"] = ag.n_resets
                    m["w_post"] = (
                        [round(float(v), 4) for v in ag.w_posterior()] if ag.cfg.cue_track else [1.0]
                    )
                rows.append(m)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            cpu = time.process_time() - t0
            print(
                f"u{a.universe} {a.phase} {name:15s} {c:13s} norm {np.mean([r['norm'] for r in rows]):.3f} ({cpu:.0f}s)",
                flush=True,
            )


def cmd_accumulate(a: argparse.Namespace) -> None:
    """Sequential novel-role worlds: IRA consolidating additively after each world vs. the same library frozen."""
    out = Path(a.out)
    check_phase(a.universe, a.phase, out)
    u = make_universe(a.universe)
    d = udir(out, a.universe) / a.phase
    d.mkdir(exist_ok=True)
    st = load_settings(out)
    base = Library.from_json(json.loads((udir(out, a.universe) / "ira_library.json").read_text()))
    acc = base.copy()
    rows = []
    for i, s in enumerate(ACC_SEEDS[a.phase]):
        rng = np.random.default_rng([a.universe, s, 610])
        ag = IRA(acc, ira_cfg("ira", st["beta"]), rng)
        m_acc = play(ag, u, "novel_role", s)
        ag.consolidate(make_world(u, "novel_role", s).x)
        m_frz = play(
            IRA(base, ira_cfg("ira", st["beta"]), np.random.default_rng([a.universe, s, 610])),
            u,
            "novel_role",
            s,
        )
        rows.append(
            {"index": i, "seed": s, "accumulating": m_acc["norm"], "frozen": m_frz["norm"], "K": acc.k}
        )
    (d / "accumulation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    last = rows[len(rows) // 2 :]
    print(
        f"u{a.universe} accumulation: last-half accumulating {np.mean([r['accumulating'] for r in last]):.3f} "
        f"frozen {np.mean([r['frozen'] for r in last]):.3f}; K {base.k} -> {acc.k}"
    )


def cmd_freeze(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hashes": code_hash(),
        "settings": load_settings(out),
    }
    (out / "FREEZE.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="u1.experiment")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-ira")
    t.add_argument("--universe", type=int, required=True)
    t.add_argument("--beta", type=float, required=True)
    t.add_argument(
        "--final", action="store_true", help="write ira_library.json (the library used for evaluation)"
    )
    t.add_argument("--out", required=True)
    m = sub.add_parser("train-metarl")
    m.add_argument("--universe", type=int, required=True)
    m.add_argument("--hidden", type=int, default=128)
    m.add_argument("--lr", type=float, default=1e-3)
    m.add_argument("--updates", type=int, default=20000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--tag", default="s0")
    m.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--universe", type=int, required=True)
    e.add_argument("--phase", choices=["dev", "confirm"], required=True)
    e.add_argument("--agents", required=True)
    e.add_argument("--conditions", default="")
    e.add_argument("--resume", action="store_true")
    e.add_argument("--out", required=True)
    c = sub.add_parser("accumulate")
    c.add_argument("--universe", type=int, required=True)
    c.add_argument("--phase", choices=["dev", "confirm"], required=True)
    c.add_argument("--out", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--phase", choices=["dev", "confirm"], default="confirm")
    a = p.parse_args(argv)
    if a.cmd == "summarize":
        from .stats import summarize

        summarize(Path(a.out), a.phase)
        return
    {
        "train-ira": cmd_train_ira,
        "train-metarl": cmd_train_metarl,
        "eval": cmd_eval,
        "accumulate": cmd_accumulate,
        "freeze": cmd_freeze,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
