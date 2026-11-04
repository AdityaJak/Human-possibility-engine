"""Gate E development experiment (thesis §12): selection and report on development seeds only.

    python -m hpe.population.continual gate-e experiments/phase3/continual/<file>.yaml \
        --out reports/phase3/continual [--resume]

Seeds (all inside the development range 0-199; confirmatory seeds 3000-3049 are never generated):
``meta_train`` (F4 meta-training and the comparators' pretrained initialisation), ``selection``
(every learner's hyper-parameters and F4's checkpoint; criterion: mean expected reward), ``report``
(untouched by any selection; the development results). Every learner gets ≤ ``max_configs`` configurations
and ≤ ``budget_s_per_model`` CPU-seconds. Outputs: ``definition.json``, ``parts/<learner>.json``,
``outcomes.jsonl.gz`` (per seed and learner), ``summary.json``, ``report.md``, ``run_info.json``.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field
from scipy import stats

from hpe.core.rng import derive_seed, stream
from hpe.population.continual import learners as L
from hpe.population.continual import suite as S
from hpe.population.core.evaluate import holm

COMPARATORS = ("ft", "ewc", "replay")


class LearnerRun(BaseModel):
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)


class GateEDefinition(BaseModel):
    experiment_id: str
    title: str
    lane: str = "research/continual-transfer"
    hypothesis: str
    seed: int
    meta_train_seeds: tuple[int, int]
    selection_seeds: tuple[int, int]
    report_seeds: tuple[int, int]
    repeats_select: int = 2
    repeats_report: int = 5
    threads: int = 1
    max_configs: int = 24
    budget_s_per_model: float = 2700.0
    n_boot: int = 2000
    pretrain: dict[str, Any] = Field(default_factory=dict)
    learners: dict[str, LearnerRun]
    references: dict[str, LearnerRun] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def preregistration_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]

    def seeds(self, which: str) -> list[int]:
        lo, hi = getattr(self, f"{which}_seeds")
        out = list(range(int(lo), int(hi)))
        bad = [s for s in out if s not in S.DEV_SEEDS]
        if bad:
            raise S.ReservedSeedError(f"{which} seeds {bad[:3]}… are outside the development range 0-199")
        return out


def load_definition(path: Path) -> GateEDefinition:
    defn = GateEDefinition.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    sets = [set(defn.seeds(w)) for w in ("meta_train", "selection", "report")]
    if any(a & b for a, b in itertools.combinations(sets, 2)):
        raise ValueError("meta-train, selection and report seeds must be disjoint")
    return defn


def grid_configs(run: LearnerRun, cap: int) -> list[dict[str, Any]]:
    keys = sorted(run.grid)
    combos = [dict(zip(keys, v, strict=True)) for v in itertools.product(*(run.grid[k] for k in keys))] or [
        {}
    ]
    if len(combos) > cap:
        raise ValueError(f"grid has {len(combos)} configurations > cap {cap}")
    return [{**run.fixed, **c} for c in combos]


# ----------------------------------------------------------------------------------------- agents
def make_factory(name: str, cfg: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Return ``seq -> agent`` for learner/reference ``name`` with configuration ``cfg``."""
    if name in COMPARATORS or name == "ewc_oracle_boundaries":
        method = "ewc" if name == "ewc_oracle_boundaries" else name
        window, hidden = int(cfg.get("window", 10)), int(cfg.get("hidden", 32))
        init = ctx["pretrained"] if cfg.get("init", "scratch") == "pretrained" else ctx["scratch"]
        if init.W1.shape != (hidden, L.feature_dim(window)):
            raise ValueError("pretrained net shape differs from the learner configuration")
        kw = {
            k: cfg[k] for k in ("lr", "beta", "lam", "period", "gamma", "buffer", "replay_batch") if k in cfg
        }

        def mk(seq: S.SwitchingSequence) -> L.OnlineNetAgent:
            b = [blk[0] for blk in seq.blocks[1:]] if name == "ewc_oracle_boundaries" else None
            return L.OnlineNetAgent(method, init, window=window, boundaries=b, **kw)

        return mk
    if name == "bayes_filter":
        return lambda seq: S.BayesFilterAgent(
            int(cfg.get("n_particles", 256)), str(cfg.get("policy", "greedy")), float(cfg.get("floor", 0.0))
        )
    if name == "delta_rule":
        return lambda seq: S.DeltaRuleAgent(float(cfg.get("alpha", 0.3)), float(cfg.get("beta", 10.0)))
    if name == "oracle":
        return lambda seq: S.OracleAgent(seq)
    if name == "random":
        return lambda seq: S.RandomAgent()
    if name == "stationary_ts":
        return lambda seq: S.StationaryThompsonAgent()
    raise KeyError(name)


def _agg(per_seed: dict[int, dict[str, float]]) -> dict[str, float]:
    keys = sorted(next(iter(per_seed.values())))
    return {k: float(np.nanmean([v[k] for v in per_seed.values()])) for k in keys}


def select_learner(
    name: str,
    run: LearnerRun,
    defn: GateEDefinition,
    sel: list[S.SwitchingSequence],
    train: list[S.SwitchingSequence],
    ctx: dict[str, Any],
    log: Any,
) -> dict[str, Any]:
    configs = grid_configs(run, defn.max_configs)
    t0 = time.process_time()
    fits: list[dict[str, Any]] = []
    best: tuple[float, int] | None = None
    state: dict[str, Any] = {}
    for ci, cfg in enumerate(configs):
        spent = time.process_time() - t0
        if spent > defn.budget_s_per_model:
            fits.append({"config": cfg, "skipped": "budget exhausted"})
            continue
        c0 = time.process_time()
        if name == "f4":
            from hpe.population.continual import f4

            net, flog = f4.train_f4(
                train,
                sel,
                seed=derive_seed(defn.seed, "f4", ci) % (2**31 - 1),
                threads=defn.threads,
                budget_s=min(float(cfg.get("budget_s", 900.0)), defn.budget_s_per_model - spent),
                val_repeats=defn.repeats_select,
                log_fn=lambda r, ci=ci: log(f"  f4[{ci}] {json.dumps(r)}"),
                **{k: v for k, v in cfg.items() if k != "budget_s"},
            )
            score = float(flog["best_val_reward_fraction"])
            agg = {"reward_fraction": score}
            state[f"net{ci}"] = net
            extra = {"train_log": flog, "params": flog["params"]}
        else:
            per = S.evaluate_agent(
                make_factory(name, cfg, ctx),
                sel,
                seed=defn.seed,
                name=f"select-{name}-{ci}",
                repeats=defn.repeats_select,
            )
            agg = _agg(per)
            score = agg["reward_fraction"]
            extra = {}
        fits.append(
            {
                "config": cfg,
                "select": {k: round(v, 5) for k, v in agg.items()},
                "cpu_s": round(time.process_time() - c0, 2),
                **extra,
            }
        )
        log(f"[select] {name} cfg {ci} {cfg}: reward_fraction {score:.4f}")
        if best is None or score > best[0] + 1e-12:
            best = (score, ci)
    if best is None:
        raise RuntimeError(f"{name}: no configuration evaluated")
    ci = best[1]
    edges = [k for k, v in run.grid.items() if len(v) > 1 and configs[ci].get(k) in (v[0], v[-1])]
    rec = {
        "learner": name,
        "selected": {
            "config": configs[ci],
            "config_index": ci,
            "select_reward_fraction": best[0],
            "grid_edges": edges,
        },
        "n_configs_evaluated": sum(1 for f in fits if "skipped" not in f),
        "fits": fits,
        "select_cpu_s": round(time.process_time() - t0, 1),
    }
    if name == "f4":
        rec["model"] = state[f"net{ci}"]
    return rec


def _params(name: str, cfg: dict[str, Any], rec: dict[str, Any]) -> int | None:
    if name in (*COMPARATORS, "ewc_oracle_boundaries"):
        w, h = int(cfg.get("window", 10)), int(cfg.get("hidden", 32))
        return L.feature_dim(w) * h + h + h * S.K_OPTIONS + S.K_OPTIONS
    if name == "f4":
        return int(rec["fits"][rec["selected"]["config_index"]]["params"])
    return None


# ----------------------------------------------------------------------------------------- stats
def seed_contrast(
    a: dict[int, float], b: dict[int, float], *, seed: int, key: tuple[str, ...], n_boot: int
) -> dict[str, float | int]:
    """a − b over common seeds (positive favours a for reward metrics); percentile CI; one-sided p for a > b
    as max(bootstrap, paired t) (A3-1 rule)."""
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks], dtype=np.float64)
    d = d[np.isfinite(d)]
    rng = stream(seed, "gate_e_bootstrap", *key)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_boot = float(max(1.0 / n_boot, (means <= 0).mean()))
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    p_t = (
        float(stats.t.sf(d.mean() / (sd / np.sqrt(len(d))), df=len(d) - 1))
        if sd and np.isfinite(sd)
        else p_boot
    )
    return {
        "estimate": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_boot": p_boot,
        "p_t": p_t,
        "p": max(p_boot, p_t),
        "sd": sd,
        "n": len(d),
    }


def mean_ci(
    v: dict[int, float], *, seed: int, key: tuple[str, ...], n_boot: int
) -> tuple[float, float, float]:
    x = np.array([v[k] for k in sorted(v)], dtype=np.float64)
    x = x[np.isfinite(x)]
    rng = stream(seed, "gate_e_bootstrap_mean", *key)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def stop_rule(
    per: dict[str, dict[int, dict[str, float]]], cand: str, *, seed: int, n_boot: int
) -> dict[str, Any]:
    """Development analogue of the Gate E pass rule (thesis §12): savings > ft, ewc, replay (CI excludes 0,
    Holm over the three on max(bootstrap, t) p) and adaptation ≥ 0.9 × the best comparator's."""
    present = [c for c in COMPARATORS if c in per]
    sav = {
        c: seed_contrast(
            {s: v["savings"] for s, v in per[cand].items()},
            {s: v["savings"] for s, v in per[c].items()},
            seed=seed,
            key=("stop", cand, c, "savings"),
            n_boot=n_boot,
        )
        for c in present
    }
    adj = holm({c: float(v["p"]) for c, v in sav.items()})
    ad = {c: float(np.nanmean([v["adaptation"] for v in per[c].values()])) for c in [cand, *present]}
    best = max(present, key=lambda c: ad[c]) if present else None
    ratio = ad[cand] / ad[best] if best else float("nan")
    savings_ok = {c: bool(sav[c]["ci_low"] > 0 and adj[c] < 0.05) for c in present}
    return {
        "candidate": cand,
        "savings_contrasts": {c: {**sav[c], "p_holm": adj[c], "passes": savings_ok[c]} for c in present},
        "adaptation": ad,
        "best_comparator_adaptation": best,
        "adaptation_ratio": ratio,
        "adaptation_ok": bool(ratio >= 0.9),
        "missing_comparators": [c for c in COMPARATORS if c not in per],
        "passes_development_analogue": bool(
            len(present) == len(COMPARATORS) and all(savings_ok.values()) and ratio >= 0.9
        ),
    }


# ----------------------------------------------------------------------------------------- driver
def run(defn: GateEDefinition, out: Path, *, resume: bool = False) -> Path:
    outdir = Path(out) / defn.experiment_id
    parts = outdir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    logf = (outdir / "run.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    (outdir / "definition.json").write_text(
        json.dumps({**defn.model_dump(), "preregistration_hash": defn.preregistration_hash()}, indent=1)
        + "\n"
    )
    t_wall, t_cpu = time.time(), time.process_time()
    train = S.generate_many(defn.seeds("meta_train"))
    sel = S.generate_many(defn.seeds("selection"))
    rep = S.generate_many(defn.seeds("report"))
    log(f"seeds: meta-train {len(train)}, selection {len(sel)}, report {len(rep)} (all development)")
    pc = dict(defn.pretrain)
    c0 = time.process_time()
    pretrained, plog = L.pretrain_net(
        train,
        seed=derive_seed(defn.seed, "pretrain") % (2**31 - 1),
        window=int(pc.get("window", 10)),
        hidden=int(pc.get("hidden", 32)),
        epochs=int(pc.get("epochs", 20)),
        lr=float(pc.get("lr", 0.05)),
    )
    plog["cpu_s"] = round(time.process_time() - c0, 2)
    log(f"pretrained init: {plog}")
    ctx = {
        "pretrained": pretrained,
        "scratch": L.scratch_net(
            derive_seed(defn.seed, "scratch") % (2**31 - 1),
            int(pc.get("window", 10)),
            int(pc.get("hidden", 32)),
        ),
    }
    selected: dict[str, dict[str, Any]] = {}
    for name, lrun in [*defn.learners.items(), *defn.references.items()]:
        ppath = parts / f"{name}.json"
        if resume and ppath.exists() and name != "f4":
            selected[name] = json.loads(ppath.read_text())
            log(f"[resume] {name}: {selected[name]['selected']['config']}")
            continue
        if resume and ppath.exists() and name == "f4" and (parts / "f4_model.pt").exists():
            import torch

            from hpe.population.continual.f4 import F4Net

            rec = json.loads(ppath.read_text())
            cfg = rec["selected"]["config"]
            net = F4Net(S.K_OPTIONS, int(cfg.get("hidden", 48)))
            net.load_state_dict(torch.load(parts / "f4_model.pt", weights_only=True))
            net.eval()
            rec["model"] = net
            selected[name] = rec
            log(f"[resume] f4: {cfg}")
            continue
        rec = select_learner(name, lrun, defn, sel, train, ctx, log)
        if name == "f4":
            import torch

            torch.save(rec["model"].state_dict(), parts / "f4_model.pt")
        ppath.write_text(json.dumps({k: v for k, v in rec.items() if k != "model"}, indent=1) + "\n")
        selected[name] = rec
    # ---- report on untouched development seeds
    per: dict[str, dict[int, dict[str, float]]] = {}
    report_cpu: dict[str, float] = {}
    order = ["oracle", "random", "stationary_ts", *defn.references, *defn.learners]
    if "ewc" in selected:
        order.append("ewc_oracle_boundaries")
    for name in dict.fromkeys(order):
        c0 = time.process_time()
        if name == "f4":
            from hpe.population.continual import f4

            per[name] = f4.evaluate_f4(
                selected["f4"]["model"], rep, seed=defn.seed, repeats=defn.repeats_report, tag="report"
            )
        else:
            cfg = (
                selected["ewc" if name == "ewc_oracle_boundaries" else name]["selected"]["config"]
                if (name in selected or name == "ewc_oracle_boundaries")
                else {}
            )
            reps = 1 if name == "oracle" else defn.repeats_report
            per[name] = S.evaluate_agent(
                make_factory(name, cfg, ctx), rep, seed=defn.seed, name=f"report-{name}", repeats=reps
            )
        report_cpu[name] = round(time.process_time() - c0, 2)
        log(
            f"[report] {name}: "
            + json.dumps({k: round(v, 4) for k, v in _agg(per[name]).items() if k in S.METRICS})
        )
    summary = analyze(defn, per, selected, report_cpu, plog)
    with gzip.open(outdir / "outcomes.jsonl.gz", "wt", encoding="utf-8") as f:
        for name, by_seed in per.items():
            for sd, m in sorted(by_seed.items()):
                f.write(json.dumps({"learner": name, "seed": sd, **m}) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (outdir / "report.md").write_text(render(defn, summary))
    info = outdir / "run_info.json"
    hist = json.loads(info.read_text())["invocations"] if info.exists() else []
    from hpe.population.core.runner import code_version

    hist.append(
        {
            "code_version": code_version(),
            "hardware": f"{platform.machine()} python {platform.python_version()}",
            "threads": defn.threads,
            "wall_s": round(time.time() - t_wall, 1),
            "cpu_s": round(time.process_time() - t_cpu, 1),
            "llm_calls": 0,
            "seed_ranges": {
                w: list(getattr(defn, f"{w}_seeds")) for w in ("meta_train", "selection", "report")
            },
            "confirmatory_seeds_generated": 0,
        }
    )
    info.write_text(json.dumps({"invocations": hist}, indent=1) + "\n")
    logf.close()
    return outdir


def analyze(
    defn: GateEDefinition,
    per: dict[str, dict[int, dict[str, float]]],
    selected: dict[str, dict[str, Any]],
    report_cpu: dict[str, float],
    pretrain_log: dict[str, Any],
) -> dict[str, Any]:
    nb, sd = defn.n_boot, defn.seed
    learners: dict[str, Any] = {}
    for name, by_seed in per.items():
        row: dict[str, Any] = {}
        for m in (*S.METRICS, "reward_realised"):
            est, lo, hi = mean_ci({s: v[m] for s, v in by_seed.items()}, seed=sd, key=(name, m), n_boot=nb)
            row[m] = {"mean": est, "ci": [lo, hi]}
        rec = selected.get("ewc" if name == "ewc_oracle_boundaries" else name)
        row["config"] = rec["selected"]["config"] if rec else None
        row["params"] = _params(name, rec["selected"]["config"], rec) if rec else None
        row["select_cpu_s"] = rec["select_cpu_s"] if rec else None
        row["n_configs"] = rec["n_configs_evaluated"] if rec else None
        row["report_cpu_s"] = report_cpu.get(name)
        row["privileged"] = name in ("oracle", "ewc_oracle_boundaries")
        learners[name] = row
    contrasts = []
    ref = "bayes_filter" if "bayes_filter" in per else None
    for a in [*defn.learners, *([ref] if ref else [])]:
        for b in [*COMPARATORS, *([ref] if ref else []), "delta_rule"]:
            if a == b or a not in per or b not in per:
                continue
            if a in COMPARATORS and b in COMPARATORS and COMPARATORS.index(a) > COMPARATORS.index(b):
                continue
            for m in S.METRICS:
                c = seed_contrast(
                    {s: v[m] for s, v in per[a].items()},
                    {s: v[m] for s, v in per[b].items()},
                    seed=sd,
                    key=(a, b, m),
                    n_boot=nb,
                )
                contrasts.append({"first": a, "second": b, "metric": m, **c})
    summary: dict[str, Any] = {
        "experiment_id": defn.experiment_id,
        "partition": "development seeds (synthetic)",
        "report_seeds": list(defn.report_seeds),
        "learners": learners,
        "contrasts": contrasts,
        "pretrained_init": pretrain_log,
    }
    if "f4" in per:
        summary["stop_rule_development"] = stop_rule(per, "f4", seed=sd, n_boot=nb)
    return summary


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(defn: GateEDefinition, s: dict[str, Any]) -> str:
    out = [
        f"# {defn.experiment_id} — {defn.title}",
        "",
        "> **EXPLORATORY (development seeds of the synthetic Gate E suite).** Synthetic data; no human data; "
        "a synthetic result does not establish human validity. Confirmatory seeds 3000–3049 were not generated.",
        "",
        f"* Lane {defn.lane}; definition hash `{defn.preregistration_hash()}`; seed {defn.seed}; threads {defn.threads}",
        f"* Seeds: meta-train {list(defn.meta_train_seeds)}, selection {list(defn.selection_seeds)}, report "
        f"{list(defn.report_seeds)} (half-open ranges); {defn.repeats_report} rollouts per report seed",
        f"* Hypothesis: {defn.hypothesis}",
        "* Metrics: expected reward of the chosen option. adaptation = mean over switches of (10 trials after the "
        "switch) / oracle (0.8); savings = first 10 trials of a return − first 10 trials of that context's first "
        "block; forgetting = first 10 of a return − last 10 of that context's previous block (negative = loss).",
        "",
        "| learner | adaptation [95% CI] | savings [95% CI] | forgetting [95% CI] | reward / oracle [95% CI] | params | configs | select CPU s | report CPU s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(s["learners"].items(), key=lambda kv: -kv[1]["reward_fraction"]["mean"]):
        cells = [
            f"{_f(r[m]['mean'])} [{_f(r[m]['ci'][0])}, {_f(r[m]['ci'][1])}]"
            for m in ("adaptation", "savings", "forgetting", "reward_fraction")
        ]
        tag = f"{name} (privileged)" if r["privileged"] else name
        out.append(
            f"| {tag} | {' | '.join(cells)} | {r['params'] or '—'} | {r['n_configs'] or '—'} | "
            f"{r['select_cpu_s'] if r['select_cpu_s'] is not None else '—'} | {r['report_cpu_s']} |"
        )
    out += [
        "",
        "## Paired contrasts over report seeds (first − second; positive favours first)",
        "",
        "| first | second | metric | estimate | 95% CI | p (max boot, t; H1 first > second) | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in s["contrasts"]:
        out.append(
            f"| {c['first']} | {c['second']} | {c['metric']} | {c['estimate']:+.4f} | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p']:.4f} | {c['n']} |"
        )
    st = s.get("stop_rule_development")
    if st:
        out += [
            "",
            "## Development analogue of the Gate E rule (F4 vs ft, ewc, replay)",
            "",
            "| comparator | savings difference | 95% CI | Holm p | savings condition |",
            "|---|---|---|---|---|",
        ]
        for c, v in st["savings_contrasts"].items():
            out.append(
                f"| {c} | {v['estimate']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {v['p_holm']:.4f} | {v['passes']} |"
            )
        out += [
            "",
            f"Adaptation: F4 {_f(st['adaptation']['f4'])} vs best comparator `{st['best_comparator_adaptation']}` "
            f"{_f(st['adaptation'].get(st['best_comparator_adaptation'] or '', float('nan')))}; ratio "
            f"{_f(st['adaptation_ratio'])} (needs ≥ 0.9): {st['adaptation_ok']}. "
            f"**Development analogue passes: {st['passes_development_analogue']}** (exploratory; cannot pass Gate E).",
        ]
    out += ["", f"Pretrained initialisation of the comparators: {json.dumps(s['pretrained_init'])}", ""]
    return "\n".join(out) + "\n"
