"""Phase III experiment runner: fit on dev, select (config, lapse) on tune, report on tune or (after a
freeze) test. Resumable: each (family, model) result is checkpointed under ``<out>/parts/``.

    python -m hpe.population run experiments/phase3/<file>.yaml --out reports/phase3/<lane> [--resume]
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from hpe.core.rng import derive_seed, stream
from hpe.population.core import evaluate
from hpe.population.core.conformance import check_causal
from hpe.population.core.data import load_family
from hpe.population.core.encoding import EncodedSeq, input_dim
from hpe.population.core.families import get_family
from hpe.population.core.interface import SequencePredictor, get_model


class ModelRun(BaseModel):
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)
    role: Literal["comparator", "candidate", "reference"] = "comparator"
    variants: list[Literal["frozen", "mismatched"]] = Field(default_factory=list)
    families: list[str] | None = None  # restrict to these families (default: all)


class P3Definition(BaseModel):
    experiment_id: str
    title: str
    lane: str
    hypothesis: str
    families: list[str]
    fit_partition: Literal["dev"] = "dev"
    select_partition: Literal["tune"] = "tune"
    report_partition: Literal["tune", "test"] = "tune"
    seed: int
    threads: int = 2
    k_max: int = 4
    lapse_grid: list[float] = Field(default_factory=lambda: list(evaluate.LAPSE_GRID))
    max_configs: int = 24
    budget_s_per_model_family: float = 2700.0
    models: dict[str, ModelRun]
    notes: list[str] = Field(default_factory=list)

    def preregistration_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


def load_definition(path: Path) -> P3Definition:
    return P3Definition.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def grid_configs(run: ModelRun, max_configs: int) -> list[dict[str, Any]]:
    keys = sorted(run.grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(run.grid[k] for k in keys))]
    combos = combos or [{}]
    if len(combos) > max_configs:
        raise ValueError(f"grid has {len(combos)} configurations > cap {max_configs}")
    return [{**run.fixed, **c} for c in combos]


def code_version() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def _scores_json(scores: list[evaluate.ParticipantScore]) -> list[dict[str, Any]]:
    return [
        {"key": s.key, "nll": s.nll, "n": s.n, "top1": s.top1, "ece": s.ece, "quarters": list(s.quarters)}
        for s in scores
    ]


def _scores_from_json(rows: list[dict[str, Any]]) -> list[evaluate.ParticipantScore]:
    return [
        evaluate.ParticipantScore(r["key"], r["nll"], r["n"], r["top1"], r["ece"], tuple(r["quarters"]))  # type: ignore[arg-type]
        for r in rows
    ]


def donors_for(seqs: list[EncodedSeq], seed: int, family: str) -> dict[str, EncodedSeq]:
    """Deterministic derangement: each participant gets another participant of the same family/partition."""
    if len(seqs) < 2:
        return {}
    rng = stream(seed, "phase3_donors", family)
    order = [int(i) for i in rng.permutation(len(seqs))]
    return {seqs[order[i]].key: seqs[order[(i + 1) % len(order)]] for i in range(len(order))}


def run_model_family(
    defn: P3Definition, family: str, name: str, run: ModelRun, data: dict[str, list[EncodedSeq]]
) -> dict[str, Any]:
    spec = get_model(name)
    d = input_dim(defn.k_max)
    configs = grid_configs(run, defn.max_configs)
    fits: list[dict[str, Any]] = []
    best: tuple[float, int, float, SequencePredictor] | None = None
    t_family = time.process_time()
    for ci, cfg in enumerate(configs):
        spent = time.process_time() - t_family
        if spent > defn.budget_s_per_model_family:
            fits.append({"config": cfg, "skipped": "budget exhausted"})
            continue
        model = spec.factory(cfg, defn.k_max, d)
        seed = derive_seed(defn.seed, family, name, ci) % (2**31 - 1)
        t0, w0 = time.process_time(), time.time()
        log = model.fit(
            data[defn.fit_partition],
            seed=seed,
            threads=defn.threads,
            budget_s=max(60.0, defn.budget_s_per_model_family - spent),
        )
        fit_cpu, fit_wall = time.process_time() - t0, time.time() - w0
        sel_seqs = data[defn.select_partition]
        lps = [model.predict_logprobs(s) for s in sel_seqs]
        by_lapse: dict[str, float] = {}
        for lam in defn.lapse_grid:
            sc = [evaluate.score(s, lp, lam) for s, lp in zip(sel_seqs, lps, strict=True)]
            by_lapse[str(lam)] = evaluate.mean_nll(sc)
        lam_best = min(defn.lapse_grid, key=lambda lam: (by_lapse[str(lam)], lam))
        sel = by_lapse[str(lam_best)]
        fits.append(
            {
                "config": cfg,
                "select_nll_by_lapse": by_lapse,
                "lapse": lam_best,
                "select_nll": sel,
                "fit_cpu_s": round(fit_cpu, 2),
                "fit_wall_s": round(fit_wall, 2),
                "fit_log": log,
                "params": model.param_count(),
            }
        )
        if best is None or sel < best[0] - 1e-12:
            best = (sel, ci, lam_best, model)
    if best is None:
        raise RuntimeError(f"{family}/{name}: no configuration fitted within budget")
    sel_nll, ci, lam, model = best
    causal = check_causal(model, data[defn.select_partition][0], seed=defn.seed)
    rep = data[defn.report_partition]
    t0 = time.process_time()
    variants: dict[str, list[dict[str, Any]]] = {}
    variants["main"] = _scores_json([evaluate.score(s, model.predict_logprobs(s), lam) for s in rep])
    adaptive = bool(spec.adaptive or getattr(model, "adaptive", False))
    if adaptive and "frozen" in run.variants:
        variants["frozen"] = _scores_json(
            [evaluate.score(s, model.predict_logprobs(s, mode="frozen"), lam) for s in rep]
        )
    if adaptive and "mismatched" in run.variants:
        donors = donors_for(rep, defn.seed, family)
        variants["mismatched"] = _scores_json(
            [
                evaluate.score(s, model.predict_logprobs(s, donor=donors[s.key]), lam)
                for s in rep
                if s.key in donors
            ]
        )
    edges = [k for k, vals in run.grid.items() if len(vals) > 1 and configs[ci].get(k) in (vals[0], vals[-1])]
    return {
        "family": family,
        "model": name,
        "role": run.role,
        "owner": spec.owner,
        "adaptive": adaptive,
        "selected": {
            "config": configs[ci],
            "config_index": ci,
            "lapse": lam,
            "select_nll": sel_nll,
            "grid_edges": edges,
        },
        "describe": model.describe(),
        "params": model.param_count(),
        "fits": fits,
        "causality_violations": causal,
        "report_partition": defn.report_partition,
        "report_cpu_s": round(time.process_time() - t0, 2),
        "variants": variants,
    }


def run(
    defn: P3Definition, out: Path, *, resume: bool = False, only_families: list[str] | None = None
) -> Path:
    outdir = Path(out) / defn.experiment_id
    parts_dir = outdir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    (outdir / "definition.json").write_text(
        json.dumps({**defn.model_dump(), "preregistration_hash": defn.preregistration_hash()}, indent=1)
        + "\n"
    )
    t_wall, t_cpu = time.time(), time.process_time()
    exclusions: dict[str, dict[str, list[str]]] = {}
    for family in defn.families:
        if only_families and family not in only_families:
            continue
        get_family(family)
        todo = [
            (n, r)
            for n, r in defn.models.items()
            if (r.families is None or family in r.families)
            and not (resume and (parts_dir / f"{family}__{n}.json").exists())
        ]
        if not todo:
            print(f"[{family}] all models checkpointed; skipping", flush=True)
            continue
        data: dict[str, list[EncodedSeq]] = {}
        for part in sorted({defn.fit_partition, defn.select_partition, defn.report_partition}):
            seqs, excl = load_family(family, part, experiment_id=defn.experiment_id, k_max=defn.k_max)
            data[part] = seqs
            exclusions.setdefault(family, {})[part] = excl
        for name, mrun in todo:
            n_cfg = len(grid_configs(mrun, defn.max_configs))
            print(f"[{family}] {name}: {n_cfg} config(s), {len(data['dev'])} dev participants", flush=True)
            res = run_model_family(defn, family, name, mrun, data)
            (parts_dir / f"{family}__{name}.json").write_text(json.dumps(res) + "\n")
            print(
                f"[{family}] {name}: select NLL {res['selected']['select_nll']:.4f} "
                f"config {res['selected']['config']} lapse {res['selected']['lapse']} "
                f"causal={'ok' if not res['causality_violations'] else 'VIOLATION'}",
                flush=True,
            )
    info_path = outdir / "run_info.json"
    history = json.loads(info_path.read_text())["invocations"] if info_path.exists() else []
    history.append(
        {
            "code_version": code_version(),
            "hardware": f"{platform.machine()} python {platform.python_version()}",
            "threads": defn.threads,
            "families": only_families or defn.families,
            "wall_s": round(time.time() - t_wall, 1),
            "cpu_s": round(time.process_time() - t_cpu, 1),
            "llm_calls": 0,
            "exclusions": exclusions,
        }
    )
    info_path.write_text(json.dumps({"invocations": history}, indent=1) + "\n")
    analyze(outdir)
    return outdir


# ----------------------------------------------------------------------------- analysis
def _load_parts(outdir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    res: dict[str, dict[str, dict[str, Any]]] = {}
    for p in sorted((outdir / "parts").glob("*.json")):
        r = json.loads(p.read_text())
        res.setdefault(r["family"], {})[r["model"]] = r
    return res


def analyze(outdir: Path) -> dict[str, Any]:
    raw = json.loads((outdir / "definition.json").read_text())
    defn = P3Definition.model_validate({k: v for k, v in raw.items() if k != "preregistration_hash"})
    parts = _load_parts(outdir)
    summary: dict[str, Any] = {
        "experiment_id": defn.experiment_id,
        "report_partition": defn.report_partition,
        "families": {},
    }
    rows = []
    best_vs: dict[str, dict[str, Any]] = {}
    for family, models in parts.items():
        fam: dict[str, Any] = {"role": get_family(family).role, "models": {}, "contrasts": []}
        scores: dict[str, list[evaluate.ParticipantScore]] = {}
        for name, r in models.items():
            for var, sc in r["variants"].items():
                label = name if var == "main" else f"{name}[{var}]"
                s = _scores_from_json(sc)
                scores[label] = s
                nll = np.array([x.nll for x in s])
                m, lo, hi, _ = evaluate.bootstrap(nll, seed=defn.seed, key=(family, label, "mean"))
                fam["models"][label] = {
                    "report_nll": m,
                    "ci": [lo, hi],
                    "n": len(s),
                    "top1": float(np.mean([x.top1 for x in s])),
                    "ece": float(np.mean([x.ece for x in s])),
                    "quarters": [float(np.nanmean([x.quarters[q] for x in s])) for q in range(4)],
                    "select_nll": r["selected"]["select_nll"],
                    "params": r["params"],
                    "role": r["role"],
                    "config": r["selected"]["config"],
                    "lapse": r["selected"]["lapse"],
                    "grid_edges": r["selected"]["grid_edges"],
                    "fit_cpu_s": round(sum(f.get("fit_cpu_s", 0.0) for f in r["fits"]), 1),
                    "causality_violations": r["causality_violations"],
                    "describe": r.get("describe", {}),
                }
                for x in s:
                    rows.append(
                        {
                            "family": family,
                            "model": label,
                            "key": x.key,
                            "nll": x.nll,
                            "n": x.n,
                            "top1": x.top1,
                        }
                    )
        comps = [n for n, r in models.items() if r["role"] == "comparator"]
        cands = [n for n, r in models.items() if r["role"] == "candidate"]
        best_comp = min(comps, key=lambda n: models[n]["selected"]["select_nll"]) if comps else None
        fam["best_comparator"] = best_comp
        for c in cands:
            targets: list[tuple[str, str, str]] = []
            if best_comp:
                targets.append((c, best_comp, "vs_best_comparator"))
            for v in ("frozen", "mismatched"):
                if f"{c}[{v}]" in scores:
                    targets.append((c, f"{c}[{v}]", f"vs_{v}"))
            targets += [(c, comp, "vs_comparator") for comp in comps if comp != best_comp]
            for a, b, kind in targets:
                con = evaluate.paired(scores[a], scores[b], seed=defn.seed, key=(family, a, b))
                fam["contrasts"].append({"first": a, "second": b, "kind": kind, **con})
                if kind == "vs_best_comparator":
                    best_vs.setdefault(c, {})[family] = con
        summary["families"][family] = fam
    for c, by_fam in best_vs.items():
        counted = {f: v for f, v in by_fam.items() if get_family(f).role.startswith("counted")}
        adj = evaluate.holm({f: float(v["p_one_sided"]) for f, v in counted.items()}) if counted else {}
        summary.setdefault("candidate_vs_best_comparator", {})[c] = {
            f: {
                **v,
                "p_holm": adj.get(f),
                "meets_0.02": bool(f in adj and evaluate.passes(v, 0.02, adj[f])),
            }
            for f, v in by_fam.items()
        }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with gzip.open(outdir / "outcomes.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (outdir / "report.md").write_text(render(defn, summary))
    return summary


def _f(x: Any, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def render(defn: P3Definition, summary: dict[str, Any]) -> str:
    tag = (
        "CONFIRMATORY (test partition, frozen)"
        if defn.report_partition == "test"
        else "EXPLORATORY (dev/tune; selection-optimistic)"
    )
    out = [
        f"# {defn.experiment_id} — {defn.title}",
        "",
        f"> **{tag}.** Real human data (Psych-101, Apache-2.0). Held-out participants; teacher-forced NLL "
        "per free choice (nats). Not evidence about any individual's future.",
        "",
        f"* Lane: {defn.lane}; preregistration hash `{defn.preregistration_hash()}`; fit {defn.fit_partition} / "
        f"select {defn.select_partition} / report {defn.report_partition}; seed {defn.seed}; threads {defn.threads}",
        f"* Hypothesis: {defn.hypothesis}",
        "",
    ]
    for family, fam in summary["families"].items():
        out += [
            f"## {family} ({fam['role']}); best comparator on {defn.select_partition}: `{fam['best_comparator']}`",
            "",
        ]
        out += [
            "| model | role | report NLL [95% CI] | n | top-1 | ECE | NLL by quarter | params | lapse | grid edges | fit CPU s | causal |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, m in sorted(fam["models"].items(), key=lambda kv: kv[1]["report_nll"]):
            q = " / ".join(_f(v, 3) for v in m["quarters"])
            out.append(
                f"| {name} | {m['role']} | {_f(m['report_nll'])} [{_f(m['ci'][0])}, {_f(m['ci'][1])}] | {m['n']} | "
                f"{_f(m['top1'], 3)} | {_f(m['ece'], 3)} | {q} | {m['params']} | {m['lapse']} | "
                f"{', '.join(m['grid_edges']) or '—'} | {m['fit_cpu_s']} | "
                f"{'ok' if not m['causality_violations'] else 'VIOLATION'} |"
            )
        if fam["contrasts"]:
            out += [
                "",
                "| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | sd | n |",
                "|---|---|---|---|---|---|---|",
            ]
            for c in fam["contrasts"]:
                out.append(
                    f"| {c['first']} − {c['second']} | {c['kind']} | {c['estimate']:+.4f} | "
                    f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_one_sided']:.4f} | {_f(c['sd'])} | {c['n']} |"
                )
        out.append("")
    if summary.get("candidate_vs_best_comparator"):
        out += [
            "## Candidate vs best comparator across counted families (Holm)",
            "",
            "| candidate | family | estimate | 95% CI | Holm p | meets 0.02 margin |",
            "|---|---|---|---|---|---|",
        ]
        for c, by in summary["candidate_vs_best_comparator"].items():
            for f, v in by.items():
                out.append(
                    f"| {c} | {f} | {v['estimate']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | "
                    f"{_f(v['p_holm'])} | {v['meets_0.02']} |"
                )
        out.append("")
    if defn.report_partition != "test":
        out.append(
            "_Exploratory: the report partition is also the selection partition. No gate is evaluated here._"
        )
    return "\n".join(out) + "\n"
