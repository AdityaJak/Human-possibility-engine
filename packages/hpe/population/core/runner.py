"""Phase III experiment runner: fit on dev, select (config, lapse) on tune, report on tune or (after a
freeze) test. Resumable: each (family, model) result is checkpointed under ``<out>/parts/``.

    python -m hpe.population run experiments/phase3/<file>.yaml --out reports/phase3/<lane> [--resume]
"""

from __future__ import annotations

import copy
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
from hpe.population.core.adaptation import overrides_harness
from hpe.population.core.conformance import check_causal, check_encoding_leak, state_digest
from hpe.population.core.data import load_family
from hpe.population.core.encoding import EncodedSeq, input_dim, rekey
from hpe.population.core.families import get_family
from hpe.population.core.interface import get_model


class ModelRun(BaseModel):
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)
    role: Literal["comparator", "candidate", "reference"] = "comparator"
    variants: list[Literal["frozen", "mismatched", "twin"]] = Field(default_factory=list)
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


def _token(seed: int, *key: str) -> str:
    return f"p{derive_seed(seed, 'phase3_rekey', *key) % (1 << 60):x}"


def predict_isolated(
    model: Any, seq: EncodedSeq, token: str, *, mode: str = "adapt", donor: EncodedSeq | None = None
) -> np.ndarray:
    """Predict one participant with a fresh copy of the model and an opaque key (critic findings L-2, L-3)."""
    m = copy.deepcopy(model)
    d = rekey(donor, token + "-donor") if donor is not None else None
    return np.asarray(m.predict_logprobs(rekey(seq, token), mode=mode, donor=d))


def score_variant(
    model: Any, seqs: list[EncodedSeq], lam: float, seed: int, tag: str, **kw: Any
) -> tuple[list[evaluate.ParticipantScore], list[evaluate.ParticipantScore]]:
    """Scores at the selected lapse and at lapse 0 (critic finding S-6)."""
    main, zero = [], []
    for s in seqs:
        lp = predict_isolated(model, s, _token(seed, tag, s.key), **kw)
        main.append(evaluate.score(s, lp, lam))
        zero.append(evaluate.score(s, lp, 0.0))
    return main, zero


def _fit_one(
    spec: Any, cfg: dict[str, Any], defn: P3Definition, train: list[EncodedSeq], seed: int, budget: float
) -> tuple[Any, dict[str, Any], float, float]:
    model = spec.factory(cfg, defn.k_max, input_dim(defn.k_max))
    t0, w0 = time.process_time(), time.time()
    log = model.fit(train, seed=seed, threads=defn.threads, budget_s=budget)
    return model, log, time.process_time() - t0, time.time() - w0


def select_model_family(
    defn: P3Definition, family: str, name: str, run: ModelRun, data: dict[str, list[EncodedSeq]]
) -> dict[str, Any]:
    """Fit every configuration on dev, select (config, lapse) on tune, run the strict checks, fit the twin."""
    spec = get_model(name)
    configs = grid_configs(run, defn.max_configs)
    fits: list[dict[str, Any]] = []
    best: tuple[float, int, float, Any] | None = None
    t_family = time.process_time()
    sel_seqs = data[defn.select_partition]
    for ci, cfg in enumerate(configs):
        spent = time.process_time() - t_family
        if spent > defn.budget_s_per_model_family:
            fits.append({"config": cfg, "skipped": "budget exhausted"})
            continue
        seed = derive_seed(defn.seed, family, name, ci) % (2**31 - 1)
        model, log, fit_cpu, fit_wall = _fit_one(
            spec, cfg, defn, data[defn.fit_partition], seed, max(60.0, defn.budget_s_per_model_family - spent)
        )
        lps = [
            predict_isolated(model, s, _token(defn.seed, "select", name, str(ci), s.key)) for s in sel_seqs
        ]
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
                "nondeterministic_budget_stop": (log or {}).get("stopped") == "budget",
                "params": model.param_count(),
            }
        )
        if best is None or sel < best[0] - 1e-12:
            best = (sel, ci, lam_best, model)
    if best is None:
        raise RuntimeError(f"{family}/{name}: no configuration fitted within budget")
    sel_nll, ci, lam, model = best
    violations: list[str] = []
    if overrides_harness(model):
        violations.append("adaptive model overrides the harness-owned predict_logprobs")
    t_check = time.process_time()
    violations += check_causal(model, sel_seqs[:20], seed=defn.seed)
    check_cpu = time.process_time() - t_check
    adaptive = bool(spec.adaptive or getattr(model, "adaptive", False))
    twin, twin_info = None, None
    tcfg = model.twin_config() if hasattr(model, "twin_config") else None
    if adaptive and "twin" in run.variants and tcfg:
        tspec = get_model("pop_gru")
        tseed = derive_seed(defn.seed, family, name, ci, "twin") % (2**31 - 1)
        twin, tlog, tcpu, _tw = _fit_one(
            tspec, dict(tcfg), defn, data[defn.fit_partition], tseed, defn.budget_s_per_model_family
        )
        tl = [predict_isolated(twin, s, _token(defn.seed, "twin-select", name, s.key)) for s in sel_seqs]
        tby = {
            str(lam_): evaluate.mean_nll(
                [evaluate.score(s, lp, lam_) for s, lp in zip(sel_seqs, tl, strict=True)]
            )
            for lam_ in defn.lapse_grid
        }
        tlam = min(defn.lapse_grid, key=lambda v: (tby[str(v)], v))
        twin_info = {
            "config": tcfg,
            "lapse": tlam,
            "select_nll": tby[str(tlam)],
            "fit_cpu_s": round(tcpu, 2),
            "fit_log": tlog,
            "params": twin.param_count(),
            "causality_violations": check_causal(twin, sel_seqs[:5], seed=defn.seed),
        }
    edges = [k for k, vals in run.grid.items() if len(vals) > 1 and configs[ci].get(k) in (vals[0], vals[-1])]
    record = {
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
            "n_configs_evaluated": sum(1 for f in fits if "skipped" not in f),
        },
        "describe": model.describe(),
        "params": model.param_count(),
        "fits": fits,
        "causality_violations": violations,
        "causality_check_cpu_s": round(check_cpu, 2),
        "twin": twin_info,
    }
    return {"record": record, "model": model, "twin": twin}


def report_model_family(
    defn: P3Definition, family: str, name: str, run: ModelRun, sel: dict[str, Any], rep: list[EncodedSeq]
) -> dict[str, Any]:
    """Score the selected model (and its twin / harness variants) on the report partition."""
    record = dict(sel["record"])
    model, twin = sel["model"], sel["twin"]
    lam = record["selected"]["lapse"]
    violations = list(record["causality_violations"])
    digest_before = state_digest(model)
    t0 = time.process_time()
    variants: dict[str, list[dict[str, Any]]] = {}
    lapse0: dict[str, float] = {}
    main, zero = score_variant(model, rep, lam, defn.seed, f"report-{name}")
    variants["main"] = _scores_json(main)
    lapse0["main"] = evaluate.mean_nll(zero)
    report_cpu = time.process_time() - t0
    if record["adaptive"] and "frozen" in run.variants:
        sc, zero = score_variant(model, rep, lam, defn.seed, f"frozen-{name}", mode="frozen")
        variants["frozen"], lapse0["frozen"] = _scores_json(sc), evaluate.mean_nll(zero)
    if record["adaptive"] and "mismatched" in run.variants:
        donors = donors_for(rep, defn.seed, family)
        sc = []
        for s in rep:
            if s.key in donors:
                lp = predict_isolated(
                    model, s, _token(defn.seed, f"donor-{name}", s.key), donor=donors[s.key]
                )
                sc.append(evaluate.score(s, lp, lam))
        variants["mismatched"] = _scores_json(sc)
    if twin is not None and record.get("twin"):
        sc, zero = score_variant(twin, rep, record["twin"]["lapse"], defn.seed, f"twin-{name}")
        variants["twin"], lapse0["twin"] = _scores_json(sc), evaluate.mean_nll(zero)
    if state_digest(model) != digest_before:
        violations.append("model state changed while predicting the report partition")
    n_trials = sum(s.T for s in rep)
    record.update(
        {
            "causality_violations": violations,
            "report_partition": defn.report_partition,
            "report_cpu_s": round(report_cpu, 2),
            "predict_ms_per_trial": round(1000.0 * report_cpu / max(1, n_trials), 4),
            "lapse0_nll": lapse0,
            "variants": variants,
        }
    )
    return record


def run_model_family(
    defn: P3Definition, family: str, name: str, run: ModelRun, data: dict[str, list[EncodedSeq]]
) -> dict[str, Any]:
    sel = select_model_family(defn, family, name, run, data)
    return report_model_family(defn, family, name, run, sel, data[defn.report_partition])


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_selection(outdir: Path, family: str, name: str, sel: dict[str, Any]) -> dict[str, str]:
    """Stage 'fit' of a confirmatory run: persist the selected model (and twin) and their SHA-256 digests."""
    import torch  # research extra; confirmatory runs need it anyway

    mdir = outdir / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for tag, obj in (("model", sel["model"]), ("twin", sel["twin"])):
        if obj is None:
            continue
        path = mdir / f"{family}__{name}__{tag}.pt"
        torch.save(obj, path)
        digests[f"{family}__{name}__{tag}"] = _sha256_file(path)
    (outdir / "parts" / f"{family}__{name}.selection.json").write_text(json.dumps(sel["record"]) + "\n")
    return digests


def load_selection(outdir: Path, family: str, name: str, expected: dict[str, str]) -> dict[str, Any]:
    """Stage 'test': load the frozen models, refusing any whose digest differs from the freeze record."""
    import torch

    record = json.loads((outdir / "parts" / f"{family}__{name}.selection.json").read_text())
    objs: dict[str, Any] = {"model": None, "twin": None}
    for tag in ("model", "twin"):
        key = f"{family}__{name}__{tag}"
        path = outdir / "models" / f"{key}.pt"
        if not path.exists():
            continue
        if expected.get(key) != _sha256_file(path):
            raise RuntimeError(f"{key}: model file digest differs from the freeze record")
        objs[tag] = torch.load(path, weights_only=False)
    return {"record": record, **objs}


def run(
    defn: P3Definition,
    out: Path,
    *,
    resume: bool = False,
    only_families: list[str] | None = None,
    stage: Literal["all", "fit", "test"] = "all",
) -> Path:
    """Exploratory runs use ``stage='all'`` (report on tune). A confirmatory definition (report on test) runs
    in two processes: ``stage='fit'`` (dev/tune only; persists the selected models and their digests, to be
    committed and named in the freeze record) and later ``stage='test'`` (validates the freeze, opens the
    single-run access log, loads the frozen models by digest, scores test). Critic finding T-1."""
    if defn.report_partition == "test" and stage == "all":
        raise ValueError("a confirmatory definition must run as stage 'fit' and then stage 'test'")
    if stage != "all" and defn.report_partition != "test":
        raise ValueError("stages are only for confirmatory (test) definitions")
    outdir = Path(out) / defn.experiment_id
    parts_dir = outdir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if stage == "test":
        return _run_test_stage(defn, outdir)
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
        parts_needed = {defn.fit_partition, defn.select_partition}
        if stage == "all":
            parts_needed.add(defn.report_partition)
        for part in sorted(parts_needed):
            seqs, excl = load_family(family, part, experiment_id=defn.experiment_id, k_max=defn.k_max)
            data[part] = seqs
            exclusions.setdefault(family, {})[part] = excl
        leak = check_encoding_leak(data[defn.fit_partition])
        if leak:
            print(f"[{family}] ENCODING LEAK SUSPECTED: {leak}", flush=True)
        exclusions.setdefault(family, {})["encoding_leak_check"] = leak
        for name, mrun in todo:
            n_cfg = len(grid_configs(mrun, defn.max_configs))
            print(f"[{family}] {name}: {n_cfg} config(s), {len(data['dev'])} dev participants", flush=True)
            if stage == "fit":
                sel = select_model_family(defn, family, name, mrun, data)
                digests = save_selection(outdir, family, name, sel)
                dpath = outdir / "models" / "digests.json"
                alld = json.loads(dpath.read_text()) if dpath.exists() else {}
                alld.update(digests)
                dpath.write_text(json.dumps(alld, indent=1, sort_keys=True) + "\n")
                res = sel["record"]
            else:
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
    if stage == "all":
        analyze(outdir)
    return outdir


def _run_test_stage(defn: P3Definition, outdir: Path) -> Path:
    from hpe.population.core.freeze import open_access_log, validate_freeze

    h = defn.preregistration_hash()
    rec = validate_freeze(defn.experiment_id, definition_hash=h)
    open_access_log(defn.experiment_id, rec, "confirmatory test-stage run")
    t_wall, t_cpu = time.time(), time.process_time()
    exclusions: dict[str, list[str]] = {}
    for family in defn.families:
        rep, excl = load_family(
            family, "test", experiment_id=defn.experiment_id, k_max=defn.k_max, definition_hash=h
        )
        exclusions[family] = excl
        for name, mrun in defn.models.items():
            if mrun.families is not None and family not in mrun.families:
                continue
            sel = load_selection(outdir, family, name, rec["model_digests"])
            res = report_model_family(defn, family, name, mrun, sel, rep)
            (outdir / "parts" / f"{family}__{name}.json").write_text(json.dumps(res) + "\n")
            print(f"[{family}] {name}: test scored ({len(rep)} participants)", flush=True)
    (outdir / "test_run_info.json").write_text(
        json.dumps(
            {
                "code_version": code_version(),
                "wall_s": round(time.time() - t_wall, 1),
                "cpu_s": round(time.process_time() - t_cpu, 1),
                "exclusions": exclusions,
                "llm_calls": 0,
            },
            indent=1,
        )
        + "\n"
    )
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
                    "select_nll": r["selected"]["select_nll"]
                    if var == "main"
                    else ((r.get("twin") or {}).get("select_nll") if var == "twin" else None),
                    "lapse0_nll": (r.get("lapse0_nll") or {}).get(var),
                    "params": r["params"] if var != "twin" else (r.get("twin") or {}).get("params"),
                    "role": r["role"],
                    "config": r["selected"]["config"],
                    "lapse": r["selected"]["lapse"],
                    "grid_edges": r["selected"]["grid_edges"],
                    "fit_cpu_s": round(sum(f.get("fit_cpu_s", 0.0) for f in r["fits"]), 1),
                    "causality_violations": r["causality_violations"]
                    if var != "twin"
                    else (r.get("twin") or {}).get("causality_violations", []),
                    "excluded": bool(r["causality_violations"]) if var != "twin" else False,
                    "predict_ms_per_trial": r.get("predict_ms_per_trial"),
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
        # models with any causality/state violation are excluded from every comparison (critic L-1)
        comps = [n for n, r in models.items() if r["role"] == "comparator" and not r["causality_violations"]]
        cands = [n for n, r in models.items() if r["role"] == "candidate" and not r["causality_violations"]]
        fam["excluded_models"] = sorted(n for n, r in models.items() if r["causality_violations"])
        best_comp = min(comps, key=lambda n: models[n]["selected"]["select_nll"]) if comps else None
        fam["best_comparator"] = best_comp
        for c in cands:
            targets: list[tuple[str, str, str]] = []
            if best_comp:
                targets.append((c, best_comp, "vs_best_comparator"))
            for v in ("twin", "frozen", "mismatched"):
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
