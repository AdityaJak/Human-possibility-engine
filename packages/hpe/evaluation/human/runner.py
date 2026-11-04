"""End-to-end runners for exp07 (held-out human choices) and exp07b (HeartSteps causal grounding)."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.contracts.schemas import OutcomeObservation
from hpe.evaluation.human import causal, data, heartsteps, psych101
from hpe.evaluation.human.definition import CausalGroundingDefinition, HumanHeldoutDefinition
from hpe.evaluation.human.harness import evaluate_paradigm
from hpe.evaluation.human.report import (
    analyze_exp07,
    analyze_exp07b,
    plot_exp07,
    plot_exp07b,
    read_json,
    write_json,
    write_outcomes,
)

Log = Callable[[str], None]


def _code_version() -> str | None:
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ============================================================================= exp07
def run_exp07(
    defn: HumanHeldoutDefinition,
    data_path: Path,
    outdir: Path,
    *,
    robustness: bool = True,
    verify_hash: bool = True,
    log: Log = print,
) -> dict[str, Any]:
    code_version = _code_version()  # captured before any output is written
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "definition.json", defn.model_dump(mode="json"))  # frozen before any analysis
    t0 = time.perf_counter()
    digest = data.sha256_file(data_path)
    source = (
        data.PSYCH101_LEARNING_SUBSET
        if digest == data.PSYCH101_LEARNING_SUBSET.sha256
        else data.PSYCH101_FULL
    )
    if verify_hash:
        data.verify(data_path, source.sha256)
    parts = {pp.paradigm: psych101.load_participants(data_path, pp.paradigm) for pp in defn.paradigms}
    write_json(
        outdir / "validation.json",
        {
            "dataset": {
                **data.provenance(source, None),
                "observed_sha256": digest,
                "hash_verified": verify_hash,
            },
            "mcpl": psych101.validate_mcpl(parts["mcpl"]) if "mcpl" in parts else None,
            "igt": psych101.validate_igt(parts["igt"]) if "igt" in parts else None,
        },
    )
    seeds = [defn.split.seed] + (list(defn.split.robustness_seeds) if robustness else [])
    fits: dict[str, dict[str, Any]] = {}
    rows: list[OutcomeObservation] = []
    for s in seeds:
        fits[str(s)] = {}
        for pp in defn.paradigms:
            results, train, test = evaluate_paradigm(
                pp.paradigm, parts[pp.paradigm], defn, split_seed=s, log=log
            )
            fits[str(s)][pp.paradigm] = {
                "n_train": len(train),
                "n_test": len(test),
                "train_keys": [p.key for p in train],
                "test_keys": [p.key for p in test],
                "models": {
                    r.model: {
                        "status": r.status,
                        "note": r.note,
                        "params": r.params,
                        "lapse": r.lapse,
                        "train_objective": r.train_objective,
                        "n_grid": r.n_grid,
                        "seconds": round(r.seconds, 2),
                    }
                    for r in results
                },
            }
            for r in results:
                for key in sorted(r.test_nll):
                    common = dict(
                        experiment_id=defn.experiment_id,
                        arm=f"{pp.paradigm}:{r.model}",
                        seed=s,
                        agent_id=key,
                        split="human_holdout",
                        source="human_dataset",
                    )
                    rows.append(
                        OutcomeObservation(
                            metric="nll_per_choice", value=r.test_nll[key], unit="nats/choice", **common
                        )
                    )  # type: ignore[arg-type]
                    rows.append(
                        OutcomeObservation(
                            metric="top1_accuracy", value=r.test_top1[key], unit="fraction", **common
                        )
                    )  # type: ignore[arg-type]
                    rows.append(
                        OutcomeObservation(
                            metric="n_choices", value=float(r.test_n_choices[key]), unit="choices", **common
                        )
                    )  # type: ignore[arg-type]
    write_json(outdir / "fits.json", fits)
    write_outcomes(outdir / "outcomes.jsonl.gz", rows)
    write_json(
        outdir / "run_info.json",
        {
            "created_at": _now(),
            "code_version": code_version,
            "wall_time_s": round(time.perf_counter() - t0, 1),
            "registered_policy_classes": registry.registered(),
            "split_seeds": seeds,
            "hardware": f"{platform.machine()} {platform.processor() or ''} python {platform.python_version()}".strip(),
            "llm_calls": 0,
        },
    )
    return finalize_exp07(outdir)


def finalize_exp07(outdir: Path) -> dict[str, Any]:
    """Regenerate result.json, summary.json, figures and report.md from the files on disk."""
    result, summary = analyze_exp07(outdir)
    write_json(outdir / "result.json", result.model_dump(mode="json"))
    write_json(outdir / "summary.json", summary)
    defn = HumanHeldoutDefinition.model_validate(read_json(outdir / "definition.json"))
    plot_exp07(summary, defn.split.seed, outdir / "figures")
    (outdir / "report.md").write_text(
        render_exp07(outdir, defn, result.model_dump(mode="json"), summary), encoding="utf-8"
    )
    return {"result": result.model_dump(mode="json"), "summary": summary}


def _fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _p(p: float, n_boot: int) -> str:
    return f"≤{1 / n_boot:.4g} (bootstrap floor)" if p <= 1.0 / n_boot else f"{p:.3g}"


def render_exp07(
    outdir: Path, defn: HumanHeldoutDefinition, result: dict[str, Any], summary: dict[str, Any]
) -> str:
    info = read_json(outdir / "run_info.json")
    val = read_json(outdir / "validation.json")
    L = [
        f"# {defn.experiment_id} — {defn.title}",
        "",
        f"> {result['synthetic_disclaimer']}",
        "",
        f"* Preregistration hash: `{defn.preregistration_hash}` (definition committed before analysis)",
        f"* **Status: {result['status'].upper()}** — {result['status_reason']}",
        f"* Generated from: {result['generated_from']}; code version `{info['code_version']}`; wall time {info['wall_time_s']} s; LLM calls: 0",
        f"* Registered policy classes in this build: {', '.join(info['registered_policy_classes'])}",
        f"* Dataset: {val['dataset']['citation']} — licence {val['dataset']['licence']} ({val['dataset']['licence_evidence']}); sha256 `{val['dataset']['sha256']}`",
        "",
        "## Hypothesis and falsification",
        "",
        defn.hypothesis,
        "",
        f"Falsification: {defn.falsification_criterion}",
        "",
        f"Primary metric: {defn.analysis.primary_metric_definition}",
        "",
    ]
    order = [str(defn.split.seed)] + [k for k in summary["splits"] if k != str(defn.split.seed)]
    paradigm_order = [pp.paradigm for pp in defn.paradigms]
    for split_key in order:
        ssum = summary["splits"][split_key]
        tag = "primary split" if int(split_key) == defn.split.seed else "robustness split"
        for paradigm in [p for p in paradigm_order if p in ssum]:
            s = ssum[paradigm]
            L += [
                f"## {paradigm.upper()} — split seed {split_key} ({tag}); {s['n_train']} train / {s['n_test']} held-out participants",
                "",
                f"Chance NLL = {s['chance_nll']:.4f}. Best on TRAIN: non-developmental = `{s['best_non_developmental_on_train']}`, "
                f"adaptive = `{s['best_adaptive_on_train']}`, static = `{s['best_static_on_train']}`.",
                "",
                "| model | status | train NLL | held-out NLL [95% CI] | vs chance | top-1 acc | fitted params | lapse |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in sorted(s["models"], key=lambda r: r.get("test_nll_mean", 99)):
                if "test_nll_mean" in r:
                    L.append(
                        f"| {r['model']} | ok | {_fmt(r['train_objective'])} | {_fmt(r['test_nll_mean'])} "
                        f"[{_fmt(r['test_nll_ci_low'])}, {_fmt(r['test_nll_ci_high'])}] | {r['minus_chance']:+.4f} | "
                        f"{_fmt(r['test_top1_mean'], 3)} | `{r['params']}` | {r['lapse']} |"
                    )
                else:
                    L.append(f"| {r['model']} | {r['status']} | — | — | — | — | {r['note']} | — |")
            L.append("")
            if s["contrasts"]:
                L += [
                    "| contrast (NLL difference; negative favours first) | estimate [95% CI] | p | dz | n | status |",
                    "|---|---|---|---|---|---|",
                ]
                for c in s["contrasts"]:
                    L.append(
                        f"| {c['contrast']} ({c.get('label', c['role'])}) | {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                        f"{_p(c['p_value'], defn.analysis.n_bootstrap)} | {_fmt(c['cohen_dz'], 2)} | {c['n_participants']} | {c['status']} |"
                    )
                L.append("")
            if not s["developmental_available"]:
                L += [
                    "**The developmental learner was not available in this build, so the primary contrast was not evaluated.**",
                    "",
                ]
    L += [
        "## Figures",
        "",
        "![MCPL](figures/heldout_nll_mcpl.png)",
        "",
        "![IGT](figures/heldout_nll_igt.png)",
        "",
        "## Parse validation",
        "",
        "```json",
        json.dumps({"mcpl": val["mcpl"], "igt": val["igt"]}, indent=1, default=str),
        "```",
        "",
        "## Validity notes and limitations",
        "",
        *[f"* {v}" for v in defn.validity_notes],
        "",
        "Regenerate everything in this folder from the derived logs: `python -m hpe.evaluation.human analyze reports/human_validation/exp07_human_heldout`.",
        "",
    ]
    return "\n".join(L)


# ============================================================================= exp07b
def _term_rows(fit: causal.WCLSFit, names: list[str]) -> list[dict[str, Any]]:
    rows = fit.summary()
    for r, n in zip(rows, names, strict=True):
        r["term"] = n
        r["percent_effect"] = 100 * causal.percent_effect(r["estimate"])
    return rows


def _fit_variant(
    pts: list[heartsteps.DecisionPoint],
    *,
    outcome: str,
    controls: bool,
    probs_mode: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if outcome == "complete":
        pts = [d for d in pts if d.steps30_complete is not None]
        y = np.log(np.array([d.steps30_complete for d in pts], dtype=float) + 0.5)
    else:
        y = np.log(np.array([d.steps30 for d in pts], dtype=float) + 0.5)
    pre = np.log(np.array([d.steps30_pre for d in pts], dtype=float) + 0.5)
    day = np.array([d.study_day for d in pts], dtype=float)
    users = np.array([d.user for d in pts])
    g = np.column_stack([np.ones(len(pts)), pre, day]) if controls else np.ones((len(pts), 1))
    a_any = np.array([bool(d.send) for d in pts], dtype=float)[:, None]
    a_walk = np.array([bool(d.send and d.send_walking) for d in pts], dtype=float)
    a_sed = np.array([bool(d.send and d.send_sedentary) for d in pts], dtype=float)
    a2 = np.column_stack([a_walk, a_sed])
    if probs_mode == "nominal":
        p_any = np.full((len(pts), 1), 0.6)
        p2 = np.column_stack([np.full(len(pts), 0.3), np.full(len(pts), 0.3)])
    else:
        p_any = np.full((len(pts), 1), a_any.mean())
        p2 = np.column_stack([np.full(len(pts), a_walk.mean()), np.full(len(pts), a_sed.mean())])
    f_any = causal.wcls(y, g, a_any, p_any, users, n_bootstrap=n_boot, seed=seed)
    f_two = causal.wcls(y, g, a2, p2, users, n_bootstrap=n_boot, seed=seed + 1)
    terms = _term_rows(f_any, ["any_suggestion"]) + _term_rows(
        f_two, ["walking_suggestion", "antisedentary_suggestion"]
    )
    return {
        "outcome": outcome,
        "controls": controls,
        "centring": probs_mode,
        "p_any": float(p_any[0, 0]),
        "p_walking": float(p2[0, 0]),
        "p_sedentary": float(p2[0, 1]),
        "n_obs": f_any.n_obs,
        "n_clusters": f_any.n_clusters,
        "mean_outcome_steps": float(np.mean(np.exp(y) - 0.5)),
        "terms": terms,
    }


def run_exp07b(
    defn: CausalGroundingDefinition,
    csv_path: Path,
    outdir: Path,
    *,
    verify_hash: bool = True,
    log: Log = print,
) -> dict[str, Any]:
    code_version = _code_version()  # captured before any output is written
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "definition.json", defn.model_dump(mode="json"))
    digest = data.sha256_file(csv_path)
    if verify_hash:
        data.verify(csv_path, data.HEARTSTEPS_SUGGESTIONS.sha256)
    points = heartsteps.load_decision_points(csv_path)
    sample = heartsteps.analysis_sample(points)
    write_json(
        outdir / "validation.json",
        {
            "dataset": {
                **data.provenance(data.HEARTSTEPS_SUGGESTIONS),
                "observed_sha256": digest,
                "hash_verified": verify_hash,
            },
            "checks": heartsteps.validate(points),
        },
    )
    nb, seed = defn.n_bootstrap, defn.seed
    log(
        f"[exp07b] analysis sample {len(sample)} decision points, {len({d.user for d in sample})} participants"
    )
    primary = _fit_variant(
        sample, outcome="zero", controls=True, probs_mode="empirical", n_boot=nb, seed=seed
    )
    sens = {
        "complete_case": _fit_variant(
            sample, outcome="complete", controls=True, probs_mode="empirical", n_boot=nb, seed=seed + 10
        ),
        "nominal_centring": _fit_variant(
            sample, outcome="zero", controls=True, probs_mode="nominal", n_boot=nb, seed=seed + 20
        ),
        "no_controls": _fit_variant(
            sample, outcome="zero", controls=False, probs_mode="empirical", n_boot=nb, seed=seed + 30
        ),
    }
    # time-moderated marginal effect: (A - p) * (b0 + b1 * (day - centre))
    y = np.log(np.array([d.steps30 for d in sample], dtype=float) + 0.5)
    pre = np.log(np.array([d.steps30_pre for d in sample], dtype=float) + 0.5)
    day = np.array([d.study_day for d in sample], dtype=float)
    centre = float(np.round(day.mean(), 1))
    a = np.array([bool(d.send) for d in sample], dtype=float)[:, None]
    fit_tm = causal.wcls(
        y,
        np.column_stack([np.ones(len(y)), pre, day]),
        a,
        np.full((len(y), 1), a.mean()),
        np.array([d.user for d in sample]),
        moderators=np.column_stack([np.ones(len(y)), day - centre]),
        treatment_names=["any_suggestion"],
        moderator_names=["intercept", "day_minus_centre"],
        n_bootstrap=nb,
        seed=seed + 40,
    )
    tm_rows = fit_tm.summary()
    for r in primary["terms"]:
        log(
            f"  {r['term']:26s} beta={r['estimate']:+.4f} ({r['percent_effect']:+.1f}%) boot CI [{r['ci_low_bootstrap']:+.4f}, {r['ci_high_bootstrap']:+.4f}] p={r['p_value']:.3f}"
        )
    estimates = {
        "created_at": _now(),
        "code_version": code_version,
        "primary": primary,
        "sensitivity": sens,
        "time_moderated": {
            "terms": tm_rows,
            "beta0": tm_rows[0]["estimate"],
            "beta1": tm_rows[1]["estimate"],
            "day_centre": centre,
            "max_day": int(day.max()),
        },
        "published": [p.model_dump(mode="json") for p in defn.published],
    }
    write_json(outdir / "estimates.json", estimates)
    return finalize_exp07b(outdir)


def finalize_exp07b(outdir: Path) -> dict[str, Any]:
    result = analyze_exp07b(outdir)
    write_json(outdir / "result.json", result.model_dump(mode="json"))
    est = read_json(outdir / "estimates.json")
    plot_exp07b(est, outdir / "figures")
    defn = CausalGroundingDefinition.model_validate(read_json(outdir / "definition.json"))
    (outdir / "report.md").write_text(
        render_exp07b(outdir, defn, result.model_dump(mode="json"), est), encoding="utf-8"
    )
    return {"result": result.model_dump(mode="json")}


def render_exp07b(
    outdir: Path, defn: CausalGroundingDefinition, result: dict[str, Any], est: dict[str, Any]
) -> str:
    val = read_json(outdir / "validation.json")
    L = [
        f"# {defn.experiment_id} — {defn.title}",
        "",
        f"> {result['synthetic_disclaimer']}",
        "",
        f"* Preregistration hash: `{defn.preregistration_hash}`",
        f"* **Status: {result['status'].upper()}** — {result['status_reason']}",
        f"* Data: {val['dataset']['citation']}; licence {val['dataset']['licence']} ({val['dataset']['licence_evidence']}); sha256 `{val['dataset']['sha256']}`",
        f"* Code version `{est['code_version']}`; LLM calls: 0",
        "",
        "## Estimand and method",
        "",
        defn.estimand,
        "",
        f"Outcome `{defn.outcome}`; controls {defn.controls}; centring: {defn.centring}",
        "",
        "## Primary estimates",
        "",
        f"Analysis sample: {est['primary']['n_obs']} available decision points from {est['primary']['n_clusters']} participants; "
        f"P(any)={est['primary']['p_any']:.3f}, P(walking)={est['primary']['p_walking']:.3f}, P(anti-sedentary)={est['primary']['p_sedentary']:.3f}; "
        f"mean 30-min steps {est['primary']['mean_outcome_steps']:.1f}.",
        "",
        "| term | beta (log) | % effect | 95% cluster bootstrap CI (%) | sandwich SE | p (t, df) |",
        "|---|---|---|---|---|---|",
    ]
    for t in est["primary"]["terms"]:
        L.append(
            f"| {t['term']} | {t['estimate']:+.4f} | {t['percent_effect']:+.1f}% | "
            f"[{100 * math.expm1(t['ci_low_bootstrap']):+.1f}, {100 * math.expm1(t['ci_high_bootstrap']):+.1f}] | "
            f"{t['se_cluster_robust']:.4f} | {t['p_value']:.3f} ({t['df']}) |"
        )
    L += ["", "## Comparison with the published estimate (verified from the abstract, PMID 30192907)", ""]
    for p in est["published"]:
        L.append(
            f'* {p["term"]}: published {p["published_percent"]:+.0f}% ({p["published_p"]}). Source text: "{p["verified_text"].strip()}"'
        )
    cmp = next(s for s in result["sensitivity"] if "comparisons_with_published" in s)[
        "comparisons_with_published"
    ]
    L += ["", "| term | published (log) | inside our 95% CI | same sign |", "|---|---|---|---|"]
    for c in cmp:
        L.append(f"| {c['term']} | {c['published_log_effect']:+.4f} | {c['inside_ci']} | {c['same_sign']} |")
    L += [
        "",
        "## Sensitivity analyses",
        "",
        "| variant | term | % effect | 95% bootstrap CI (%) | p |",
        "|---|---|---|---|---|",
    ]
    for name, v in est["sensitivity"].items():
        for t in v["terms"]:
            L.append(
                f"| {name} | {t['term']} | {t['percent_effect']:+.1f}% | [{100 * math.expm1(t['ci_low_bootstrap']):+.1f}, "
                f"{100 * math.expm1(t['ci_high_bootstrap']):+.1f}] | {t['p_value']:.3f} |"
            )
    tm = est["time_moderated"]
    L += [
        "",
        "## Time-moderated effect (exploratory secondary)",
        "",
        f"(A - p)(b0 + b1 (day - {tm['day_centre']})): b0 = {tm['beta0']:+.4f}, b1 = {tm['beta1']:+.5f} per study day "
        f"(bootstrap CI [{tm['terms'][1]['ci_low_bootstrap']:+.5f}, {tm['terms'][1]['ci_high_bootstrap']:+.5f}]).",
        "",
        "![effects](figures/effects_vs_published.png)",
        "",
        "![time](figures/effect_by_study_day.png)",
        "",
        "## Identification assumptions",
        "",
        *[f"* {a}" for a in defn.identification_assumptions],
        "",
        "## Validity notes",
        "",
        *[f"* {v}" for v in defn.validity_notes],
        "",
        "## Data checks",
        "",
        "```json",
        json.dumps(val["checks"], indent=1),
        "```",
        "",
    ]
    return "\n".join(L)
