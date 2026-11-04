"""Regenerate experiment results, figures and reports from RAW LOG FILES only.

Inputs (all on disk): ``definition.json``, ``protocol.json``, ``runs/index.json`` and every
``runs/<run_id>/outcomes.jsonl.gz`` + ``manifest.json``. Nothing is taken from in-memory state, so
``python -m hpe.evaluation analyze <dir>`` reproduces ``result.json`` byte-for-byte.

Unit of analysis: one value per (arm, seed) = mean over agents of the per-agent metric. Seeds are
independent replications (separate task worlds and populations), so n = number of seeds.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hpe.contracts.schemas import (
    ArmSummary,
    EffectEstimate,
    ExperimentDefinition,
    ExperimentResult,
    RunManifest,
)
from hpe.evaluation import stats
from hpe.evaluation.runner import MAIN_VARIANT, read_jsonl_gz
from hpe.evaluation.spec import ContrastSpec, ProtocolSpec, StrongestOf

CURVE_PREFIXES = ("curve/", "transfer_curve/")
Values = dict[str, dict[str, dict[int, float]]]  # variant -> arm -> seed -> value (one metric)


class RawData:
    """Per-seed unit values and curves loaded from raw outcome logs."""

    def __init__(self, exp_dir: Path) -> None:
        self.exp_dir = exp_dir
        self.definition = ExperimentDefinition.model_validate_json((exp_dir / "definition.json").read_text())
        self.protocol = ProtocolSpec.model_validate_json((exp_dir / "protocol.json").read_text())
        idx = json.loads((exp_dir / "runs" / "index.json").read_text())
        self.skipped_arms: dict[str, str] = idx.get("skipped_arms", {})
        self.jobs: list[dict[str, Any]] = idx["jobs"]
        self.manifests: dict[str, RunManifest] = {}
        # (variant, arm, seed, metric) -> list of agent-level values
        summary: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
        # (variant, arm, seed, metric, tick) -> list
        curves: dict[tuple[str, str, int, str, int], list[float]] = defaultdict(list)
        self.missing_runs: list[str] = []
        for j in sorted(self.jobs, key=lambda x: x["run_id"]):
            rd = exp_dir / "runs" / j["run_id"]
            if not (rd / "outcomes.jsonl.gz").exists():
                self.missing_runs.append(j["run_id"])
                continue
            self.manifests[j["run_id"]] = RunManifest.model_validate_json((rd / "manifest.json").read_text())
            for o in read_jsonl_gz(rd / "outcomes.jsonl.gz"):
                if o.metric.startswith(CURVE_PREFIXES):
                    curves[(j["variant"], j["arm"], int(j["seed"]), o.metric, int(o.tick or 0))].append(
                        o.value
                    )
                else:
                    summary[(j["variant"], j["arm"], int(j["seed"]), o.metric)].append(o.value)
        self.summary = {k: float(np.mean(v)) for k, v in summary.items()}
        self.curves = {k: float(np.mean(v)) for k, v in curves.items()}
        self.metrics = sorted({k[3] for k in self.summary})
        self.curve_metrics = sorted({k[3] for k in self.curves})

    def values(self, metric: str, variant: str = MAIN_VARIANT) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = defaultdict(dict)
        for (v, arm, seed, m), val in self.summary.items():
            if v == variant and m == metric:
                out[arm][seed] = val
        return dict(out)

    def arms_in(self, variant: str) -> list[str]:
        return sorted({k[1] for k in self.summary if k[0] == variant})

    def variants(self) -> list[str]:
        vs = sorted({k[0] for k in self.summary})
        return [MAIN_VARIANT] + [v for v in vs if v != MAIN_VARIANT] if MAIN_VARIANT in vs else vs


def _seed_for(prereg: str, salt: int) -> int:
    return (int(prereg[:8], 16) + 7919 * salt) % (2**31)


def _resolve_control(c: ContrastSpec, vals: dict[str, dict[int, float]]) -> tuple[str | None, str]:
    if not isinstance(c.control, StrongestOf):
        return (c.control if c.control in vals else None), ""
    cands = [a for a in c.control.strongest_of if vals.get(a)]
    if not cands:
        return None, ""
    means = {a: float(np.mean(list(vals[a].values()))) for a in cands}
    best = (
        max(cands, key=lambda a: means[a]) if c.direction == "greater" else min(cands, key=lambda a: means[a])
    )
    note = "strongest of " + ", ".join(f"{a}={means[a]:.4f}" for a in sorted(cands)) + f" -> {best}"
    return best, note


def evaluate_contrast(c: ContrastSpec, data: RawData, variant: str, salt: int) -> dict[str, Any]:
    d = data.definition
    vals = data.values(c.metric, variant)
    ctrl, note = _resolve_control(c, vals)
    base: dict[str, Any] = {
        "contrast": c.name,
        "variant": variant,
        "metric": c.metric,
        "kind": c.kind,
        "treatment": c.treatment,
        "control": ctrl,
        "control_selection": note,
        "primary": c.primary,
    }
    if ctrl is None or c.treatment not in vals:
        missing = [a for a in [c.treatment, ctrl] if a is None or a not in vals]
        return {**base, "verdict": "not_computed", "reason": f"arm data unavailable: {missing or 'control'}"}
    t, k = vals[c.treatment], vals[ctrl]
    seeds = sorted(set(t) & set(k)) if c.paired else sorted(set(t) | set(k))
    if len(seeds) < 2:
        return {**base, "verdict": "not_computed", "reason": f"only {len(seeds)} paired seed(s)"}
    tv = np.array([t[s] for s in seeds if s in t])
    kv = np.array([k[s] for s in seeds if s in k])
    n_boot = d.analysis.n_bootstrap
    seed = _seed_for(d.preregistration_hash or "0" * 8, salt)
    level = 1 - d.analysis.alpha
    if c.paired:
        b = stats.paired_bootstrap(tv, kv, n_boot=n_boot, level=level, seed=seed)
        p = stats.permutation_test_paired(tv, kv, seed=seed)
        std = stats.cohen_dz(tv, kv)
        method = (
            f"paired percentile bootstrap (n_boot={n_boot}) over seeds; sign-flip permutation p; Cohen dz"
        )
    else:
        b = stats.unpaired_bootstrap(tv, kv, n_boot=n_boot, level=level, seed=seed)
        p = stats.permutation_test_unpaired(tv, kv, seed=seed)
        std = stats.hedges_g(tv, kv)
        method = f"unpaired percentile bootstrap (n_boot={n_boot}); label permutation p; Hedges g"
    sign = 1.0 if c.direction == "greater" else -1.0
    imp, lo, hi = (
        sign * b.estimate,
        min(sign * b.ci_low, sign * b.ci_high),
        max(sign * b.ci_low, sign * b.ci_high),
    )
    control_mean = float(np.mean(kv))
    thr = None
    if c.min_effect is not None:
        thr = c.min_effect * abs(control_mean) if c.min_effect_kind == "relative" else c.min_effect
    rel = imp / abs(control_mean) if control_mean != 0 else float("nan")
    if c.kind == "superiority":
        target = thr if thr is not None else 0.0
        if imp >= target and lo > 0:
            verdict = "passed"
        elif (thr is not None and hi < thr) or (thr is None and hi < 0):
            verdict = "failed"
        else:
            verdict = "inconclusive"
    elif c.kind == "equivalence":
        m = float(c.margin or 0.0)
        if -m < b.ci_low and b.ci_high < m:
            verdict = "passed"
        elif b.ci_low > m or b.ci_high < -m:
            verdict = "failed"
        else:
            verdict = "inconclusive"
    elif c.kind == "non_inferiority":
        m = float(c.margin or 0.0)
        verdict = "passed" if lo > -m else ("failed" if hi < -m else "inconclusive")
    else:
        verdict = "exploratory"
    return {
        **base,
        "verdict": verdict,
        "estimate": b.estimate,
        "ci_low": b.ci_low,
        "ci_high": b.ci_high,
        "ci_level": level,
        "improvement": imp,
        "relative_improvement": rel,
        "threshold": thr,
        "treatment_mean": float(np.mean(tv)),
        "control_mean": control_mean,
        "p_value": p,
        "standardized_effect": std,
        "n_units": int(len(tv)),
        "method": method,
        "seeds": seeds,
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def _figures(data: RawData, out: Path, primary: ContrastSpec) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    figs: list[str] = []
    arms = data.arms_in(MAIN_VARIANT)
    colors = {a: plt.get_cmap("tab10")(i % 10) for i, a in enumerate(arms)}
    synthetic = "Synthetic simulation — not evidence about real people"

    # 1. primary metric per arm (mean over seeds, 95% bootstrap CI)
    vals = data.values(primary.metric)
    if vals:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        names = [a for a in arms if a in vals]
        for i, a in enumerate(names):
            s = stats.summarize(list(vals[a].values()), seed=i)
            ax.bar(i, s["mean"], color=colors[a], alpha=0.85)
            ax.errorbar(
                i,
                s["mean"],
                yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                color="k",
                capsize=4,
            )
            ax.scatter([i] * len(vals[a]), list(vals[a].values()), color="k", s=8, zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
        ax.set_ylabel(primary.metric)
        ax.set_title(
            f"{data.definition.experiment_id}: primary metric (mean, 95% CI; dots = seeds)", fontsize=9
        )
        fig.text(0.01, 0.01, synthetic, fontsize=7, alpha=0.7)
        fig.tight_layout()
        fig.savefig(out / "primary_metric.png", dpi=120)
        plt.close(fig)
        figs.append("figures/primary_metric.png")

    # 2. learning curves: mean over training families of curve/heldout_accuracy/<f>
    lc = [m for m in data.curve_metrics if m.startswith("curve/heldout_accuracy/")]
    if lc:
        fig, ax = plt.subplots(figsize=(7, 4))
        for a in arms:
            per_tick: dict[int, list[float]] = defaultdict(list)
            for (v, arm, _seed, m, tick), val in data.curves.items():
                if v == MAIN_VARIANT and arm == a and m in lc:
                    per_tick[tick].append(val)
            if not per_tick:
                continue
            ticks = sorted(per_tick)
            # average families within seed first is implicit (equal counts per tick)
            mean = [float(np.mean(per_tick[t])) for t in ticks]
            ax.plot(ticks, mean, marker="o", ms=3, label=a, color=colors[a])
        ax.set_xlabel("practice episodes per family")
        ax.set_ylabel("held-out accuracy (unseen symbols)")
        ax.set_title("Skill acquisition on held-out items (learning disabled during assessment)", fontsize=9)
        ax.legend(fontsize=8)
        fig.text(0.01, 0.01, synthetic, fontsize=7, alpha=0.7)
        fig.tight_layout()
        fig.savefig(out / "learning_curves.png", dpi=120)
        plt.close(fig)
        figs.append("figures/learning_curves.png")

    # 3. transfer curves
    for mode in sorted({m.split("/")[1] for m in data.curve_metrics if m.startswith("transfer_curve/")}):
        fig, ax = plt.subplots(figsize=(7, 4))
        for a in arms:
            for label, ls in (("trained", "-"), ("fresh", ":")):
                m = f"transfer_curve/{mode}/{label}"
                by_shot: dict[int, list[float]] = defaultdict(list)
                for (v, arm, _seed, mm, tick), val in data.curves.items():
                    if v == MAIN_VARIANT and arm == a and mm == m:
                        by_shot[tick].append(val)
                if by_shot:
                    shots = sorted(by_shot)
                    ax.plot(
                        shots,
                        [float(np.mean(by_shot[t])) for t in shots],
                        ls=ls,
                        marker="o",
                        ms=3,
                        color=colors[a],
                        label=f"{a} ({label})",
                    )
        ax.set_xscale("symlog", linthresh=4)
        ax.set_xlabel("episodes on the never-practised 'barter' family")
        ax.set_ylabel("held-out accuracy")
        ax.set_title(f"Transfer to an unseen task family — {mode} latent structure", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        fig.text(0.01, 0.01, synthetic, fontsize=7, alpha=0.7)
        fig.tight_layout()
        fig.savefig(out / f"transfer_{mode}.png", dpi=120)
        plt.close(fig)
        figs.append(f"figures/transfer_{mode}.png")
    return figs


def analyze(exp_dir: str | Path, *, make_figures: bool = True) -> ExperimentResult:
    exp_dir = Path(exp_dir)
    data = RawData(exp_dir)
    d, proto = data.definition, data.protocol
    prereg = d.preregistration_hash or ""

    # Arm summaries (main variant)
    arms: list[ArmSummary] = []
    for i, a in enumerate(data.arms_in(MAIN_VARIANT)):
        metrics: dict[str, dict[str, float]] = {}
        for m in data.metrics:
            v = data.values(m).get(a)
            if v:
                metrics[m] = stats.summarize(
                    list(v.values()), seed=_seed_for(prereg, 1000 + i), n_boot=d.analysis.n_bootstrap
                )
        n_runs = len({j["seed"] for j in data.jobs if j["variant"] == MAIN_VARIANT and j["arm"] == a})
        arms.append(ArmSummary(arm=a, n_runs=n_runs, metrics=metrics))

    # Contrasts (main variant)
    main = [evaluate_contrast(c, data, MAIN_VARIANT, salt=i) for i, c in enumerate(proto.contrasts)]
    confirmatory = [r for r in main if not r["primary"] and r["kind"] != "exploratory" and "p_value" in r]
    adj = stats.holm([r["p_value"] for r in confirmatory])
    for r, a_ in zip(confirmatory, adj, strict=True):
        r["p_value_holm"] = a_
    effects = [
        EffectEstimate(
            metric=r["metric"],
            contrast=f"{r['contrast']}: {r['treatment']} - {r['control']}",
            estimate=r["estimate"],
            ci_low=r["ci_low"],
            ci_high=r["ci_high"],
            ci_level=r["ci_level"],
            p_value=r.get("p_value_holm", r["p_value"]),
            standardized_effect=r["standardized_effect"] if np.isfinite(r["standardized_effect"]) else None,
            n_units=r["n_units"],
            unit_of_analysis="seed (independent task world + population)",
            method=r["method"] + ("; Holm-adjusted p" if "p_value_holm" in r else ""),
            passed_minimum_effect=(r["verdict"] == "passed") if r["kind"] != "exploratory" else None,
        )
        for r in main
        if "estimate" in r
    ]
    prim = next(r for r in main if r["primary"])
    if prim["verdict"] == "not_computed":
        status, reason = "inconclusive", f"primary contrast not computed: {prim['reason']}"
    else:
        status = {"passed": "passed", "failed": "failed"}.get(prim["verdict"], "inconclusive")
        thr = prim["threshold"]
        reason = (
            f"primary contrast '{prim['contrast']}' ({prim['kind']}): {prim['treatment']} vs {prim['control']} on "
            f"{prim['metric']}: estimate {prim['estimate']:+.4f} (95% CI [{prim['ci_low']:+.4f}, {prim['ci_high']:+.4f}], "
            f"n={prim['n_units']} seeds; relative {prim['relative_improvement']:+.1%}); "
            f"threshold {_fmt(thr)} -> {prim['verdict']}"
        )
        if prim["control_selection"]:
            reason += f"; control = {prim['control_selection']}"

    # Sensitivity variants
    sensitivity: list[dict[str, Any]] = []
    variant_specs = {v.name: v for v in proto.sensitivity}
    for vname in data.variants():
        if vname == MAIN_VARIANT:
            continue
        vs = variant_specs.get(vname)
        names = vs.contrasts if vs and vs.contrasts else [c.name for c in proto.contrasts]
        for i, c in enumerate(proto.contrasts):
            if c.name in names:
                r = evaluate_contrast(c, data, vname, salt=100 + i)
                r["variant_params"] = vs.params if vs else {}
                sensitivity.append(r)

    # Ablations: each ablation arm vs the primary treatment arm, on the primary metric
    ablations: list[dict[str, Any]] = []
    for i, ab in enumerate(d.ablations):
        c = ContrastSpec(
            name=f"ablation:{ab}",
            treatment=ab,
            control=proto.primary.treatment,
            metric=proto.primary.metric,
            kind="exploratory",
            direction=proto.primary.direction,
        )
        ablations.append(evaluate_contrast(c, data, MAIN_VARIANT, salt=500 + i))

    figures = _figures(data, exp_dir / "figures", proto.primary) if make_figures else []

    limitations = list(d.validity_notes)
    for arm, why in sorted(data.skipped_arms.items()):
        limitations.append(f"Arm '{arm}' was NOT run: {why}.")
    if data.missing_runs:
        limitations.append(
            f"{len(data.missing_runs)} preregistered run(s) missing raw logs: {data.missing_runs[:5]}"
        )
    if proto.pipeline_validation:
        limitations.insert(
            0,
            "PIPELINE VALIDATION RUN: scaled-down budget and seeds; not the preregistered experiment. "
            "Do not cite as a research result.",
        )
    created = max((m.created_at for m in data.manifests.values()), default=d.created_at)
    result = ExperimentResult(
        experiment_id=d.experiment_id,
        preregistration_hash=prereg,
        status=status,  # type: ignore[arg-type]
        status_reason=reason,
        arms=arms,
        effects=effects,
        run_ids=sorted(data.manifests),
        sensitivity=[_jsonable(s) for s in sensitivity],
        ablations=[_jsonable(a) for a in ablations],
        figures=figures,
        limitations=limitations,
        generated_from="raw outcome logs: runs/*/outcomes.jsonl.gz (+ manifests)",
        created_at=created,
    )
    (exp_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    (exp_dir / "contrasts.json").write_text(json.dumps(_jsonable(main), indent=2, sort_keys=True) + "\n")
    (exp_dir / "report.md").write_text(render_report(result, d, proto, main, data))
    return result


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.floating | np.integer):
        return _jsonable(o.item())
    return o


def render_report(
    result: ExperimentResult,
    d: ExperimentDefinition,
    proto: ProtocolSpec,
    main: list[dict[str, Any]],
    data: RawData,
) -> str:
    L: list[str] = []
    title = d.title
    L.append(f"# {title}")
    L.append("")
    if proto.pipeline_validation:
        L.append(
            "> **PIPELINE VALIDATION RUN** — scaled-down budget/seeds to validate code paths. Not a research result."
        )
        L.append("")
    L.append(f"> {result.synthetic_disclaimer}")
    L.append("")
    L.append(f"- Experiment id: `{d.experiment_id}`  ")
    L.append(
        f"- Preregistration hash: `{result.preregistration_hash}` (written to definition.json before any run)  "
    )
    L.append(f"- Protocol: `{proto.kind}`; seeds: {d.seeds}; arms: {', '.join(a.name for a in d.arms)}  ")
    L.append(f"- Status: **{result.status.upper()}** — {result.status_reason}")
    L.append("")
    L.append("## Hypothesis and falsification criterion")
    L.append("")
    L.append(f"**Hypothesis.** {d.hypothesis}")
    L.append("")
    L.append(f"**Falsification criterion.** {d.falsification_criterion}")
    L.append("")
    L.append(f"**Primary metric.** `{d.analysis.primary_metric}` — {d.analysis.primary_metric_definition}")
    L.append("")
    L.append("## Arm summaries (mean over seeds, 95% bootstrap CI)")
    L.append("")
    show = [d.analysis.primary_metric, *d.analysis.secondary_metrics]
    show = [m for m in show if any(m in a.metrics for a in result.arms)]
    L.append("| arm | n seeds | " + " | ".join(f"`{m}`" for m in show) + " |")
    L.append("|---|---|" + "---|" * len(show))
    for a in result.arms:
        cells = []
        for m in show:
            s = a.metrics.get(m)
            cells.append("—" if not s else f"{s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
        L.append(f"| {a.arm} | {a.n_runs} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Preregistered contrasts")
    L.append("")
    L.append("| contrast | kind | treatment | control | estimate | 95% CI | relative | p | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in main:
        if "estimate" not in r:
            L.append(
                f"| {r['contrast']}{' (primary)' if r['primary'] else ''} | {r['kind']} | {r['treatment']} | {r['control']} | — | — | — | — | {r['verdict']}: {r['reason']} |"
            )
            continue
        p = r.get("p_value_holm", r["p_value"])
        L.append(
            f"| {r['contrast']}{' (primary)' if r['primary'] else ''} | {r['kind']} | {r['treatment']} | {r['control']} | "
            f"{r['estimate']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['relative_improvement']:+.1%} | "
            f"{_fmt(p, 4)} | **{r['verdict']}** |"
        )
    L.append("")
    if result.sensitivity:
        L.append("## Sensitivity analyses")
        L.append("")
        L.append("| variant | params | contrast | estimate | 95% CI | verdict |")
        L.append("|---|---|---|---|---|---|")
        for r in result.sensitivity:
            est = "—" if r.get("estimate") is None else f"{r['estimate']:+.4f}"
            ci = "—" if r.get("ci_low") is None else f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            L.append(
                f"| {r['variant']} | `{json.dumps(r.get('variant_params', {}), sort_keys=True)}` | {r['contrast']} | {est} | {ci} | {r['verdict']} |"
            )
        L.append("")
    if result.ablations:
        L.append("## Ablations (ablated arm minus full learner, primary metric)")
        L.append("")
        L.append("| ablation | estimate | 95% CI | n |")
        L.append("|---|---|---|---|")
        for r in result.ablations:
            if r.get("estimate") is None:
                L.append(f"| {r['treatment']} | — | — | {r.get('reason', '')} |")
            else:
                L.append(
                    f"| {r['treatment']} | {r['estimate']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['n_units']} |"
                )
        L.append("")
    leak = [m for m in data.metrics if m.startswith("leakage/")]
    if leak:
        L.append("## Contamination checks (from raw logs)")
        L.append("")
        for m in leak:
            vals = [v for arm_vals in data.values(m).values() for v in arm_vals.values()]
            L.append(f"- `{m}`: max over runs = {max(vals):g}")
        L.append("")
    if result.figures:
        L.append("## Figures (regenerated from raw logs)")
        L.append("")
        for f in result.figures:
            L.append(f"![{f}]({f})")
        L.append("")
    L.append("## Compute")
    L.append("")
    walls = [m.compute.wall_time_s for m in data.manifests.values() if m.compute]
    L.append(
        f"- Runs: {len(data.manifests)}; total wall time {sum(walls):.1f} s; LLM calls: 0; paid API cost: $0."
    )
    if data.manifests:
        hw = next(iter(data.manifests.values())).compute
        L.append(f"- Hardware: {hw.hardware if hw else 'n/a'}")
    L.append("")
    L.append("## Limitations")
    L.append("")
    for lim in result.limitations:
        L.append(f"- {lim}")
    L.append(f"- {result.synthetic_disclaimer}")
    L.append("")
    L.append("## Reproduce")
    L.append("")
    L.append("```bash")
    L.append(
        f"python -m hpe.evaluation run experiments/{d.experiment_id.replace('__pipeline_validation', '')}.yaml"
        + (" --pipeline-validation" if proto.pipeline_validation else "")
    )
    L.append(
        f"python -m hpe.evaluation analyze <out>/{d.experiment_id}   # regenerates result.json from raw logs"
    )
    L.append("```")
    L.append("")
    return "\n".join(L)
