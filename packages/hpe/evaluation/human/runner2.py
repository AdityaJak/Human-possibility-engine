"""Phase II runner: partition-aware held-out human evaluation with equal tuning and a locked test set.

Outputs (``<out>/<experiment_id>/``): definition.json (frozen before analysis), partitions_ref.json,
validation.json (parse descriptives on the partitions used), fits.json, outcomes.jsonl.gz
(per-participant metrics as OutcomeObservation rows), summary.json, report.md. ``analyze2``
regenerates summary.json and report.md from fits.json + outcomes.jsonl.gz only.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hpe.contracts import registry
from hpe.contracts.schemas import OutcomeObservation
from hpe.evaluation.human import data, psych101, psych101_v2
from hpe.evaluation.human.definition import HumanV2Definition
from hpe.evaluation.human.harness2 import ModelResult2, evaluate_paradigm2, expected_calibration_error
from hpe.evaluation.human.models2 import REFERENCE_MODELS2
from hpe.evaluation.human.partitions import Partitions, check_partition_access, iter_partition_records
from hpe.evaluation.human.psych101 import Participant
from hpe.evaluation.human.report import bootstrap_mean, read_json, read_outcomes, write_json, write_outcomes

Log = Callable[[str], None]
CHANCE = {
    p: math.log(k)
    for p, k in {
        "mcpl": 9,
        "igt": 4,
        "bandit2": 2,
        "srl3": 3,
        "horizon": 2,
        "casino": 2,
        "wpt": 2,
        "garcia": 2,
        "described": 2,
    }.items()
}


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


def load_partition_participants(
    data_path: Path,
    paradigm: str,
    experiments: list[str],
    parts: Partitions,
    partition: str,
    *,
    experiment_id: str,
) -> list[Participant]:
    """Parse only the participants of ``partition`` (test records are skipped unread unless unlocked)."""
    check_partition_access(partition, experiment_id)
    if paradigm in ("mcpl", "igt"):
        parser = psych101.parse_mcpl if paradigm == "mcpl" else psych101.parse_igt
        prefixes: tuple[str, ...] = (psych101.MCPL_PREFIX,) if paradigm == "mcpl" else (psych101.IGT_PREFIX,)
    else:
        prefixes = psych101_v2.PARADIGM_PREFIXES[paradigm]
        parser = psych101_v2.PARSERS[paradigm]
    out: list[Participant] = []
    excluded: list[str] = []
    for rec in iter_partition_records(
        psych101.iter_records(data_path, prefixes),
        parts,
        partition,
        experiments=experiments,
        experiment_id=experiment_id,
    ):
        try:
            p = parser(rec["text"], rec["experiment"], str(rec["participant"]))
        except psych101.ParseError as e:
            excluded.append(str(e)[:200])
            continue
        if not p.trials or all(getattr(t, "forced", False) for t in p.trials):
            excluded.append(f"{p.key}: no scorable trials")
            continue
        out.append(p)
    out.sort(key=lambda p: p.key)
    if excluded:
        out_meta = {"excluded": excluded}
        for p in out:
            p.meta.setdefault("_partition_exclusions", out_meta)
    return out


def _validation(paradigm: str, parts_by_partition: dict[str, list[Participant]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part, plist in parts_by_partition.items():
        if not plist:
            out[part] = {"participants": 0}
            continue
        if paradigm == "mcpl":
            rep: Any = psych101.validate_mcpl(plist)
        elif paradigm == "igt":
            rep = psych101.validate_igt(plist)
        else:
            rep = psych101_v2.validate_generic(plist)
        excl = plist[0].meta.get("_partition_exclusions", {}).get("excluded", []) if plist else []
        ambiguous = sum(int(p.meta.get("ambiguous_problem_choices", 0)) for p in plist)
        out[part] = {
            "participants": len(plist),
            "descriptives": rep,
            "excluded": excl,
            "ambiguous_choices_skipped": ambiguous,
        }
    return out


def run_v2(
    defn: HumanV2Definition, data_path: Path, outdir: Path, *, verify_hash: bool = True, log: Log = print
) -> dict[str, Any]:
    code_version = _code_version()
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "definition.json", defn.model_dump(mode="json"))  # frozen before any analysis
    t0, c0 = time.perf_counter(), time.process_time()
    parts = Partitions.load(Path(defn.partitions_file))
    write_json(
        outdir / "partitions_ref.json",
        {
            "file": defn.partitions_file,
            "split_seed": parts.split_seed,
            "test_partition_sha256": parts.test_partition_sha256,
        },
    )
    digest = data.sha256_file(data_path)
    source = (
        data.PSYCH101_LEARNING_SUBSET
        if digest == data.PSYCH101_LEARNING_SUBSET.sha256
        else data.PSYCH101_FULL
    )
    if verify_hash:
        data.verify(data_path, source.sha256)
    needed = sorted({defn.fit_partition, defn.select_partition, defn.report_partition})
    fits: dict[str, Any] = {}
    validation: dict[str, Any] = {
        "dataset": {**data.provenance(source, None), "observed_sha256": digest},
        "paradigms": {},
    }
    rows: list[OutcomeObservation] = []
    for pp in defn.paradigms:
        plist: dict[str, list[Participant]] = {
            part: load_partition_participants(
                data_path, pp.paradigm, pp.experiments, parts, part, experiment_id=defn.experiment_id
            )
            for part in needed
        }
        validation["paradigms"][pp.paradigm] = _validation(pp.paradigm, plist)
        results = evaluate_paradigm2(
            pp.paradigm,
            plist,
            models=defn.models + defn.reference_models,
            grids=defn.grids,
            lapse_grid=defn.lapse_grid,
            igt_scales=defn.igt_reward_scale_grid,
            seed=defn.seed,
            fit=defn.fit_partition,
            select=defn.select_partition,
            report=defn.report_partition,
            log=log,
        )
        fits[pp.paradigm] = {
            "experiments": pp.experiments,
            "role": pp.role,
            "n": {part: len(v) for part, v in plist.items()},
            "models": {
                r.model: {
                    "status": r.status,
                    "note": r.note,
                    "params": r.params,
                    "lapse": r.lapse,
                    "n_grid": r.n_grid,
                    "grid_edges": r.grid_edges,
                    "select_objective": r.select_objective,
                    "cpu_seconds": round(r.cpu_seconds, 2),
                    "param_count": r.param_count,
                    "fit_log": _trim_log(r.fit_log),
                }
                for r in results
            },
        }
        rows += _outcome_rows(defn, pp.paradigm, results)
    write_json(outdir / "validation.json", validation)
    write_json(outdir / "fits.json", fits)
    write_outcomes(outdir / "outcomes.jsonl.gz", rows)
    write_json(
        outdir / "run_info.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "code_version": code_version,
            "wall_time_s": round(time.perf_counter() - t0, 1),
            "cpu_time_s": round(time.process_time() - c0, 1),
            "registered_policy_classes": registry.registered(),
            "hardware": f"{platform.machine()} {platform.processor() or ''} python {platform.python_version()}".strip(),
            "llm_calls": 0,
            "partitions": {
                "fit": defn.fit_partition,
                "select": defn.select_partition,
                "report": defn.report_partition,
            },
        },
    )
    return finalize_v2(outdir)


def _trim_log(log: dict[str, Any] | None) -> dict[str, Any] | None:
    if not log:
        return None
    return {
        "param_count": log.get("param_count"),
        "best_val_nll": log.get("best_val_nll"),
        "epochs": len(log.get("epochs", [])),
    }


def _outcome_rows(
    defn: HumanV2Definition, paradigm: str, results: list[ModelResult2]
) -> list[OutcomeObservation]:
    rows: list[OutcomeObservation] = []
    for r in results:
        for key, s in sorted(r.scores.items()):
            common = dict(
                experiment_id=defn.experiment_id,
                arm=f"{paradigm}:{r.model}",
                seed=defn.seed,
                agent_id=key,
                split="human_holdout",
                source="human_dataset",
            )
            rows.append(
                OutcomeObservation(metric="nll_per_choice", value=s.nll, unit="nats/choice", **common)
            )  # type: ignore[arg-type]
            rows.append(OutcomeObservation(metric="top1_accuracy", value=s.top1, unit="fraction", **common))  # type: ignore[arg-type]
            rows.append(OutcomeObservation(metric="n_choices", value=float(s.n), unit="choices", **common))  # type: ignore[arg-type]
            for qi, v in enumerate(s.nll_by_quarter):
                if not math.isnan(v):
                    rows.append(
                        OutcomeObservation(
                            metric=f"nll_quarter_{qi + 1}", value=v, unit="nats/choice", **common
                        )
                    )  # type: ignore[arg-type]
            ece, _ = expected_calibration_error(s.conf, s.hit)
            rows.append(OutcomeObservation(metric="ece", value=ece, unit="abs", **common))  # type: ignore[arg-type]
            rows.append(
                OutcomeObservation(
                    metric="mean_confidence",
                    value=float(s.conf.mean()) if len(s.conf) else float("nan"),
                    unit="prob",
                    **common,
                )
            )  # type: ignore[arg-type]
    return rows


# ============================================================================= analysis from logs
def analyze_v2(outdir: Path) -> dict[str, Any]:
    defn = HumanV2Definition.model_validate(read_json(outdir / "definition.json"))
    fits = read_json(outdir / "fits.json")
    rows = read_outcomes(outdir / "outcomes.jsonl.gz")
    nb, seed = defn.analysis.n_bootstrap, defn.seed
    by: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        dict
    )  # (paradigm, model, participant) -> metric -> value
    for r in rows:
        par, model = (r.arm or "").split(":", 1)
        by[(par, model, r.agent_id or "")][r.metric] = r.value
    summary: dict[str, Any] = {
        "experiment_id": defn.experiment_id,
        "report_partition": defn.report_partition,
        "paradigms": {},
    }
    for par, pf in fits.items():
        models = {}
        parts_all = sorted({k[2] for k in by if k[0] == par})
        for model, info in pf["models"].items():
            keys = [(par, model, p) for p in parts_all if (par, model, p) in by]
            m: dict[str, Any] = {
                "status": info["status"],
                "note": info["note"],
                "params": info["params"],
                "lapse": info["lapse"],
                "n_grid": info["n_grid"],
                "grid_edges": info["grid_edges"],
                "select_objective": info["select_objective"],
                "cpu_seconds": info["cpu_seconds"],
                "param_count": info["param_count"],
            }
            if keys:
                nll = np.array([by[k]["nll_per_choice"] for k in keys])
                mean, lo, hi, _ = bootstrap_mean(nll, nb, seed, par, model)
                m.update(
                    n_participants=len(keys),
                    nll_mean=mean,
                    nll_ci_low=lo,
                    nll_ci_high=hi,
                    minus_chance=mean - CHANCE[par],
                    top1_mean=float(np.mean([by[k]["top1_accuracy"] for k in keys])),
                    ece_mean=float(np.mean([by[k]["ece"] for k in keys])),
                    nll_by_quarter=[
                        float(np.nanmean([by[k].get(f"nll_quarter_{q}", np.nan) for k in keys]))
                        for q in (1, 2, 3, 4)
                    ],
                    by_experiment={
                        exp: float(
                            np.mean([by[k]["nll_per_choice"] for k in keys if k[2].startswith(exp + "#")])
                        )
                        for exp in pf["experiments"]
                        if any(k[2].startswith(exp + "#") for k in keys)
                    },
                )
            models[model] = m
        # subgroup by predictability tercile (participant's NLL under ref_bias, lower = more predictable)
        bias_keys = {k[2]: by[k]["nll_per_choice"] for k in by if k[0] == par and k[1] == "ref_bias"}
        terciles: dict[str, dict[str, float]] = {}
        if bias_keys:
            vals = np.array(list(bias_keys.values()))
            q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
            group = {
                p: ("most_predictable" if v <= q1 else "middle" if v <= q2 else "least_predictable")
                for p, v in bias_keys.items()
            }
            for model in models:
                if models[model].get("n_participants"):
                    terciles[model] = {
                        g: float(
                            np.mean(
                                [
                                    by[(par, model, p)]["nll_per_choice"]
                                    for p in bias_keys
                                    if group[p] == g and (par, model, p) in by
                                ]
                            )
                        )
                        for g in ("most_predictable", "middle", "least_predictable")
                    }
        ok = {m: v for m, v in models.items() if v.get("n_participants")}
        best_ref = min(
            (m for m in ok if m in REFERENCE_MODELS2), key=lambda m: ok[m]["nll_mean"], default=None
        )
        best_non_dev = min(
            (m for m in ok if m != "developmental"), key=lambda m: ok[m]["nll_mean"], default=None
        )
        contrasts = []
        for name, a, b in (
            ("developmental_vs_best_reference", "developmental", best_ref),
            ("developmental_vs_best_non_developmental", "developmental", best_non_dev),
            ("best_reference_vs_ref_bias", best_ref, "ref_bias"),
        ):
            if isinstance(a, str) and isinstance(b, str) and a in ok and b in ok and a != b:
                common = [p for p in parts_all if (par, a, p) in by and (par, b, p) in by]
                d = np.array(
                    [by[(par, a, p)]["nll_per_choice"] - by[(par, b, p)]["nll_per_choice"] for p in common]
                )
                est, lo, hi, pv = bootstrap_mean(d, nb, seed, par, name)
                contrasts.append(
                    {
                        "contrast": name,
                        "first": a,
                        "second": b,
                        "estimate": est,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_value": pv,
                        "n": len(common),
                        "dz": float(est / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else None,
                    }
                )
        summary["paradigms"][par] = {
            "role": pf["role"],
            "experiments": pf["experiments"],
            "n": pf["n"],
            "chance_nll": CHANCE[par],
            "models": models,
            "terciles": terciles,
            "best_reference": best_ref,
            "best_non_developmental": best_non_dev,
            "contrasts": contrasts,
        }
    return summary


def finalize_v2(outdir: Path) -> dict[str, Any]:
    summary = analyze_v2(outdir)
    write_json(outdir / "summary.json", summary)
    defn = HumanV2Definition.model_validate(read_json(outdir / "definition.json"))
    (outdir / "report.md").write_text(render_v2(outdir, defn, summary), encoding="utf-8")
    return {"summary": summary}


def _f(x: Any, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_v2(outdir: Path, defn: HumanV2Definition, summary: dict[str, Any]) -> str:
    info = read_json(outdir / "run_info.json")
    val = read_json(outdir / "validation.json")
    label = (
        "EXPLORATORY (dev/tune partitions; selection-optimistic)"
        if defn.report_partition != "test"
        else "CONFIRMATORY (frozen test partition)"
    )
    L = [
        f"# {defn.experiment_id} — {defn.title}",
        "",
        f"> **{label}.** Real human data (Psych-101, Apache-2.0). Held-out participants per partition; teacher-forced per-choice NLL. Not a validation of the simulator and not evidence about any individual's future.",
        "",
        f"* Preregistration hash: `{defn.preregistration_hash}`; partitions `{defn.partitions_file}` (test sha256 `{read_json(outdir / 'partitions_ref.json')['test_partition_sha256'][:16]}…`)",
        f"* Fit on **{defn.fit_partition}**, select on **{defn.select_partition}**, report on **{defn.report_partition}**; code `{info['code_version']}`; wall {info['wall_time_s']} s; CPU {info['cpu_time_s']} s; LLM calls 0",
        f"* Registered policy classes: {', '.join(info['registered_policy_classes'])}",
        "",
        "## Hypothesis and falsification",
        "",
        defn.hypothesis,
        "",
        f"Falsification: {defn.falsification_criterion}",
        "",
    ]
    for par, s in summary["paradigms"].items():
        L += [
            f"## {par} ({s['role']}) — {', '.join(s['experiments'])}; n = {s['n']}",
            "",
            f"Chance NLL = {s['chance_nll']:.4f}. Best reference: `{s['best_reference']}`; best non-developmental: `{s['best_non_developmental']}`.",
            "",
            "| model | status | report NLL [95% CI] | vs chance | top-1 | ECE | NLL by quarter (Q1→Q4) | params | fitted | lapse | grid edges | CPU s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for model, m in sorted(s["models"].items(), key=lambda kv: kv[1].get("nll_mean", 99)):
            if m.get("n_participants"):
                q = " / ".join(_f(x, 3) for x in m["nll_by_quarter"])
                L.append(
                    f"| {model} | ok | {_f(m['nll_mean'])} [{_f(m['nll_ci_low'])}, {_f(m['nll_ci_high'])}] | {m['minus_chance']:+.4f} | {_f(m['top1_mean'], 3)} | {_f(m['ece_mean'], 3)} | {q} | {m['param_count'] if m['param_count'] is not None else '—'} | `{m['params']}` | {m['lapse']} | {', '.join(m['grid_edges']) or '—'} | {m['cpu_seconds']} |"
                )
            else:
                L.append(
                    f"| {model} | {m['status']} | — | — | — | — | — | — | {m['note'][:80]} | — | — | {m['cpu_seconds']} |"
                )
        L.append("")
        if s["contrasts"]:
            L += [
                "| contrast (NLL difference; negative favours first) | estimate | 95% CI | p | dz | n |",
                "|---|---|---|---|---|---|",
            ]
            for c in s["contrasts"]:
                L.append(
                    f"| {c['contrast']} ({c['first']} − {c['second']}) | {c['estimate']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_value']:.3g} | {_f(c['dz'], 2)} | {c['n']} |"
                )
            L.append("")
        exps = s["experiments"]
        if any(m.get("by_experiment") for m in s["models"].values()):
            L += [
                "Robustness — report NLL by experiment file:",
                "",
                "| model | " + " | ".join(exps) + " |",
                "|---|" + "---|" * len(exps),
            ]
            for model, m in sorted(s["models"].items(), key=lambda kv: kv[1].get("nll_mean", 99)):
                if m.get("by_experiment"):
                    L.append(
                        f"| {model} | " + " | ".join(_f(m["by_experiment"].get(e), 3) for e in exps) + " |"
                    )
            L.append("")
        if s["terciles"]:
            L += [
                "Subgroups — report NLL by participant predictability tercile (tercile of NLL under `ref_bias`):",
                "",
                "| model | most predictable | middle | least predictable |",
                "|---|---|---|---|",
            ]
            for model, t in s["terciles"].items():
                L.append(
                    f"| {model} | {_f(t['most_predictable'], 3)} | {_f(t['middle'], 3)} | {_f(t['least_predictable'], 3)} |"
                )
            L.append("")
        v = val["paradigms"].get(par, {})
        L += [
            "Parse validation (partitions used): ",
            "",
            "```json",
            json.dumps(
                {
                    k: {
                        "participants": vv.get("participants"),
                        "excluded": vv.get("excluded"),
                        "ambiguous_choices_skipped": vv.get("ambiguous_choices_skipped"),
                    }
                    for k, vv in v.items()
                },
                indent=1,
            ),
            "```",
            "",
        ]
    L += [
        "## Validity notes",
        "",
        *[f"* {n}" for n in defn.validity_notes],
        "",
        f"Regenerate from logs: `python -m hpe.evaluation.human analyze2 {outdir}`",
        "",
    ]
    return "\n".join(L)
