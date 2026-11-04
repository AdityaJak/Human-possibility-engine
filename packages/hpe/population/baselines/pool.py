"""Aggregate the comparator pool across committed result directories (tune partition, exploratory).

    python -m hpe.population.baselines.pool reports/phase3/baselines/x3_b01_pop_rnn reports/phase3/baselines/x3_b02_restless4 ... \
        --out reports/phase3/baselines/pool

For every family: each model's tune NLL (main variant and, for adaptive models, ``[frozen]`` / ``[mismatched]``),
the best comparator (lowest selection NLL among main comparator variants; selection = report = tune here),
paired contrasts of the best comparator against every other pool member, and every adaptive model against its own
frozen and mismatched variants (Gate B1/A2-style **exploratory** contrasts; no gate is evaluated). All result
directories must share the seed and report partition so that participants and bootstrap streams match.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hpe.population.core import evaluate


def _load(
    outdir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    defn = json.loads((outdir / "definition.json").read_text())
    parts: dict[str, dict[str, Any]] = {}
    for p in sorted((outdir / "parts").glob("*.json")):
        r = json.loads(p.read_text())
        parts[f"{r['family']}::{r['model']}"] = r
    per: dict[tuple[str, str], dict[str, float]] = {}
    with gzip.open(outdir / "outcomes.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            per.setdefault((row["family"], row["model"]), {})[row["key"]] = float(row["nll"])
    return defn, parts, per


def _scores(d: dict[str, float]) -> list[evaluate.ParticipantScore]:
    return [evaluate.ParticipantScore(k, v, 1, 0.0, 0.0, (v, v, v, v)) for k, v in sorted(d.items())]  # type: ignore[arg-type]


def aggregate(dirs: list[Path]) -> dict[str, Any]:
    seeds, partitions = set(), set()
    fams: dict[str, dict[str, Any]] = {}
    per_all: dict[tuple[str, str], dict[str, float]] = {}
    for d in dirs:
        defn, parts, per = _load(d)
        seeds.add(defn["seed"])
        partitions.add(defn["report_partition"])
        summ = json.loads((d / "summary.json").read_text())
        for key, r in sorted(parts.items()):
            fam, model = key.split("::")
            fsum = summ["families"].get(fam, {}).get("models", {})
            for var in r["variants"]:
                label = model if var == "main" else f"{model}[{var}]"
                if label in fams.get(fam, {}).get("models", {}):
                    raise ValueError(f"{fam}/{label} appears in two result directories")
                m = fsum.get(label, {})
                fams.setdefault(fam, {"models": {}})["models"][label] = {
                    "source": str(d),
                    "experiment_id": defn["experiment_id"],
                    "model": model,
                    "variant": var,
                    "role": r["role"],
                    "adaptive": r["adaptive"],
                    "select_nll": r["selected"]["select_nll"],
                    "report_nll": m.get("report_nll"),
                    "ci": m.get("ci"),
                    "n": m.get("n"),
                    "top1": m.get("top1"),
                    "ece": m.get("ece"),
                    "params": r["params"],
                    "per_person_params": r.get("describe", {}).get("per_person_params"),
                    "config": r["selected"]["config"],
                    "lapse": r["selected"]["lapse"],
                    "grid_edges": r["selected"]["grid_edges"],
                    "n_configs": len(r["fits"]),
                    "n_skipped": sum(1 for f in r["fits"] if f.get("skipped")),
                    "fit_cpu_s": round(sum(f.get("fit_cpu_s", 0.0) for f in r["fits"]), 1),
                    "report_cpu_s_all_variants": r.get("report_cpu_s"),
                    "n_variants": len(r["variants"]),
                    "causal": not r["causality_violations"],
                }
                per_all[(fam, label)] = per.get((fam, label), {})
    if len(seeds) > 1 or len(partitions) > 1:
        raise ValueError(f"result directories disagree on seed {seeds} or report partition {partitions}")
    seed = int(next(iter(seeds)))
    out: dict[str, Any] = {"seed": seed, "report_partition": next(iter(partitions)), "families": {}}
    for fam, fd in sorted(fams.items()):
        models = fd["models"]
        mains = [
            lab
            for lab, m in models.items()
            if m["variant"] == "main" and m["role"] == "comparator" and m["causal"]
        ]
        best = min(mains, key=lambda lab: (models[lab]["select_nll"], lab)) if mains else None
        contrasts = []
        if best:
            for lab in sorted(mains):
                if lab == best:
                    continue
                c = evaluate.paired(
                    _scores(per_all[(fam, best)]),
                    _scores(per_all[(fam, lab)]),
                    seed=seed,
                    key=(fam, best, lab),
                )
                contrasts.append({"first": best, "second": lab, "kind": "best_vs_other", **c})
        for lab, m in sorted(models.items()):
            if m["variant"] != "main" or not m["adaptive"]:
                continue
            for v in ("frozen", "mismatched"):
                other = f"{lab}[{v}]"
                if other in models:
                    c = evaluate.paired(
                        _scores(per_all[(fam, lab)]),
                        _scores(per_all[(fam, other)]),
                        seed=seed,
                        key=(fam, lab, other),
                    )
                    contrasts.append({"first": lab, "second": other, "kind": f"adapt_vs_{v}", **c})
        out["families"][fam] = {"best_comparator": best, "models": models, "contrasts": contrasts}
    return out


def _f(x: Any, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _per(m: dict[str, Any]) -> str:
    c, n, v = m.get("report_cpu_s_all_variants"), m.get("n"), m.get("n_variants") or 1
    return "—" if c is None or not n else f"{c / (n * v):.3f}"


def render(res: dict[str, Any]) -> str:
    lines = [
        "# Phase III comparator pool — tune partition (exploratory, selection-optimistic)",
        "",
        f"Seed {res['seed']}; report partition `{res['report_partition']}` (= selection partition). Real human data "
        "(Psych-101, Apache-2.0); per-participant mean NLL per free choice (nats), lapse selected on tune. "
        "No gate is evaluated here.",
        "",
    ]
    for fam, fd in res["families"].items():
        lines += [f"## {fam} — best comparator on tune: `{fd['best_comparator']}`", ""]
        lines += [
            "| model | tune NLL [95% CI] | n | top-1 | params (pop / person) | configs (skipped) | lapse | grid edges | fit CPU s | predict CPU s / participant / variant | causal | source |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for lab, m in sorted(fd["models"].items(), key=lambda kv: (kv[1]["report_nll"] or 9.0, kv[0])):
            ci = m["ci"] or [None, None]
            lines.append(
                f"| {lab} | {_f(m['report_nll'])} [{_f(ci[0])}, {_f(ci[1])}] | {m['n']} | {_f(m['top1'], 3)} | "
                f"{m['params']} / {m['per_person_params'] if m['per_person_params'] is not None else '—'} | "
                f"{m['n_configs']} ({m['n_skipped']}) | {m['lapse']} | {', '.join(m['grid_edges']) or '—'} | "
                f"{m['fit_cpu_s']} | {_per(m)} | {'ok' if m['causal'] else 'VIOLATION'} | `{m['experiment_id']}` |"
            )
        if fd["contrasts"]:
            lines += [
                "",
                "| contrast (negative favours first) | kind | estimate | 95% CI | one-sided p | n |",
                "|---|---|---|---|---|---|",
            ]
            for c in fd["contrasts"]:
                lines.append(
                    f"| {c['first']} − {c['second']} | {c['kind']} | {c['estimate']:+.4f} | "
                    f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_one_sided']:.4f} | {c['n']} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hpe.population.baselines.pool")
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/phase3/baselines/pool"))
    args = ap.parse_args(argv)
    res = aggregate([d for d in args.dirs if (d / "summary.json").exists()])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pool_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    (args.out / "POOL.md").write_text(render(res))
    for fam, fd in res["families"].items():
        print(f"{fam}: best comparator {fd['best_comparator']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
