"""Ablation contrasts for design-team runs (reads the runner's checkpointed parts; no refitting).

    python -m hpe.population.candidates.analysis reports/phase3/candidates/<experiment_id> [...]

Writes ``ablations.json`` and ``ablations.md`` next to the runner's ``report.md``. Every contrast is a
paired participant-level difference (first − second; negative favours the first) with the core
bootstrap (``core.evaluate.paired``). CPU cost includes the shared backbone even when it was cached.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hpe.population.core import evaluate
from hpe.population.core.runner import _load_parts, _scores_from_json

PAIRS: list[tuple[str, str, str]] = [
    # (first, second, question)
    ("f1_latent", "pop_gru", "F1 vs same-size population GRU"),
    ("f1_latent", "f1_latent[frozen]", "F1 adaptation on vs off (B1-style)"),
    ("f1_latent", "f1_latent[mismatched]", "F1 own vs donor history (A2-style)"),
    ("f1_latent", "f1_trait", "split latent vs trait only"),
    ("f1_latent", "f1_context", "split latent vs context only"),
    ("f1_latent", "f1_fixedstep", "surprise gating vs fixed step"),
    ("f1_latent", "f1_bolt", "meta-trained latent vs bolted-on latent"),
    ("f1_trait", "f1_trait[frozen]", "trait-only adaptation on vs off"),
    ("f1_bolt", "f1_bolt[frozen]", "bolted-on latent on vs off"),
    ("f3_arbiter", "pop_gru", "F3 vs same-size population GRU"),
    ("f3_arbiter", "f3_arbiter[frozen]", "F3 episodic memory on vs off (B1-style)"),
    ("f3_arbiter", "f3_arbiter[mismatched]", "F3 own vs donor memory (A2-style)"),
    ("f3_arbiter", "f3_fixedmix", "learned gate vs constant mixture (tuned on tune)"),
    ("f3_arbiter", "f3_fixedshare", "learned gate vs fixed-share Bayesian mixture"),
    ("f3_arbiter", "f3_gruepi", "learned gate vs GRU fed p_epi"),
    ("f3_fixedshare", "f3_fixedshare[frozen]", "fixed-share memory on vs off"),
    ("f3_gruepi", "pop_gru", "GRU fed p_epi vs same-size population GRU"),
    ("f2_world_policy", "pop_gru", "F2 vs same-size population GRU"),
    ("f2_world_policy", "f2_world_policy[frozen]", "individual vs population-only policy (B1-style)"),
    ("f2_world_policy", "f2_world_policy[mismatched]", "own vs donor posterior (A2-style)"),
    ("f2_world_policy", "f2_delta", "learned world model vs delta rule"),
    ("f2_delta", "f2_delta[frozen]", "delta-rule individual vs population policy"),
]


def _cpu(r: dict[str, Any]) -> float:
    tot = 0.0
    for f in r["fits"]:
        tot += float(f.get("fit_cpu_s", 0.0))
        for part, cpu_key in (("backbone", "backbone_cpu_s"), ("world_model", "world_cpu_s")):
            bb = (f.get("fit_log") or {}).get(part)
            if isinstance(bb, dict) and bb.get("cached"):
                tot += float(bb.get(cpu_key, 0.0))
    return round(tot, 1)


def analyze(outdir: Path) -> dict[str, Any]:
    raw = json.loads((outdir / "definition.json").read_text())
    seed = int(raw["seed"])
    parts = _load_parts(outdir)
    out: dict[str, Any] = {"experiment_id": raw["experiment_id"], "families": {}}
    for family, models in parts.items():
        scores: dict[str, list[evaluate.ParticipantScore]] = {}
        table: dict[str, Any] = {}
        for name, r in models.items():
            for var, sc in r["variants"].items():
                label = name if var == "main" else f"{name}[{var}]"
                s = _scores_from_json(sc)
                scores[label] = s
                nll = np.array([x.nll for x in s])
                m, lo, hi, _ = evaluate.bootstrap(nll, seed=seed, key=(family, label, "mean"))
                table[label] = {
                    "tune_nll": m,
                    "ci": [lo, hi],
                    "n": len(s),
                    "params": r["params"],
                    "per_person_params": (r.get("describe") or {}).get("per_person_params"),
                    "config": r["selected"]["config"],
                    "lapse": r["selected"]["lapse"],
                    "grid_edges": r["selected"]["grid_edges"],
                    "fit_cpu_s_incl_backbone": _cpu(r),
                    "n_configs": len(r["fits"]),
                    "causal": not r["causality_violations"],
                }
        contrasts = []
        for a, b, q in PAIRS:
            if a in scores and b in scores:
                c = evaluate.paired(scores[a], scores[b], seed=seed, key=(family, a, b, "ablation"))
                contrasts.append({"first": a, "second": b, "question": q, **c})
        out["families"][family] = {"models": table, "contrasts": contrasts}
    (outdir / "ablations.json").write_text(json.dumps(out, indent=1) + "\n")
    (outdir / "ablations.md").write_text(render(out))
    return out


def render(out: dict[str, Any]) -> str:
    lines = [
        f"# {out['experiment_id']} — candidate ablations (tune; exploratory)",
        "",
        "Paired per-participant NLL differences (first − second; negative favours the first), 95 % percentile "
        "bootstrap CI, one-sided p. Selection and report are both on tune: optimistic, no gate is evaluated.",
        "",
    ]
    for fam, d in out["families"].items():
        lines += [
            f"## {fam}",
            "",
            "| model | tune NLL [95% CI] | n | params | per-person | configs | fit CPU s (incl. backbone) | lapse | causal |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, m in sorted(d["models"].items(), key=lambda kv: kv[1]["tune_nll"]):
            lines.append(
                f"| {name} | {m['tune_nll']:.4f} [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}] | {m['n']} | {m['params']} | "
                f"{m['per_person_params']} | {m['n_configs']} | {m['fit_cpu_s_incl_backbone']} | {m['lapse']} | "
                f"{'ok' if m['causal'] else 'VIOLATION'} |"
            )
        lines += [
            "",
            "| first − second | question | estimate | 95% CI | one-sided p | n |",
            "|---|---|---|---|---|---|",
        ]
        for c in d["contrasts"]:
            lines.append(
                f"| {c['first']} − {c['second']} | {c['question']} | {c['estimate']:+.4f} | "
                f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['p_one_sided']:.4f} | {c['n']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    for p in argv if argv is not None else sys.argv[1:]:
        analyze(Path(p))
        print(f"wrote {Path(p) / 'ablations.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
