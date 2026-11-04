"""Render ``summary.json`` of an affect-lane exploratory run as ``report.md`` (no new computation).

python -m hpe.population.affect.report reports/phase3/affect/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _f(v: Any, nd: int = 4) -> str:
    return f"{float(v):+.{nd}f}" if isinstance(v, int | float) else str(v)


def render(summary: dict[str, Any]) -> str:
    d = summary["data"]
    lines = [
        f"# {summary['run_id']} — {summary['status']}",
        "",
        f"Dataset `{d['dataset']}` ({d['source'].get('licence', '')}); partition `{d['partition_file']}`.",
        f"Fit: {d['n_fit']} dev participants ({d['fit_choices']} choices). "
        f"Selection and report: {d['n_tune_scored']} tune participants ({d['tune_choices']} choices). "
        f"Code `{summary['code_version']}`, seed {summary['seed']}, threads {summary['threads']}.",
        "",
        "| model | selected config | λ | tune NLL/choice | top-1 | ECE | params | CPU-s (all configs) | causal violations |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary["models"].items():
        cfg = {k: v for k, v in m["selected_config"].items() if k in ("hidden", "lr")}
        lines.append(
            f"| `{name}` | {cfg} | {m['selected_lapse']} | {m['tune_mean_nll']:.5f} | {m['tune_top1']:.4f} | "
            f"{m['tune_ece']:.4f} | {m['params']} | {m['model_cpu_s']} | {len(m['causal_violations'])} |"
        )
    lines += [
        "",
        "Per configuration (fit log):",
        "",
        "| model | config | epochs | stop | best val NLL | fit CPU-s | tune NLL at selected λ |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in summary["models"].items():
        for r in m["configs"]:
            if "skipped" in r:
                lines.append(f"| `{name}` | {r['config']} | — | {r['skipped']} | — | — | — |")
                continue
            fl = r["fit_log"]
            lam = str(r["best_lapse"])
            lines.append(
                f"| `{name}` | hidden={r['config'].get('hidden')} | {fl.get('epochs')} | {fl.get('stopped')} | "
                f"{fl.get('best_val_nll', float('nan')):.5f} | {r['fit_cpu_s']} | {r['tune_nll_by_lapse'][lam]:.5f} |"
            )
    sub = summary.get("subset", "choices right after a new rating")
    lines += [
        "",
        f"Paired contrasts on tune (a − b, nats/choice; negative favours a; 95 % bootstrap CI). Subset: {sub}.",
        "",
        "| a | b | question | all choices | subset | other choices | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in summary["contrasts"]:
        a = c["all_choices"]
        s = c.get("choices_in_subset") or c.get("choices_right_after_a_rating")
        o = c["other_choices"]
        lines.append(
            f"| `{c['a']}` | `{c['b']}` | {c['label']} | {_f(a['estimate'])} [{_f(a['ci_low'])}, {_f(a['ci_high'])}] "
            f"(sd {a['sd']:.4f}) | {_f(s['estimate'])} [{_f(s['ci_low'])}, {_f(s['ci_high'])}] | "
            f"{_f(o['estimate'])} [{_f(o['ci_low'])}, {_f(o['ci_high'])}] | {a['n']} |"
        )
    lines += ["", "Exploratory: dev/tune only; no gate can pass on these numbers.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    out = Path((argv or sys.argv[1:])[0])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    (out / "report.md").write_text(render(summary), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
