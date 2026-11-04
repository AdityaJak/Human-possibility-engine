"""Render results_r1/summary_{phase}.json (revision R1, universe-level analysis) as Markdown.

python make_report_r1.py confirm > ../results_r1/report_confirm.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results_r1"
CONDS = ("iid", "cue_reversed", "recombined", "novel_role", "role_switch")


def fp(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else "<1e-4"


def pu(r: dict) -> str:
    return ", ".join(f"{v:+.3f}" for v in r["per_universe"].values())


def main(phase: str) -> None:
    s = json.loads((OUT / f"summary_{phase}.json").read_text())
    print(f"### Normalized expected return ({phase}; universes {s['universes']})\n")
    print("| agent | " + " | ".join(CONDS) + " |")
    print("|---|" + "---|" * len(CONDS))
    for ag, row in s["table"].items():
        print(f"| `{ag}` | " + " | ".join(f"{row[c]:.3f}" if c in row else "—" for c in CONDS) + " |")
    for comp, v in s["primary"].items():
        print(f"\n### H1: IRA − `{comp}` (universe-level; Holm over the 4 shift conditions)\n")
        if "status" in v:
            print(v["status"])
            continue
        print(
            "| condition | estimate | per universe | universes > 0 | t | p | Holm p | passes | world-level 95% CI |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for c in CONDS:
            r = v["contrasts"][c]
            holm = fp(r["p_holm"]) if "p_holm" in r else "—"
            passes = ("**yes**" if r["passes"] else "no") if "passes" in r else "non-inferiority ≥ −0.05"
            ci = r["world_level_ci95"]
            print(
                f"| {c} | {r['estimate']:+.4f} | {pu(r)} | {r['n_universes_positive']} | {r['t']} | "
                f"{fp(r['p_universe_t'])} | {holm} | {passes} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |"
            )
        print(
            f"\nShift conditions passed: **{v['shift_conditions_passed']} of 4**; iid non-inferior: "
            f"**{v['iid_noninferior']}**; H1 against `{comp}`: **{v['H1_vs_this_comparator']}**"
        )
    print(f"\n**H1 supported: {s['H1_supported']}**\n")
    print("### Mechanisms (IRA − ablation; Holm over M1–M5)\n")
    print("| mechanism | ablation | condition(s) | margin | estimate | per universe | Holm p | passes |")
    print("|---|---|---|---|---|---|---|---|")
    for r in s["mechanisms"]:
        print(
            f"| {r['mechanism']} | `{r['comparator']}` | {', '.join(r['conditions'])} | {r['margin']} | "
            f"{r['estimate']:+.4f} | {pu(r)} | {fp(r['p_holm'])} | {'**yes**' if r['passes'] else 'no'} |"
        )
    if "accumulation" in s:
        a = s["accumulation"]
        print(
            f"\n**Accumulation (A):** accumulating − frozen (last 25 of 50 unseen-role worlds) = {a['estimate']:+.4f}, "
            f"per universe {pu(a)}, p = {fp(a['p_universe_t'])}; passes: **{a['passes']}** (margin 0.05).\n"
        )
    print("### Diagnostics\n")
    for k, v in s["diagnostics"].items():
        print(f"* {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "confirm")
