"""Render results/summary_{phase}.json as Markdown tables (pasted into TRANSFER_RESULTS.md).

    python make_report.py confirm > ../results/report_confirm.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"
CONDS = ("iid", "cue_reversed", "recombined", "novel_role", "role_switch")
AGENTS = ("ira", "metarl", "metarl_s0", "metarl_s1", "ira_noactive", "ira_closed", "ira_nocuetrack", "ira_nochange",
          "ira_nolibrary", "entity_bayes", "ucb", "random")


def fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else "<1e-4"


def main(phase: str) -> None:
    s = json.loads((OUT / f"summary_{phase}.json").read_text())
    t = s["table"]
    print(f"### Normalized expected return ({phase}; universes {s['universes']}; n per cell shown in the first row)\n")
    print("| agent | " + " | ".join(CONDS) + " |")
    print("|---|" + "---|" * len(CONDS))
    for ag in AGENTS:
        if ag in t:
            print(f"| `{ag}` | " + " | ".join(f"{t[ag][c]['mean']:.3f}" if c in t[ag] else "—" for c in CONDS) + " |")
    print(f"\n(n = {t['ira']['iid']['n']} worlds per cell.)\n")
    print("### Primary contrasts: IRA − meta-RL (mean of two seeds), paired over worlds\n")
    print("| condition | estimate | 95% CI | per universe | p (one-sided) | Holm p | passes (≥ 0.10, Holm < 0.05, all universes > 0) |")
    print("|---|---|---|---|---|---|---|")
    for c in CONDS:
        r = s["primary"]["contrasts"][c]
        pu = ", ".join(f"{v:+.3f}" for v in r["per_universe"].values())
        holm = fmt_p(r["p_holm"]) if "p_holm" in r else "—"
        passes = ("**yes**" if r["passes"] else "no") if "passes" in r else "(non-inferiority: ≥ −0.05)"
        print(f"| {c} | {r['estimate']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {pu} | {fmt_p(r['p_one_sided'])} | {holm} | {passes} |")
    p = s["primary"]
    print(f"\nShift conditions passed: **{p['shift_conditions_passed']} of 4**; IRA non-inferior in-distribution: "
          f"**{p['iid_noninferior']}**; **H1 supported: {p['H1_supported']}**.\n")
    print("### Mechanism attributions (IRA − ablation; Holm over M1–M6)\n")
    print("| mechanism | comparator | condition(s) | margin | estimate | 95% CI | per universe | Holm p | passes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in s["mechanisms"]:
        pu = ", ".join(f"{v:+.3f}" for v in r["per_universe"].values())
        print(f"| {r['mechanism']} | `{r['comparator']}` | {', '.join(r['conditions'])} | {r['margin']} | {r['estimate']:+.4f} | "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {pu} | {fmt_p(r['p_holm'])} | {'**yes**' if r['passes'] else 'no'} |")
    if "accumulation" in s:
        a = s["accumulation"]
        pu = ", ".join(f"{v:+.3f}" for v in a["per_universe"].values())
        print(f"\n### Accumulation without retraining (last 25 of 50 novel-role worlds per universe)\n\n"
              f"accumulating − frozen = **{a['estimate']:+.4f}** [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}], per universe {pu}, "
              f"p = {fmt_p(a['p_one_sided'])}, n = {a['n']}; **passes: {a['passes']}** (margin 0.05).\n")
    print("### Diagnostics\n")
    for k, v in s["diagnostics"].items():
        print(f"* {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "confirm")
