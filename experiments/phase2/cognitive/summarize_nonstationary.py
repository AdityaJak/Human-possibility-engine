"""Recovery after a regime switch (v2 `nonstationary`), from the committed synthetic_dev results.

Assessments happen every 50 episodes; the rule switches after 150 and 300 training samples, so the
curve points are: 100 = before the first switch, 150 = immediately after it (new regime, old policy),
200/250 = recovery, 300 = immediately after the second switch. The runner's "final" value is therefore
a post-switch drop, not an end state; this script reports the informative points. Exploratory.
"""

import json
from pathlib import Path

import numpy as np

res = json.loads(Path("reports/phase2/cognitive/synthetic_dev/results.json").read_text())
seeds = [str(s) for s in res["protocol"]["seeds"]]
out = {}
print("| world | arm | before switch (ep 100) | right after (ep 150) | recovery (ep 250) | drop | regained |")
print("|---|---|---|---|---|---|---|")
for world, arms in res["raw"].items():
    for arm, by_seed in arms.items():
        curves = np.array([by_seed[s][a]["nonstationary"] for s in seeds for a in by_seed[s]])  # (n, 7)
        before, after, rec = curves[:, 2].mean(), curves[:, 3].mean(), curves[:, 5].mean()
        out[(world, arm)] = (before, after, rec)
        print(
            f"| {world} | {arm} | {before:.3f} | {after:.3f} | {rec:.3f} | {after - before:+.3f} | {rec - after:+.3f} |"
        )
Path("reports/phase2/cognitive/synthetic_dev/nonstationary_recovery.md").write_text(
    "# nonstationary: recovery after a regime switch (exploratory)\n\n"
    + "| world | arm | before (ep 100) | after switch (ep 150) | recovery (ep 250) | drop | regained |\n|---|---|---|---|---|---|---|\n"
    + "\n".join(
        f"| {w} | {a} | {b:.3f} | {s:.3f} | {r:.3f} | {s - b:+.3f} | {r - s:+.3f} |"
        for (w, a), (b, s, r) in out.items()
    )
    + "\n"
)
