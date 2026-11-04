# Dev finding D-1 (environment v1, before any confirmatory run)

Environment v1 (`env.py` at commit eaa2e37): 5 entities, glow-only pokes, reaction law density 0.3, worlds with ≥1
reacting pair. Universe 0, 60 worlds per condition, IRA with β ∈ {0.5, 1, 2}:

| library | iid | cue_reversed | recombined | novel_role | role_switch |
|---|---|---|---|---|---|
| empty (β=1) | 0.920 | 0.897 | 0.902 | 0.787 | 0.703 |
| oracle concepts = true roles (β=1) | 0.920 | 0.917 | 0.908 | 0.754 | 0.841 |
| UCB1 over pairs (40 worlds) | 0.703 | 0.667 | 0.659 | 0.406 | 0.579 |

(normalized expected return; script `env_v1_diag.py.txt`, runs against `u1/env.py` at commit eaa2e37; the UCB row is from the smoke run.)

**Conclusion:** in v1 an agent with *no* cross-world knowledge already retains 0.92 of the oracle in-distribution,
the same as with a perfect concept library — reacting pairs are so common that blind search finds one in ~1–2 steps.
The environment cannot distinguish knowledge-driven learners, so it was redesigned (v2: sparse laws with 1–2
reacting pairs per world, 3-bit probe signatures per role, a 3-dimensional appearance code that is valid in
training). v1 is preserved in git history.

A second v1 finding: **additive MAP consolidation collapsed the learned library** into one dominant concept (300
training worlds, concept sizes [1028, 41, 298], adjusted Rand index vs. true roles ≈ 0.00) and lowered IRA's
in-distribution score from 0.92 (empty library) to 0.84. Rich-get-richer membership prior + hard assignments under
weak per-world evidence. Whether v2's more informative probes avoid this is checked before anything else.
