# Risks

| # | Risk | Likelihood / impact | Current mitigation | Open |
|---|---|---|---|---|
| R1 | **The core hypothesis is wrong**: developmental mechanisms add nothing over simple online learners on realistic tasks. H1 already failed on linear tasks. | medium / critical | Preregistered negative result published; NEXT_STEPS §1 redesigns tasks (non-linear, non-stationary) and components | Yes — this is the research programme |
| R2 | Simulator realism is mistaken for human evidence (by us, users or investors) | medium / high | Claims policy, synthetic disclaimer on every result, Gate 4 unpassed and shown as such, UI zone separation | Human validation of the learner pending (Exp 7 rerun) |
| R3 | Scripted assumptions (age gates, schedules) masquerade as development | medium / high | Same gates for all classes; ablations; sensitivity variants | Lifespan curves still partly scripted |
| R4 | Tuning leakage inflates results | low / medium | Audit recomputed without the overlapping seed; baselines tuned on separate seeds | Commit tuning scripts; re-tune on seeds ≥1000 |
| R5 | Future Self gives harmful advice or false certainty | low / critical | No probabilities (validator), consent gate, crisis and high-stakes routing (audit-tested), reversible experiments only, no autonomous actions | Keyword classifier is not a risk detector; needs clinical review before public exposure |
| R6 | Real-user data handled before controls exist | low / high | Demo users only; local SQLite; export/delete verified; no auth yet, documented in SECURITY.md | Auth, encryption at rest, retention before any real data |
| R7 | External data licence or consent problems | low / high | Licences verified and recorded (DATA_SOURCES.md); no raw data committed; DfE-DB excluded | Re-verify terms on each new dataset |
| R8 | Scale claims outrun evidence | medium / medium | Ledger marks 1k/10k as not tested; benchmarks disclose hardware and load | Run benchmarks on a quiet machine |
| R9 | Reproducibility drift (code versions, dirty trees during experiments) | medium / low | Manifests record code version; re-runs bit-identical so far | Pin experiments to clean commits (audit M-3) |
| R10 | Accessibility gaps exclude users | medium / medium | axe audit run; no critical violations | Fix light-mode contrast and keyboard focus on scrollable regions (audit M-6) |
| R11 | Key-person / single-session knowledge loss | medium / medium | AGENTS.md, ADRs, BUILD_STATUS, task board, generated ledger | — |
| R12 | Reputational harm from over-claiming in investor material | low / high | Every number traces to an artifact; failed tests shown first | Keep it that way |
