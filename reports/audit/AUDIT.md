# Independent audit — Human Possibility Engine

- **Audited commit:** `3485924` (all lanes merged; learning-lab results, stored 100×80 run, human-validation results committed). Later commits checked where noted (`3da7504` evidence ledger, `73c20f4` investor demo doc, `89205d0` claims & next steps).
- **Date:** 2026-11-04. **Auditor:** `audit/review` lane. I did not author any model, experiment or result reviewed here. Machine-readable findings: [`findings.json`](findings.json). Scripts: [`scripts/`](scripts/).
- **Out of scope / not audited:** society experiments Exp2–Exp5 (still running in the main checkout, not in the audited commit); the exp07 rerun with all policy classes (running; the committed exp07 result is the reference-model-only run); Docker images (no daemon in this container).

## Executive summary

The research pipeline is **reproducible and honest**. Every committed learning-lab result regenerates byte-identically from raw logs. All 80 main-variant jobs of H1 and Exp6 re-run bit-identically from scratch at the audited commit, despite the manifests' mixed and dirty code versions. My independent statistics reproduce the headline numbers: **H1 failed** (+0.0121, 95% CI [−0.0081, +0.0313] vs a +0.0837 threshold) and **Exp6 passed**.

Excluding evaluation seed 7, which overlaps a developmental-lane tuning seed, changes no verdict:

- **H1:** +0.0135 [−0.0089, +0.0344], still failed.
- **Exp6:** +0.0896 [+0.0476, +0.1282], still passed.

I found no hard-coded outcomes or item-level leakage: policies never read task internals, and answers are only revealed under instructional feedback.

The most important problems are on the consumer side:

1. **Crisis routing misses indirect suicidal language (H-1).** "I don't see the point of going on anymore" gets 3 normal scenarios.
2. **Health and legal questions are not routed (M-1).** Examples: "being sued", "doctor… heart condition".
3. **No input-size limits (M-2).**

Accessibility has serious but fixable axe violations (M-6). External human validity (Gate 4) and 10k-agent scale (Gate 5) are not met.

**Recommendation:** GO for showing the research slice to investors, with the caveats in the claims ledger. **NO-GO for any public Future Self exposure** until H-1 and M-1 are fixed and re-tested.

## Gate verdicts

### Spec §52 integration gates

| Gate | Verdict | Evidence |
|---|---|---|
| 0 — monorepo, contracts, ownership, local run | **GO** | contracts check, `make demo-check` 14/14, AGENTS.md ownership |
| 1 — determinism, persistence, replay, coverage goal, no critical runtime issues | **GO** | re-analysis byte-identical (4/4); from-scratch re-run 80/80 bit-identical; fresh `verify-replay` passes from seed and from checkpoint; branch coverage 91% total (gaps: `core/hashing` 56%, `world/population` 74%) |
| 2 — measurable learning, train/eval separation, static/memory ablations, honest stats | **GO** | contamination 0; frozen policies never change; ablations reproduce; H1 failure reported as failure |
| 3 — web/API integration, cross-component flow, responsive demo, accessibility review, secrets | **CONDITIONAL GO** | 0 console/HTTP errors on 9 routes × 2 themes; 0 overflow at 390 px; secrets scan clean; **axe serious violations remain (M-6)** |
| 4 — external evidence: human-data provenance + suitable baseline comparison | **NO-GO (partial)** | licences and hashes verified; HeartSteps consistency "supported" but low-power (M-5); held-out human prediction with project policies not in committed results (inconclusive) |
| 5 — performance to 10,000 agents | **NO-GO** | only 100-agent benchmarks committed; 1k/10k `not_tested` in ledger |

### Spec §42 acceptance gates

| Gate | Verdict | Evidence |
|---|---|---|
| 1 — full stack locally; 100 agents seeded life-stage run with saved state and replay | **GO** | `make demo-check` 14/14; stored 100×80 run with checkpoints; replay verified by orchestrator and by fresh audit run |
| 2 — tests prove policy/parameters change from experience; held-out improvement; baselines | **GO** | conformance tests; `scripts/frozen_policies.py` (dev 0.50 → 0.90 held-out, frozen 0.50 → 0.50, static unchanged) |
| 3 — randomised simulated intervention reproduces, exports data + uncertainty, model-dependence flagged | **NOT VERIFIED** | Exp2 (primary twin-society) not in audited commit. Learning-lab Exp1 (randomised feedback conditions) reproduces with CIs and the synthetic disclaimer |
| 4 — Future Self functional, consent-based, no unsupported individualized predictions | **CONDITIONAL** | consent enforced; 0 probability phrases across 9 adversarial prompts; all scenarios labelled; **routing misses H-1, M-1** |
| 5 — CI green from clean checkout, no credentials, remote verified | **GO** | GitHub Actions run #13 (`89205d0`) success; secrets scan 0 findings |
| 6 — investor can follow a 5–7 min script and reproduce a stated result | **CONDITIONAL GO** | `/demo` route and `docs/INVESTOR_DEMO.md` exist (`73c20f4`); I reproduced H1 and Exp6 from scratch; I did not time a walkthrough |

## Findings

| ID | Sev | Area | Finding | File:line |
|---|---|---|---|---|
| H-1 | high | Future Self safety | Indirect suicidal ideation not routed to crisis; normal scenarios returned | `packages/hpe/futures/safety.py:15-28` |
| M-1 | medium | Future Self safety | "sued", "doctor", "heart condition" not routed (health/legal) | `packages/hpe/futures/safety.py:29-58` |
| M-2 | medium | Security/privacy | 200 KB question and 10K-char user_id accepted and persisted | `packages/hpe/contracts/schemas.py` (DecisionRequest) |
| M-3 | medium | Provenance | 403/460 manifests `+dirty`; H1 spans 6 code versions (mitigated: 80/80 bit-identical re-run) | `packages/hpe/evaluation/runner.py:65-75` |
| M-4 | medium | Validity | Dev tuned on task seed 7 ∈ eval seeds; tuning unlogged for both learners (verdicts unchanged without seed 7) | `packages/hpe/development/learner.py:40-43`, `packages/hpe/evaluation/learning_lab.py:385` |
| M-5 | medium | Validity (human data) | exp07b "consistent" although CI includes 0 → test would also pass a null | `reports/human_validation/exp07b_causal_grounding/report.md:6` |
| M-6 | medium | Accessibility | contrast (light mode, all routes), scroll regions not focusable, heading order | `apps/web` (`.chip--ok`, `.table-wrap`, timeline `ol`) |
| L-1 | low | Mechanism | held-out scores steer tutoring family/difficulty and hiring (aggregate only; lab results unaffected) | `packages/hpe/world/engine.py:600-605, 1010-1016` |
| L-2 | low | Security | unescaped `%`/`_` in LIKE agent filter (parameterised; read-only) | `packages/hpe/storage/repositories.py:150-151` |
| L-3 | low | Demo integrity | demo-check passes with 0 claims (ledger added later in `3da7504`) | `scripts/demo.py` |
| L-4 | low | Demo integrity | exp07/exp07b not served by `/experiments` | `apps/api/hpe_api/experiment_service.py` |
| L-5 | low | Privacy | audit log keeps pseudonymous user id after delete | `packages/hpe/storage/repositories.py` |
| L-6 | low | Testing | coverage gaps in hashing/population | — |

**Resolved after the audit snapshot:** the Evidence page was empty at `3485924` (`/research/claims` returned `[]`). Commit `3da7504` added `reports/evidence/claims.json` with 17 claims (8 supported, 6 not supported, 2 not tested, 1 inconclusive); its statuses match the result files I verified.

## What I re-ran (exact commands, actual outputs)

1. Re-analysis from raw logs (copies in a temp dir):

   `python reports/audit/scripts/reanalyze.py <tmp>`

   ```
   exp_baselines_heldout   rc=0 result_identical=True report_identical=True (3fe6b95ae478932e)
   exp01_acquired_learning rc=0 result_identical=True report_identical=True (8f19af2b189c24cb)
   exp06_transfer          rc=0 result_identical=True report_identical=True (3908b48539ba6d5c)
   exp_ablations           rc=0 result_identical=True report_identical=True (ecf5d38dab3bf2e8)
   ```

2. From-scratch re-run at `3485924`, all arms × 10 seeds, main variant:

   `python reports/audit/scripts/rerun_from_scratch.py <tmp> exp_baselines_heldout all all`

   ```
   SUMMARY exp_baselines_heldout: 40/40 main-variant jobs bit-identical to committed raw logs
   ```

   The same command for `exp06_transfer` gives **40/40** (263.7 s total job wall time).

3. `hpe run --agents 20 --years 4 --seed 11 --checkpoint-every 2` followed by `hpe verify-replay`: from_seed chain/state identical; from_checkpoint (tick 25) chain/state identical; `passed: true`.

4. `python reports/audit/scripts/manifest_check.py`: 460 manifests; 0 missing fields; 403 `+dirty`. Per-experiment code versions are listed in M-3.

5. `python reports/audit/scripts/independent_stats.py reports/audit/independent_stats.json` (numpy only, 2,000 paired bootstrap resamples over seeds):

   | Contrast | Seeds | Estimate | 95% CI | Relative | Verdict |
   |---|---|---|---|---|---|
   | H1 dev − plain_rl (strongest) | 0–9 | +0.0121 | [−0.0081, +0.0313] | +1.4% | failed (threshold +0.0837) |
   | H1 dev − plain_rl | excl. 7 | +0.0135 | [−0.0089, +0.0344] | +1.6% | failed |
   | Exp6 shared − specialists (AUC) | 0–9 | +0.0895 | [+0.0516, +0.1231] | +13.7% | passed (≥0.03) |
   | Exp6 shared − specialists | excl. 7 | +0.0896 | [+0.0476, +0.1282] | +13.8% | passed |
   | Exp6 dev − plain_rl (AUC) | 0–9 | +0.0493 | [+0.0137, +0.0836] | +7.1% | — |
   | Exp6 dev − plain_rl | excl. 7 | +0.0515 | [+0.0126, +0.0904] | +7.5% | — |

   Per-seed H1 differences (dev − plain_rl): s0 −0.012, s1 −0.002, s2 +0.009, s3 −0.054, s4 +0.051, s5 +0.036, s6 +0.039, s7 −0.001, s8 −0.002, s9 +0.057. Four of ten seeds favour plain RL.

   Holm: `hpe.evaluation.stats.holm([0.01, 0.04, 0.03, 0.005])` = [0.03, 0.06, 0.06, 0.02], identical to a reference step-down implementation.

6. `python reports/audit/scripts/frozen_policies.py`. Held-out compare accuracy before → after 300 episodes:

   | Policy | Before → after | State changed |
   |---|---|---|
   | static_persona | 0.433 → 0.433 | no |
   | memory_only | 0.500 → 0.817 | yes |
   | plain_rl | 0.500 → 0.867 | yes |
   | developmental | 0.500 → 0.900 | yes |
   | developmental (`use_learning=False`) | 0.500 → 0.500 | no |

7. `python reports/audit/scripts/adversarial_api.py <tmp>`, run once with default limits and once with `AUDIT_RAISE_RATE_LIMIT=1` (outputs: `adversarial_api*.json`).

   **Routing and safety:**
   - The baseline question and prompt-injection and probability-elicitation prompts produced 3 labelled scenarios each, with 0 probability phrases.
   - Explicit crisis wording was routed to crisis (0 scenarios).
   - Financial and relationship questions were routed.
   - **Indirect crisis wording, medical and legal questions were not routed.**

   **Consent and limits:**
   - No consent gives 400.
   - With the default limits, 40/40 burst requests got 429.
   - With the limits raised, a 200 KB question and a 10K-character user id were accepted.

   **Privacy:**
   - Export then delete removes every user row (direct SQLite check: none left).

   **Security probes:**
   - Path traversal returns 404 everywhere.
   - A SQL-injection probe returns total 0.
   - `limit=1e9` returns 422; 20,000 agents returns 400.
   - A foreign CORS origin is not allowed.

8. `python scripts/scan_secrets.py` → `secret scan: 1277 files scanned, 0 finding(s)`.

9. `make demo-check DEMO_ARGS="--api-port 8798 --web-port 5198"` → `demo-check: 14/14 checks passed`. Port 8000 was occupied by the orchestrator's e2e server.

10. `node reports/audit/scripts/a11y_e2e.js http://127.0.0.1:5199 <axe.min.js> reports/audit/a11y_e2e.json`: axe-core 4.10.2 against the production build plus a live API on :8799 with real stored data. Violations are summarised under M-6; 0 critical, 0 console errors, 0 failed requests, mobile overflow 0 px on all routes.

11. Coverage: `pytest -q -m "not slow" --cov` → 273 passed; TOTAL 91%.

12. Citations (Crossref API) and HeartSteps (PubMed E-utilities): see `verified_ok` in `findings.json`.

13. Preregistration order (`git log`): experiment YAMLs were added in `5817dff` (03:46 UTC) and are unchanged since; H1 results were committed in `ac1bb22` (04:12 UTC). exp07 was preregistered in `1e3d019` (03:42) and its results committed in `efc5be4` (04:22).

## Cost and performance (from committed manifests; local CPU; $0 API)

| Workload | Runs | CPU s | Agent-years | CPU s / agent-year | LLM calls |
|---|---|---|---|---|---|
| Stored society 100 agents × 80 y (developmental) | 1 | 106.8 | 9,398 (lived) | 0.0114 | 0 |
| H1 learning lab (incl. sensitivity) | 200 | 1,085.0 | 40,000 (nominal) | 0.0271 | 0 |
| Exp1 | 90 | 320.6 | 18,000 (nominal) | 0.0178 | 0 |
| Exp6 | 40 | 248.7 | 8,000 (nominal) | 0.0311 | 0 |
| Ablations | 90 | 328.2 | 18,000 (nominal) | 0.0182 | 0 |
| Benchmark 100 × 2 y developmental | 1 | 5.4 | 211 | 38.7 agent-years/s wall (load avg ≈ 3.5) | 0 |

The learning-lab "agent-years" are nominal budget units (agents × the age span the practice is spread over), not simulated lived years, so they aren't comparable with the society row.

**Cost assumption (illustrative, not a quote):** at $0.05 per vCPU-hour, the 100×80 society costs about 106.8/3600 × $0.05 ≈ **$0.0015**, or roughly **$1.6 × 10⁻⁷ per simulated agent-year**. Paid model cost is $0 because no LLM is called. 1k and 10k agents have not been measured.

## Recommended fix order

1. **H-1 and M-1:** expand the safety lexicon and add regression tests, using `adversarial_api.py` as the test bed.
2. **M-2:** add field length limits (contract minor bump plus ADR) and a body-size limit.
3. **M-6:** accessibility fixes, then rerun `a11y_e2e.js`.
4. **M-4 and M-3:** commit the tuning scripts and logs; use disjoint tuning seeds; pin experiments to a clean tree.
5. **M-5:** reword the exp07b claim; add a precision criterion.
6. **L-3 and L-4:** strengthen demo-check; serve the human-data results in the UI.
