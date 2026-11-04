# Social mechanism v2 — Candidate E notes (Phase II, lane `research/social-developmental`)

**Status: EXPLORATORY.** Every number below comes from dev seeds 1000–1004 (never 0–9 or 500–509),
from committed scripts under `experiments/phase2/social/` with logs under `reports/phase2/social/`.
Nothing here is a confirmatory result and nothing here touches the Phase II test partitions.

## 1. The problem this lane was given

Phase I measured (Exp 3, `RESEARCH_RESULTS.md`): developmental agents form 0.018 friends in 15 years
(memory-only agents: 0.49); connected − isolated on held-out accuracy = −0.004; the learning-lab social
ablation = −0.004 [−0.013, +0.004]. The directive named two suspected faults: (a) the activity bandit learns
away from socialising so demonstrations never arrive; (b) even injected demonstrations add nothing.

Re-diagnosis on dev seed 1000 (100 agents × 15 years, Phase I mechanism): 1 friendship and 5,710 one-off
acquaintance ties. The plaza drew partners afresh every tick, so no tie ever reached the three interactions a
friendship needs; and peer demonstrations were a random training item with outcome-only (bandit) feedback.
The transfer audit (§4) shows that fault (a) is mis-stated: demonstrations *do* arrive under Phase I, about
nine per agent-year, mostly from family at home. What never happens is repeat contact.

## 2. The mechanism (flag `social_v2`, off by default)

Code: `packages/hpe/world/social.py` (new), flag-guarded dispatch in `packages/hpe/world/engine.py`,
learner wrapper `packages/hpe/development/social.py` (policy class `developmental_social`).

The flag lives in `SimulationConfig.policy_params["social_v2"]` because `WorldParams` is frozen. The
`developmental_social` policy strips that key (and `teaching_weight`) before building the unchanged Phase I
`DevConfig`, so the same `policy_params` dict can carry world options without touching contracts. With the
flag absent the engine runs the Phase I code paths byte for byte (§6).

| component | what it does | why it should make social information *useful* (not rewarded) |
|---|---|---|
| persistent relationships | plaza pairing prefers a present known partner with probability `p_known` (0.7), weighted by tie strength; ties not contacted for a year decay (`interactions × 0.7`), friend → acquaintance below strength 2 | repeat contact is what lets trust be *learned* about a specific person |
| trust from demonstration reliability | after every demonstration the learner's trust in that demonstrator moves toward 1 if the demonstrated action was correct and toward 0 if not (`rate` 0.2), on top of the Phase I trust game | trust becomes an estimate of informational reliability, so friendship (trust ≥ 0.6 after ≥ 3 interactions, Phase I thresholds unchanged) emerges from being a reliable informant |
| teaching on learner-uncertain items | the demonstration's task family is the learner's weakest by its own self-estimate; a demonstrator measured ≥ 0.7 accurate in that family gives instructional (full) feedback with its own answer as the label, otherwise outcome-only feedback | information lands where the learner's uncertainty is highest; competence gates instruction so wrong teachers cannot instruct |
| conflicting information | misinformers (intervention flag, unchanged) assert a wrong label as instruction; the reliability update then lowers trust in them | misinformation is defeated by observed unreliability, not by a hard-coded discount |
| cooperation task | each non-isolated meeting also shares one environment item; both act, both learn from full environment feedback; trust moves toward the partner's correctness (`0.1`) | a second, cheap channel through which a partner's competence is observable |
| learning-progress signal | socialising agents report the mean measured information gain of the demonstrations they received (clipped to [0, 1]) as the same `learn` term self-study reports; `learn_signal: false` removes it | the activity bandit can value socialising only insofar as it actually taught something; no reward for having friends |
| social-reward-only control | `information: false`: meetings, pairing, trust game, tie decay, friendship rules all run, but no demonstrations and no cooperation items | isolates the social *reward* from social *information* |

Information gain is measured per demonstration as `log p_after(correct) − log p_before(correct)` from the
learner's own `action_probabilities`, so "knowledge transferred" is a change in the learner's policy on the
correct action, never self-report.

**Instrumentation.** `SocialV2State.stats` (cumulative) and `year_stats`: meetings, repeat meetings, ties
formed/decayed, friendships formed/lost, demonstrations (total, instructional, correct, from misinformers,
correct from misinformers), information-gain sum/count, cooperation items/joint-correct. Emitted once per
simulated year as a `social_interaction` event with mechanism `world.social.v2.stats`, together with mean
trust toward honest vs misinforming demonstrators and the friend/acquaintance tie counts. State is part of
`snapshot()`/`load_snapshot()`, so resume is bit-identical (tested).

## 3. Dev results — the grid (`experiments/phase2/social/run_dev.py`)

Seeds 1000–1004; 100 agents × 15 years; policy `developmental_social`; `--workers 1`; CPU 0.30 h
(`reports/phase2/social/dev/{summary.json,report.md,run.log,runs/}`). Paired bootstrap over seeds
(4,000 resamples), 95% CI.

| arm | accuracy 10–60 | youth 10–20 | mean friends 10–80 | socialise share | demos / agent-yr | repeat-meeting share | instructional share | info gain (nats) | trust: honest / misinformer |
|---|---|---|---|---|---|---|---|---|---|
| v1_connected | 0.807 | 0.817 | 0.055 | 0.188 | — | — | — | — | — |
| v1_isolated | 0.805 | 0.810 | 0.000 | 0.198 | — | — | — | — | — |
| v2_connected | 0.810 | 0.816 | **1.459** | 0.210 | 10.23 | 0.58 | 0.53 | 0.152 | 0.60 / — |
| v2_isolated | 0.813 | 0.820 | 0.000 | 0.212 | 17.24 | 0.96 | 0.51 | 0.129 | — |
| v2_reward_only | 0.789 | 0.806 | 0.027 | 0.201 | 0.00 | 0.59 | — | — | 0.38 / — |
| v2_no_learn_signal | 0.809 | 0.812 | 1.292 | 0.186 | 9.53 | 0.58 | 0.54 | 0.142 | 0.60 / — |
| v1_misinfo | 0.798 | 0.806 | 0.041 | 0.190 | — | — | — | — | — |
| v2_misinfo | 0.810 | 0.817 | 1.279 | 0.188 | 10.33 | 0.58 | 0.54 | 0.141 | 0.60 / 0.40 |

| contrast (treatment − control) | metric | estimate | 95% CI | reading |
|---|---|---|---|---|
| v2_connected − v1_connected | mean friends | **+1.40** | [+1.28, +1.57] | relationship formation repaired (67–88 friend ties per 100-agent society at year 15) |
| v2_connected − v2_reward_only | mean friends | +1.43 | [+1.31, +1.59] | friendships come from reliable information exchange, not from the trust game alone |
| v2_connected − v2_isolated | accuracy 10–60 | −0.0030 | [−0.0074, +0.0022] | **connected societies still do not out-learn isolated ones** |
| v1_connected − v1_isolated | accuracy 10–60 | +0.0026 | [−0.0017, +0.0070] | same under Phase I |
| v2_connected − v2_isolated | youth 10–20 | −0.0043 | [−0.0110, +0.0020] | no youth effect either |
| v2_connected − v2_reward_only | accuracy 10–60 | **+0.0209** | [+0.0146, +0.0276] | demonstration information is worth about two accuracy points |
| v2_connected − v2_reward_only | youth 10–20 | +0.0092 | [−0.0049, +0.0288] | not resolved at 5 seeds |
| v2_connected − v1_connected | accuracy 10–60 | +0.0029 | [−0.0015, +0.0073] | the mechanism itself adds no measurable accuracy |
| v2_connected − v2_no_learn_signal | accuracy 10–60 | +0.0013 | [−0.0050, +0.0058] | the learning-progress signal does not change skill … |
| v2_connected − v2_no_learn_signal | socialise share | +0.0233 | [+0.0057, +0.0407] | … although it does raise time spent socialising |
| v1_connected − v1_misinfo | accuracy 10–60 | +0.0094 | [+0.0067, +0.0120] | under Phase I, misinformation costs about one point on these seeds |
| v2_connected − v2_misinfo | accuracy 10–60 | −0.0001 | [−0.0035, +0.0026] | **under v2 the same misinformation costs nothing**; trust in misinformers 0.40 vs 0.60 for honest demonstrators |

Phase I reported misinformation harm −0.002 with a CI including zero on seeds 0–9; here, on seeds
1000–1004, the Phase I mechanism shows a small harm whose CI excludes zero. Both are small; the discrepancy is
noted, not resolved.

### 3a. Misinformation damage and recovery (`experiments/phase2/social/misinfo_recovery.py`)

Misinformation from 10% of adults for years 0–5 only, then switched off; 5 seeds; CPU 0.13 h
(`reports/phase2/social/misinfo/`). Under v2 the misinformers' share of demonstrations is ~4% while active;
mean trust toward them is 0.29 in year 1 against 0.58 for honest demonstrators, i.e. reliability tracking
isolates them within the first year.

| contrast | estimate | 95% CI |
|---|---|---|
| damage at year 5, v2 (clean − misinformed) | +0.0005 | [−0.0021, +0.0024] |
| residual at year 15, v2 | −0.0007 | [−0.0043, +0.0029] |
| recovery (gap@5 − gap@15), v2 | +0.0012 | [−0.0013, +0.0035] |
| v2 clean − v1 misinformed at year 5 | +0.0118 | [+0.0033, +0.0202] |

There is no damage to recover from under v2; the last row mixes mechanism and misinformation and is only a
bound on the Phase I loss.

## 4. What is actually transferred (`experiments/phase2/social/transfer_audit.py`)

Measurement hooks only (no mechanism change, no randomness consumed): every demonstration is recorded with
the learner's probability of the correct action before and after the update. Seeds 1000–1002; CPU 0.05 h
(`reports/phase2/social/transfer/`).

| measure | Phase I (v1) | v2 |
|---|---|---|
| demonstrations per agent-year | 9.38 | 10.17 |
| share with instructional feedback | 0.00 | 0.47 |
| share whose demonstrated action was correct | 0.78 | 0.76 |
| mean information gain per demonstration (nats) | 0.155 | 0.179 |
| modal action wrong → correct because of the demonstration ("flip") | 11.2% | 10.7% |
| correct → wrong ("anti-flip") | 0.8% | 1.7% |
| flips per agent-year | 1.05 | 1.09 |
| (agent, family) cells where demonstrations outnumber practice | 2.9% | 4.2% |
| demonstrations vs own practice episodes per cell | 44 vs 262 | 48 vs 257 |
| cells learned only through demonstration (zero practice) | 0 | 0 |

Demonstrations transfer real information in both mechanisms (a tenth of them correct the learner's modal
action), but they are about one sixth of each agent's learning input and no skill is ever acquired only
socially: every family is also practised hundreds of times. That bounds the marginal value of any social
channel in this world at the current practice volume.

## 5. Honest interpretation

1. **Fault (a) is not what the data show.** Demonstrations arrive under Phase I at nine per agent-year and
   socialising time is ~19% in both mechanisms. The missing piece was repeat contact; v2's known-partner
   preference (58% repeat meetings) plus reliability-tracked trust gives 1.3–1.5 friends per agent. In the
   social-reward-only control trust stays at 0.38 and friends at 0.03, so under v2 friendship is a consequence
   of exchanging reliable information, which is the behaviour the directive asked for, not a reward for it.
2. **Fault (b) is partly contradicted.** Removing demonstration information costs 0.021 accuracy
   [0.015, 0.028]. Social information is useful. It is not *peer* information that is useful, though: the
   isolation intervention blocks non-family contact only, and family teaching at home more than substitutes
   (17 vs 10 demonstrations per agent-year under isolation, 96% repeat contact). The H3 contrast "connected vs
   isolated" is therefore not an information-access contrast in this world, and it stays null.
3. **Misinformation robustness is a real, mechanism-level gain**: trust that tracks observed reliability
   removes the Phase I misinformation cost (+0.009 → 0.000) with no learner-side special-casing.
4. **The learning-progress signal** raises socialising time but not skill. It is the right kind of signal
   (it values socialising only when demonstrations measurably taught something) and it is harmless, but at
   this practice volume there is nothing extra to gain from socialising more.
5. **Nothing here supports H3 as stated.** Connected societies do not out-learn isolated ones on
   population accuracy on dev seeds with either mechanism, for either learner. The one positive signal is
   youth-only and hybrid-only (§5a).

### 5a. Hybrid exemplar learner (Candidate A) under the same arms

`reports/phase2/social/dev_hybrid/` (4 arms × 5 seeds, 100 agents × 15 years, `--policy hybrid_exemplar`,
CPU 0.45 h; the `PolicyClass` enum has no hybrid member, so the config label falls back to `developmental`
while each agent's policy descriptor reports the real class).

| arm | accuracy 10–60 | youth 10–20 | mean friends | socialise share | demos / agent-yr | info gain (nats) | trust honest |
|---|---|---|---|---|---|---|---|
| v1_connected | 0.801 | 0.810 | 0.077 | 0.190 | — | — | — |
| v2_connected | 0.806 | 0.819 | **1.681** | 0.192 | 10.15 | 0.057 | 0.63 |
| v2_isolated | 0.806 | 0.807 | 0.000 | 0.234 | 16.87 | 0.056 | — |
| v2_reward_only | 0.788 | 0.808 | 0.076 | 0.222 | 0.00 | — | 0.39 |

| contrast (treatment − control) | metric | estimate | 95% CI |
|---|---|---|---|
| v2_connected − v2_isolated | accuracy 10–60 | +0.0000 | [−0.0061, +0.0054] |
| v2_connected − v2_isolated | youth 10–20 | **+0.0117** | [+0.0056, +0.0174] |
| v2_connected − v2_reward_only | accuracy 10–60 | +0.0188 | [+0.0145, +0.0249] |
| v2_connected − v2_reward_only | youth 10–20 | +0.0108 | [+0.0075, +0.0139] |
| v2_connected − v1_connected | accuracy 10–60 | +0.0054 | [−0.0027, +0.0113] |
| v2_connected − v1_connected | mean friends | +1.60 | [+1.48, +1.73] |

Same picture as the developmental learner on the whole population (no connected-vs-isolated effect; a
two-point information effect; friendships repaired), with one difference: for the hybrid learner the
**youth cohort (ages 10–20) does learn more when connected**, +0.012 [+0.006, +0.017]. This is one
secondary metric among a dozen contrasts on five dev seeds, with no matching effect for the developmental
learner (−0.004 [−0.011, +0.002]); it is a candidate for a pre-registered confirmatory contrast, not a
finding. A plausible mechanism, untested: the hybrid learner moves less per demonstration (0.057 nats vs
0.15), so it benefits from the *repeated* teaching that persistent ties provide, and adolescents are the
cohort whose plaza contact is not substituted by family teaching.

## 6. Bit-identity with the flag off

* `python -m hpe.cli smoke` before and after the engine edits: `df969b261a5ecbe53c9620f416547e644a7637b2e30a673abd8e641968c6eda4` both times.
* Existing determinism and snapshot/resume tests in `tests/world/test_world.py` pass unchanged.
* `tests/world/test_world_social_v2.py::test_flag_absent_never_touches_v2` asserts no v2 state, no v2 events.

## 7. What a confirmatory test could and could not claim

Supported by the dev evidence, and testable on the frozen seeds 500–509 if the orchestrator pre-registers it:
(i) social mechanism v2 yields ≥ 1 friend per agent where Phase I yields ≈ 0; (ii) demonstration
information raises held-out accuracy versus the social-reward-only control; (iii) misinformation that costs
accuracy under Phase I costs none under v2. Not supported, and not worth a confirmatory run in this world:
connected > isolated on population accuracy (ages 10–60), for either learner. Worth one pre-registered
secondary contrast at most: connected > isolated on youth accuracy (ages 10–20) for the hybrid learner
(§5a), with the explicit expectation that a dev-seed effect of +0.012 may not replicate.

To make the H3 contrast meaningful, the next exploratory step would be an isolation variant that also removes
family teaching (a flag-guarded `social_v2` option, not a contract change), or a task regime where practice
is capacity-limited so social input is not redundant; both are world-lane decisions.

## 8. CPU accounting (this lane, logged)

| run | CPU-h |
|---|---|
| dev grid, 8 arms × 5 seeds | 0.30 |
| misinformation recovery, 3 arms × 5 seeds | 0.13 |
| transfer audit, 2 arms × 3 seeds | 0.05 |
| hybrid grid, 4 arms × 5 seeds | 0.45 |
| tests, smoke runs, diagnosis | ≈ 0.10 |
| **total** | **≈ 1.03** |

Budget was 4 CPU-h.

## 9. Gaps and requests

* `hpe.contracts.registry._PROVIDERS` has no entry for `developmental_social`; scripts must
  `import hpe.development.social` first. Request to the orchestrator: add the mapping (backward compatible).
* Test files are named `tests/world/test_world_social_v2.py` and `tests/development/test_dev_social_v2.py`
  rather than the `test_social_v2.py` in the directive, because the test tree has no `__init__.py` files and
  pytest refuses duplicate basenames.
* `teaching_weight` (learner-side scaling of instructional demonstrations) is implemented and tested but was
  left at 1.0 in every run above; it is a tuning knob for the tune partition, not a result.
* Five dev seeds resolve effects of about ±0.004 accuracy; smaller effects are indistinguishable from zero.
