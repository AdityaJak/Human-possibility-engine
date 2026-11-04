# U1 — Independent Critique (design and implementation review, before confirmatory results)

_Audit role; the critic did not design or build U1. Reviewed on 2026-11-04 at the freeze commit `6776cfc`, while the
confirmatory run was still training the meta-RL agents. No confirmatory result existed or was read, and no
confirmatory universe was generated. Numbers come from files in `results/` (cited) or from the critic's own runs,
marked **[critic run]**: development universe 0 only, single thread, about 2.5 CPU-min in total, nothing written to
the repository. No existing file was modified._

## Summary verdict

The process is sound. The freeze precedes the confirmatory run, the agents see no hidden state, outcome encodings
agree across env, IRA and meta-RL, and Holm is implemented correctly. I found **no bug that inflates IRA or deflates
the meta-RL agent**.

The problem is different: **the frozen agent does not implement the principle it is meant to test.**

* IRA never chooses a probe. Its information gain for a probe is identically zero.
* It never opens a concept for the unseen role.
* Its library does not recover the roles (ARI 0.276).
* Cue tracking is inert, and so is the information-gain term: β=0 matches or beats β=1 on dev.
* The `ira_noactive` ablation (M4) measures Thompson sampling against greedy action selection, not active
  identification.

What IRA does have is a hand-specified, nearly well-specified generative model of this environment, plus
designer-chosen priors for exactly the four shift types being tested. Its opponent is a flat GRU whose training
distribution contains none of those shifts. With no cross-world learning at all, IRA already reaches 89–96 % of its
above-random return on three of the four shift conditions.

Consequences:

* A passing H1 can establish only a **descriptive** advantage of this frozen agent over this comparator, in three
  co-designed universes.
* Claims about active identification, interventional concept formation, opening new concepts, accumulation, or
  "general principles vs incidental correlations" are **vetoed in advance on dev evidence**, whatever the
  confirmatory outcome.

## Process checks that passed

* **Freeze.** The hashes in `results/FREEZE.json` match the current `u1/*.py` and `EXPERIMENTS_AND_BASELINES.md`.
  * The freeze was written at 20:37:14 UTC and committed at 20:37:21 (`6776cfc`). `run_confirm.sh` started at
    20:37:23 (file times).
  * The running jobs use `128 0.001 20000` (checked with `ps`), matching the frozen settings.
  * From the draft (`c5c18bb`) to the freeze, the preregistration only *gained* the selection table and D-5. No
    threshold changed and no code changed.
* **Information hygiene.** IRA, the baselines and `MetaRLAgent` receive only `x`, `t`, their own actions, outcomes and
  rewards. `ira.py`/`recluster.py` import only constants and helpers from `env`. True roles are written only to
  diagnostic `_assign.json` files that no agent reads. `SWITCH_T`, ρ and the condition are not visible to any agent.
* **Encodings.** All four places agree that bit *b* is `(o>>b)&1`:
  * `env.encode_bits` (env.py:177);
  * `ira.decode_bits` (ira.py:258, 274);
  * `metarl.make_obs` (metarl.py:81-82);
  * meta-RL training, which encodes `b0+2b1+4b2` (metarl.py:152).
* **Meta-RL sampler.** It is a re-implementation of the environment, but its distribution matches
  `make_world('train')`. **[critic run]** 3000 draws each gave identical (distinct roles, reacting pairs)
  frequencies: (3,1) 606 vs 607 and (3,2) 2394 vs 2393.
* **Other checks.**
  * Common random numbers are used for outcomes.
  * Seed ranges are disjoint: training 100000+, dev 0–199, confirmatory 10000–10199, accumulation 20000+/30000+.
  * `holm` is correct (step-down with running max).
  * Tests: **[critic run]** `15 passed in 1.39s`.

## Findings

### F1 — IRA never runs an experiment: probe actions cannot be reached — **blocker** (for every mechanism claim)

* **The cause.** The probe score is β_t·IG_probe (ira.py:244-251). IG is computed from the *expected* law under each
  assignment, so it is information about z only. MATHEMATICAL_FORMULATION.md:58 says "assignment/law posterior",
  which is not what the formula or the code computes.
* **Why it is exactly zero.** Probe-bit laws start at Beta(½,½) with no counts (ira.py:159-162). Every hypothesis
  therefore predicts 0.5 for every bit, so IG_probe ≡ 0. Every pair, meanwhile, scores p̂ > 0.
* **Why it never changes.** Training starts from an empty library, so IRA never probes. The library's probe counts
  stay zero, so it never probes at evaluation either.
* **Evidence:**
  * Every library has all-zero `bits`: `results/u0/ira_library.json` and the β=0.5 and β=2.0 libraries.
  * **[critic run]** At world start with the trained library, max |IG_probe| = 1.8e-14 and min pair p̂ = 0.13.
  * In `results/u0/dev/*.jsonl` (`actions`), the probe fraction is **0.0000 for every IRA variant in every
    condition** (6 variants × 5 conditions × 200 worlds).
* **Consequences:**
  * The claim "identifies those concepts … by running the interventions that best discriminate" (principle §1) is
    not what the agent does. Every IRA action is a reward-bearing pair.
  * The identification channel that the v2 redesign added (informative 3-bit probe signatures; prereg line 118) is
    available only to the meta-RL agent.
  * The unseen role's distinct signature — the most direct consequence-based evidence of a new concept — is never
    collected.
* **Direction of the effect.** This handicaps IRA; it does not inflate it. But the confirmatory run tests a different
  agent from the one the documents describe. This cannot be fixed after the freeze; report it as it stands.
* **Limits:** every statement about "active identification", "self-designed experiments" or "interventional"
  identification.

### F2 — The concept machinery the principle names is inert on dev — **blocker** (for concept-formation claims)

* **The open concept set never fires.**
  * **[critic run]** In dev `novel_role` worlds (seeds 0–59), none of the 82 role-5 entities received a fresh MAP
    label. Their labels were k0:22, k1:30, k2:4, k3:13, k4:13.
  * `ira_closed` behaves the same way (0.832 vs 0.836).
  * The reason is the prior. The per-entity prior for a fresh label is α/M/(N+α), with N = 1,091 memberships, so
    against the main concepts it starts 5.7–6.6 nats down (ira.py:173-178). The more IRA learns, the harder novelty
    becomes. D-3 saw the same pattern: an oracle library scored 0.60 on `novel_role`.
* **The library is not a set of roles.**
  * **[critic run]** ARI between the final library assignment and the true roles is **0.276** (1,091 items,
    `ira_library_beta1.0_assign.json`).
  * Concepts 0, 1 and 4 each mix 4–5 roles, with at least 13 items from each.
  * The mixed concepts carry strong *self*-reaction laws: 1–1 has 772 successes vs 93 failures, 4–4 has 432 vs 32,
    and 3–3 has 310 vs 27. Yet no known role reacts with itself in universe 0 (the law diagonal is 0).
  * In other words, the library encodes "which entities react together", not interventional equivalence classes.
* **Cue tracking is inert.**
  * The w posterior does move: mass on w=0 is 0.66 under `cue_reversed` vs 0.21 under `iid` (dev jsonl `w_post`).
  * But removing it changes return by only +0.005 (dev, M1).
* **Accumulation adds no concept.** See F8.
* **What actually works.** The active ingredients are:
  * exact within-world Beta-Bernoulli inference over pair laws under a known model form;
  * greedy exploitation under an optimistic prior (F11);
  * the surprise reset (dev `ira_nochange` on `role_switch` 0.510 vs 0.681).
* **Limits:** "concepts defined by interventional consequences", "opens a new concept", "concept library learned
  across worlds" as explanations of any result.

### F3 — Shift robustness is supplied by the designers, not learned — **blocker** (for "general vs incidental" and for any "learning" interpretation)

**What IRA's model fixes in advance:**

* latent per-entity types;
* pair laws that depend only on the unordered type pair (ira.py:56-62);
* conditionally independent probe bits;
* appearance signs per type;
* the action→entity binding (`action_entities`, ira.py:19). The meta-RL agent must learn which entities each of its
  15 output units touches.

**Its shift-specific priors are never fitted to training data:**

* w is uniform on {0, ½, 1} in every world (ira.py:185-189; there is no w term in `_logpost`, ira.py:201-202), even
  though the cue is valid in 100 % of training worlds;
* change detection is hard-wired (τ = 4 over two observations, ira.py:287);
* the CRP fresh labels have α fixed;
* the meta-RL agent is trained only on unshifted worlds (metarl.py:90-108).

Each mechanism is a prior for exactly one of the test shifts, chosen by people who knew the test shifts.

**How much is built in.** In dev (`dev_eval_u0.log`), compare `ira_nolibrary` (zero cross-world learning) with `ira`.
Share of IRA's above-random return reached with no library:

| condition | share |
|---|---|
| `iid` | 86 % |
| `cue_reversed` | 96 % |
| `recombined` | 89 % |
| `novel_role` | 94 % |
| `role_switch` | 43 % |

The shares are computed as (nolib − random)/(ira − random).

**Is this fatal?** It is fatal to "the learner separates general principles from incidental correlations". It is
*not* fatal to a narrower, legitimate question: does a hand-specified structural Bayesian agent beat a generic RL²
GRU under shifts outside the GRU's training distribution? That question has a strongly expected answer and close
precedent (F10).

A fair test of the principle would need either:

* a meta-learner trained with the same nuisance variation IRA's priors admit (random ρ, occasional switches,
  held-out roles from a separate pool), tested on held-out shift *parameters*; or
* IRA priors estimated from the same training data.

**Limits:** "general principles vs incidental correlations"; sample-efficiency claims (prereg line 49: "8,500× more
training episodes"); any claim that the advantage is *learned*.

### F4 — The M4 ablation measures Thompson vs greedy, not active identification — **major**

* `ira_noactive` changes two things at once (ira.py:241-243): it removes the information-gain term, *and* it replaces
  argmax(p̂ + β·IG) with Thompson sampling of z.
* The β grid {0.5, 1, 2} (prereg line 55) never included β = 0.
* **[critic run]** Same library, dev seeds 0–59, `rng [0, s, 606]`:

  | condition | β=1 (IRA) | β=0 greedy | Thompson (`noactive`) |
  |---|---|---|---|
  | `iid` | 0.884 | 0.893 | 0.614 |
  | `novel_role` | 0.836 | 0.858 | 0.618 |
  | `role_switch` | 0.666 | 0.677 | 0.447 |

* The IG term contributes nothing or slightly hurts. The M4 gap (about 0.25 averaged over the five dev conditions; 0.281 on `iid`) is entirely the
  action-selection rule.
  60 worlds is a small sample, but the direction is the same in all three conditions.
* **Limits:** M4's label "active identification", even if it passes.

### F5 — The meta-RL comparator is narrow — **major**

* **Architecture.** One flat Linear(56→128)+ReLU → GRUCell(128) with linear heads (metarl.py:56-66; 106,640
  parameters). The 5×4 appearance vector is simply concatenated (metarl.py:73). There is no weight sharing across
  entities, no permutation equivariance, and pair actions are slot-indexed outputs. This is a weak inductive bias for
  `recombined` and for binding entities to actions.
* **Search.** Two configurations, one seed each, one universe. The A2C recipe was not tuned: entropy schedule, λ,
  batch, no advantage normalisation. The training curve is still rising at 20k updates (16.539 → 16.565 over the last
  2k; `metarl_u0_h128_lr0.001.log`).
* **Greedy action.** Greedy was chosen on `iid` (0.931 vs 0.925 sampled). Deterministic recurrent policies can repeat
  a failing action after a change, and the sampled variant is never run on shifts.
* **Matching.** Neither compute nor data is matched. The meta-RL agent gets more data and more training compute; IRA
  gets about 20–40× more inference compute (0.2–0.4 s/world per the prereg, 0.39 s for the selected β in
  `select_ira.json`, vs ≈0.01 s).
* **Missing comparators:**
  * an entity-structured (permutation-equivariant or attention) meta-learner;
  * a belief-based meta-RL (VariBAD: Zintgraf et al., ICLR 2020 / JMLR 2021 ✓);
  * a meta-RL agent trained with randomised nuisances (F3).
* **What is fair.** It is a reasonable baseline in-distribution: it beats IRA on dev `iid` (0.931 vs 0.899).
* **Limits:** "meta-learning cannot match …" can only become "this RL² GRU, trained without shift variation, did not
  match …".

### F6 — The statistics do not support inference beyond three trained instances — **major**

* **Clustering ignored.** `paired()` bootstraps worlds as if independent (stats.py:57-63). But the IRA library and the
  meta-RL weights are fitted once per universe, so the p-values concern new worlds inside three trained instances,
  not new universes or new training runs.
* **The p-value condition never binds.** **[critic run]** On dev, the per-world SD of paired differences is
  0.14–0.21, so the SE at n = 600 is about 0.006–0.008, and any δ ≳ 0.015 is "significant". Whenever condition (i)
  δ ≥ 0.10 holds, condition (ii) holds too.
* **What the test reduces to.** A point estimate ≥ 0.10 plus 3/3 per-universe signs. Under a symmetric null, 3/3
  signs happen with probability 1/8, so the sign rule is a replication check, not a significance test.
* **Margins are point estimates.** Margins and the `iid` non-inferiority bound (−0.05) are checked on point
  estimates (stats.py:119-125, 148), not CI bounds. The dev gap (−0.032) already sits close to the non-inferiority
  margin.
* **Asymmetric variance.** The meta-RL score is averaged over 2 training seeds (stats.py:43-49). IRA's library comes
  from one stochastic Gibbs run per universe, and its training variance is unsampled: the dev β-libraries differ in
  K and cluster sizes.
* **Limits:** any statement about the environment *family* or about "universes"; read H1 as three replications.

### F7 — The environment and shifts were co-designed while watching IRA, including one undisclosed change — **major**

* **v1 was rejected after IRA's v1 results on all five conditions** (`env_v1_findings.md`).
* **v2 tightened `role_switch` without disclosure.**
  * In v1, only the chosen pair stopped reacting (`eaa2e37` env.py:153-164).
  * In v2, every pre-switch reacting pair must stop, and roles are resampled until such a switch exists (env.py:117-135,
    165-172).
  * This change is not in the listed v2 changes (`env_v1_findings.md`:16-18; D-1).
  * It makes post-switch return depend entirely on change detection, the mechanism IRA has by design. `novel_role`
    also changed (k ∈ {1,2} → {2,3}).
* **Dev leakage (D-3/D-5).**
  * IRA's shift performance was seen while v2 and its shift definitions were being written. The 0.10 margin was
    written with IRA's D-3 shift levels in view.
  * The meta-RL agent never influenced the design. The only tuned knob (β) was selected on `iid`, and nothing in the
    thresholds or code changed after D-5.
  * So the bias acts at the level of the environment, not the thresholds: internal validity holds, external validity
    is limited to a family co-designed with IRA.
* **How realistic the shifts are.**
  * `cue_reversed` penalises using a cue that was valid in every training world. A Bayes-optimal learner for the
    training distribution *should* use it.
  * How severe the reversal is depends on the code geometry. In u0, only roles 2↔3 map onto each other's codes;
    roles 0, 1 and 4 map onto unused codes. With n = 3 universes, this heterogeneity is uncontrolled.

### F8 — The accumulation test (A) is weakly designed and fails on a dev pilot — **major**

* **The baseline is trivial.** The comparator is IRA's own frozen library (experiment.py:214-244). Any learner that
  updates should beat it, so the test does not check "without retraining" against a retrained alternative.
* **The stated prediction is untested.** Principle §7 (NEW_INTELLIGENCE_PRINCIPLE.md:56-58) predicts that the meta-RL
  agent shows no improvement across repeated unseen-concept worlds. The meta-RL agent is never run in a sequence, and
  with frozen weights it cannot improve by construction.
* **It uses the path already known to fail.** A runs on additive consolidation, the path D-2 found collapses.
* **It was never piloted.** There is no `u0/dev/accumulation.jsonl`.
* **[critic run] Pilot on dev seeds 30000–30049:**
  * K stayed 6 for all 50 worlds.
  * Role-5 entities were consolidated into existing concepts 0–4.
  * Over the last 25 worlds: accumulating 0.855 vs frozen 0.856 (difference −0.002, margin 0.05).
  * Over the first 25: 0.883 vs 0.898.
* **Prediction:** A fails, and no concept is ever created.

### F9 — The decision rule and the M5/M6 tests are weak attributions — **major**

* **The "advance" gate is effectively already met.** "Advance" needs H1 plus one of M1–M3. On dev, M3 = +0.171
  against a margin of 0.05, while M1 (+0.005) and M2 (+0.007) are expected to fail (D-5). So "advance" ≈ H1 alone,
  and the only mechanism it can name is the change heuristic, which bears only on `role_switch`.
* **M5 includes a capacity artefact.** `ira_nolibrary` has only 3 labels (ira.py:151), but `recombined` worlds have
  4–5 distinct roles. Part of M5's `recombined` gap is therefore a capacity artefact.
* **M6 is confounded.** `entity_bayes` ignores appearance and uses Thompson sampling (baselines.py:49-67). M6 mixes
  "concepts" with "uses appearance" and with greedy-vs-Thompson, which alone is worth about 0.25–0.28 (F4).

### F10 — The novelty ledger omits close precedents — **major**

All of the following were checked by web search on 2026-11-04:

* **Alchemy** (Wang et al. 2021, arXiv 2102.02926, NeurIPS Datasets & Benchmarks). Reports a Bayes-optimal ideal
  observer with the task's structural prior far ahead of deep meta-RL agents on a latent-causal-structure task
  distribution. This is the generic version of U1's comparison. (Abstract and listing checked; table values not
  re-read.)
* **EMPA / theory-based RL** (Tsividis et al. 2021, arXiv 2107.12544). An acting agent that infers object types and
  interaction rules by Bayesian inference, explores where its theory is uncertain, and is compared with deep RL. This
  removes "embedded in an acting agent" as a difference from Kemp et al. (2010).
* **Dasgupta et al. 2019** (arXiv 1901.08162). Meta-RL agents learn to choose informative interventions. This bears
  on the prediction that generic meta-RL cannot identify actively.
* **VariBAD** (Zintgraf et al. 2020). Belief-based meta-RL.
* **The infinite relational model** (Kemp, Tenenbaum, Griffiths, Yamada & Ueda, AAAI 2006). `recluster.py` cites it
  as its model, but the ledger does not list it, even though the library is learned by IRM-style offline Gibbs
  clustering (D-2).

Not re-checked by the critic: Bramley et al. (2017) and Gershman, Blei & Niv (2010). Both were cited by the team from
memory and are plausible.

* **What survives:** "New combination: modest" stands.
* **What must narrow:** "Possible new empirical finding" must be limited to *this specific shift battery*. The
  general result "a structured Bayesian agent beats meta-RL" already has precedent.

### F11 — The prior reaction rate is biased by the agent's own selections — **minor**

* r̄ pools every pair outcome the agent observed (ira.py:337; recluster.py:73-75), and most of those come from
  exploiting pairs that react.
* **[critic run]** The dev library has r̄ = **0.781**, while the true fraction of pairs that react in training worlds
  is **0.18** (2000 worlds).
* r̄ is the prior mean for every unseen concept pair and every fresh label (ira.py:155-170), and it is also handed to
  `entity_bayes` (experiment.py:102).
* It acts as a policy-dependent optimism bonus, and it is what makes greedy β=0 explore (F4). This is not a leak, but
  it belongs in the mechanism if IRA wins on `recombined`.

### F12 — Documentation has drifted from the code — **minor**

* MATHEMATICAL_FORMULATION.md §1–2 still describes environment v1: glow propensities, pokes, a single cue dimension
  (lines 6, 14, 28, 44, 53). Yet env.py:1 cites it as the specification.
* The information-gain text (line 58) does not match its own formula (F1).
* The thesis promised "a Bayesian learner with the same machinery but no concepts" and "an oracle" agent
  (ORIGINALITY_THESIS.md:93-94). `entity_bayes` is not the same machinery, and the oracle exists only as the
  normaliser.

### F13 — The pipeline can fail silently — **minor**

* **Missing seeds drop universes.** `scores('metarl')` intersects the per-seed keys (stats.py:43-49). If `metarl_s1`
  is missing for a universe, that universe silently drops out of every primary contrast and out of the sign rule.
* **Failures do not propagate.** `run_confirm.sh` runs its jobs in the background, and the bare `wait` (lines 25-31)
  does not pass on failures, so `summarize` can run on partial data.
* **No size check.** `summarize` does not assert n = 600.
* **The freeze covers too little.** `check_phase` hashes only `u1/*.py` and the preregistration (experiment.py:49-51).
  `settings.json`, `run_confirm.sh` and the CLI arguments are not checked; the `ps` check above confirmed they match
  this time.
* **Shared initial weights.** `torch.manual_seed(cfg.seed)` (metarl.py:118) gives `s0` the same initial weights in all
  three universes.

### F14 — The tests give false assurance — **minor**

* `test_recluster_recovers_roles_from_clean_records` (tests/test_u1.py:147-163) checks re-clustering on synthetic
  probe-only records, a regime that never occurs in training (F1).
* No test checks that:
  * IRA ever probes;
  * a library trained on real records recovers the roles;
  * `MetaRLAgent` has the right interface (the interface test at lines 97-101 omits it).

### F15 — The world distribution is narrower than documented — **minor**

* **[critic run]** In u0, every `train`/`iid` world has exactly **3** distinct roles (3000/3000 in both samplers).
  Two-role worlds cannot satisfy the "1–2 reacting pairs" constraint when no known role reacts with itself.
* So "2–3 distinct roles" (prereg line 23) is nominal, and the composition should be reported per universe.

## Claims this critique would veto

1. **A new principle or a new algorithm.** The team already concedes this.
2. **"IRA identifies concepts with self-designed experiments" / "active identification explains the advantage."**
   Vetoed even if M4 passes: no probe is ever chosen, and the IG term is not useful (F1, F4).
3. **"IRA forms concepts defined by interventional consequences" / "the library recovers the hidden roles."** Vetoed
   unless the confirmatory libraries show high ARI (F2).
4. **"IRA opens a new concept for an unseen concept" / "accumulates new concepts without retraining."** Vetoed unless
   A passes *and* the logs show a new concept assigned to role-5 entities. Even then, this is no claim against the
   meta-RL agent (F2, F8).
5. **"Separates general principles from incidental correlations" or "learned what generalises."** The robustness
   comes from designer priors and a hand-specified model (F3).
6. **"Generic meta-learning cannot match this" / "the meta-RL agent learned incidental correlations."** One flat
   architecture, one algorithm, no shift variation in training (F5).
7. **Sample-efficiency claims** (e.g. "8,500× fewer episodes") offered as evidence for the principle (F3).
8. **Mechanism attributions from M4 and M6, and from M5 on `recombined`** (F4, F9).
9. **Generalisation beyond the three trained universes or this co-designed environment family**, and any use of
   world-level p-values as evidence about universes (F6, F7).
10. **A generic "new empirical finding"** that a structured Bayesian agent beats meta-RL (F10).

## Claims that survive if the preregistered tests pass

1. **A descriptive H1 result.** "In three universes of an environment family co-designed with IRA, the frozen IRA
   agent retained ≥ 0.10 more normalised expected return than a flat RL²-style GRU under [the conditions that passed],
   with the same sign in each universe." The comparator must be described as: A2C, 2.56 M episodes, greedy, trained
   without shifts. The IRA agent must be described *as implemented*: no probing, mixed-concept library, surprise
   reset. The F1/F3/F5 qualifiers must be stated next to the result.
2. **If M3 passes:** "IRA's surprise-triggered reset accounts for part of its `role_switch` return." This is
   heuristic change detection, not concept formation.
3. **If M5 also passes on `iid` alone:** "the cross-world library improves in-distribution return over within-world
   inference." The library still must not be called a set of roles.
4. **The in-distribution comparison**, in either direction.
5. **Any negative outcome** (H1, M1, M2 or A failing) survives in full. Given F1–F3, a negative result is the more
   informative outcome.

**To report next to the confirmatory result** (labelled exploratory; uses existing artefacts; no frozen code
changes):

* `ira_nolibrary` − meta-RL per shift condition;
* IRA with β = 0;
* the meta-RL agent's probe frequency (from the logged `actions`), and the sampled meta-RL agent on the shifts;
* ARI, `bits` counts and K for the confirmatory libraries;
* the MAP label of role-5 entities, and the K trajectory in A;
* per-universe estimates as the primary display, with n checked to be 600 for every contrast.

**What a fair test of the principle needs** (a new preregistration):

* an IG that includes uncertainty about the laws, so probes have value;
* a meta-learner trained with randomised nuisances, plus a relational or belief-based meta-learner;
* at least 10 universes, analysed at the universe level;
* shift types specified by someone other than IRA's designers.

## Review of revision R1 (before the R1 freeze)

_Reviewed at commit `6dc56ef` (R1 code and the R1 section of `EXPERIMENTS_AND_BASELINES.md`), with attempt 1's
results (`dfe2071`, `results/report_confirm.md`) in view.

Critic runs are marked **[critic run R1]**:
* development universe 0 of environment v3 only, single thread, about 77 CPU-s in total;
* they used the director's dev libraries (`ira_library_beta0.5.json`) from the scratch folder `r1/results_r1/u0/`;
* no confirmatory universe (4–8) was generated or used._

**Statement: R1 must not be frozen until all five items below are done.**

1. The `ira_noactive` crash is fixed and covered by a test (R1-a).
2. The R1.3 development protocol has run end to end on the *committed* code (R1-b).
3. The preregistration states what the selected agent actually does, and the resulting M2/M4 predictions (R1-c).
4. It says honestly that `role_switch` is not held out for the broad comparators — or holds out t = 10 (R1-f).
5. It restores or discloses the cut in `gru_narrow`'s training budget (R1-g).

Items 1 and 2 are blockers. Items 3–5 are text changes or one-line changes. With all five done, R1 may be frozen. R1-h
and R1-j are recommended but not required.

### R1-a — `ira_noactive` crashes, and the crash would be reported as "H1 fails" — **blocker**

* **The bug.** `ira_cfg` calls `IRAConfig(beta=beta, **IRA_VARIANTS[name])` (experiment.py:87). R1 sets
  `IRA_VARIANTS["ira_noactive"] = {"beta": 0.0}`, so `beta` is passed twice and raises
  `TypeError: got multiple values for keyword argument 'beta'`. **[critic run R1]** I built every variant; only
  `ira_noactive` fails.
* **Effect on the run.** In `run_confirm_r1.sh:15`, `eval --agents ira,ira_noactive,…` stops at `ira_noactive` in
  every universe. Because the job chain uses `&&`, none of the following runs anywhere: `ira_closed`,
  `ira_nocuetrack`, `ira_nochange`, `ira_nolibrary`, `entity_bayes`, `ucb`, `random`, or the accumulation test.
* **How it would be misreported.** The strict `summarize` then raises *before* writing `summary_confirm.json`. The
  stage-2 gate (`run_confirm_r1.sh:20`) cannot read the file and falls through to the message "stage 2 skipped: H1
  does not hold against gru_narrow". A crash would be reported as a scientific result.
* **Fix:**
  * let the variant override β (`IRAConfig(**{"beta": beta, **IRA_VARIANTS[name]})`);
  * add a test that builds every agent through `make_agent`;
  * make the gate fail loudly when the summary is missing or `summarize` exits non-zero.

### R1-b — The R1.3 development protocol has not run yet — **blocker (procedural)**

* R1.3 commits to evaluating every agent on all five conditions in universe 0, with the ablations as a 50-world bug
  check, before the freeze. That check would have caught R1-a. R1-D does not exist yet.
* The dev artefacts so far came from a scratch copy. Its `env.py`, `metarl.py` and `stats.py` differ from the
  committed files **only by formatting** (diff checked). That is acceptable, but say so in R1-D: the freeze hashes the
  committed bytes.

### R1-c — F1 is fixed in the mathematics, not in the selected agent's behaviour — **major (must be stated before the freeze)**

**What is fixed:**

* `_expected_hb` is the correct closed form. For θ ~ Beta(a, b), with n = a + b,
  E[H_b(θ)] = ψ(n+1) − (a/n)ψ(a+1) − (b/n)ψ(b+1). The critic derived this independently, and there is a Monte-Carlo
  test.
* `act` now computes the mutual information between the outcome and (z, θ). With no probe counts, a probe is worth
  about 0.31 nats per bit, so IRA probes during training.
* The R1 dev libraries do carry probe counts:
  * β = 2: crisp signatures, 303–843 bit observations per concept;
  * β = 0.5: 0–51 per concept.

**What is not fixed.** The selection rule (`iid` only) picked β = 0.5 (0.8646) over β = 1 (0.857).

* The difference, 0.008, is below the paired SE at 200 worlds (≈ 0.014, using the per-world SDs from attempt 1's
  dev data). The choice is effectively a coin flip.
* The winner barely probes: 0.04 probes per world on `iid` (`select_ira.json`), against 1.42 for β = 1.

**[critic run R1]** β = 0.5 library, 15 dev worlds per condition:

| condition | probes/world (β=0.5) | norm β=0.5 | norm β=0 (`ira_noactive`) |
|---|---|---|---|
| `iid` | 0.13 | 0.861 | 0.874 |
| `cue_reversed` | **0.00** | 0.802 | 0.802 |
| `novel_role` | 0.33 | 0.681 | 0.651 |

In `novel_role`, 0 of 21 role-5 entities ended with a new MAP concept, under both β values.

**Consequences:**

* The agent that would be frozen again identifies almost only through pair outcomes. It does not probe even when its
  cue is reversed, and it never opens a concept.
* M4 is predicted to fail: +0.006 averaged over these three conditions, against a 0.03 margin. M2 is predicted to fail
  as well.
* The "advance" branch requires M4, so it is close to unreachable by construction.

The test stays interpretable — a failing M4 is informative. But the freeze must not happen without:

* writing these predictions into R1-D;
* not describing F1 as "fixed" without this qualifier.

**Optional.** A pre-stated tie rule, e.g. "the largest β within one paired SE of the best `iid` score" (which would
pick β = 1). This is still a dev-only choice on `iid` data, so it is permissible, but it must be disclosed as decided
after seeing the dev `iid` numbers. Whatever is chosen must be fixed before the freeze.

### R1-d — The prior reaction rate (F11) — **resolved**

* `Library.rbar` is now the mean posterior rate over concept pairs, each pair counted once. **[critic run R1]** It is
  0.284 for the selected dev library, against a true reacting-pair density of 4/15 = 0.267 among the known roles of
  v3 universe 0. (The fraction of pairs that react per training world is 0.178.)
* Re-clustering uses a symmetric Beta(½,½) pair prior.
* Minor: the online agent (prior r̄) and the Gibbs re-clustering (prior ½) now use different pair priors, so the
  `recluster.py` docstring ("same generative model the agent uses online") is no longer exact.

### R1-e — The M4 ablation (F4) — **resolved once R1-a is fixed**

`ira_noactive` is now β = 0. `ira_thompson` is kept as a descriptive, dev-only variant.

### R1-f — `train_broad` as an answer to F3: good, but `role_switch` is not held out — **major (text or one-line fix)**

**What is good.** Three test values are genuinely held out:

* cue validity: training uses ρ ~ U(0.3, 1), which admits uninformative and mildly reversed cues, while the test uses
  ρ = 0.1;
* the unseen role: extra roles 6–7, whose signatures differ from role 5, stand in for it in training;
* recombination: 4–5 distinct roles never occur in training.

**`role_switch` is not held out.** `train_broad` uses the identical `_switch_options` construction (every reacting
pair stops) at t ∈ {5, …, 15} (env.py:208-212). That range includes the test's t = 10 (env.py:217-221). So
`role_switch` is *in-distribution* for `gru_broad` and `rel_broad`, and R1.5's phrase "under held-out shift values" is
false for it. Either:

* hold out t = 10, e.g. by training on t ∈ {5–8, 12–15}; or
* say explicitly that `role_switch` is in-distribution for the broad comparators.

**Broad training gives more than IRA's priors.** It also hands the comparators task-structure regularities that IRA's
priors do not encode:

* special-role worlds in which *every* reacting pair involves the special entity — the same sampler serves roles 6/7
  and role 5 (env.py:166-177);
* switches that stop every reacting pair.

Under the intersection–union test this is conservative for H1. But it means "H1 fails against the broad comparators"
would show *"a learner trained on near-test task structure matches IRA"*, not cleanly *"IRA's advantage is only its
priors"*. State that reading in R1.5 before the freeze.

### R1-g — A change that favours IRA, made after attempt 1 and not disclosed — **major (restore or disclose)**

* `gru_narrow` now trains for 15,000 updates (R1.2; `--updates` default 15000). Attempt 1's comparator — the one that
  beat IRA on 3 of 4 shifts — trained for 20,000.
* The cut is not listed among the R1.1 changes. In attempt 1's dev curve the difference is small (mean training return
  16.477 at 15k vs 16.565 at 20k).
* But it weakens the stage-1 gatekeeper after seeing it win. Restore 20k for `gru_narrow` (about 4 more CPU-min per
  universe). Otherwise, list it in R1.1 with its reason and the dev `iid` score at 15k.

### R1-h — The relational comparator is correct and equivariant, with a memory limitation — **minor (disclose)**

**Correct.**

* `make_ent_obs` layout (17 dims) checked, with no overlaps:

  | dims | content |
  |---|---|
  | 0–3 | own appearance |
  | 4 | probed flag |
  | 5 | paired flag |
  | 6–8 | probe bits |
  | 9 | reaction |
  | 10 | reward |
  | 11–14 | partner appearance |
  | 15 | t |
  | 16 | first-step flag |

* `RelNet` is permutation-equivariant:
  * shared per-entity encoder and GRU;
  * position-free self-attention;
  * a symmetric pair head on (zᵢ+zⱼ, zᵢ⊙zⱼ);
  * a mean-pooled value.
* It learns: dev training return was 12.26 after 1,000 updates.

**Limitation.** The attention output is not fed back into the recurrent state (metarl.py:108-114 returns the
pre-attention `h`). Each entity's memory therefore holds only its own interactions plus its partners' appearances.
Cross-entity facts, such as "pair (i, j) already failed", must be re-derived at every step by matching appearances.
Disclose this together with the single configuration (hidden 64).

### R1-i — Statistics, staging and the intersection–union logic — **sound, with caveats**

**What is correct.**

* One-sided t-tests over universe means (df = 4), Holm within each comparator, and at least 4 of 5 universes positive.
* The intersection–union test needs no correction across comparators.
* A comparator that was not run counts as not passing (stats.py:151-153).
* Strict loading refuses missing or unpaired cells.
* Staging is valid for the H1 decision: H1 cannot be supported if it fails against `gru_narrow`.

**Caveats:**

* If stage 2 is skipped, R1 says nothing about the broad or relational comparators, and the final text must not imply
  they were tested.
* The mechanism tests have no per-universe sign rule (unlike H1). That is acceptable, but state it.
* Power. Attempt 1's between-universe SDs of the shift contrasts were about 0.03–0.05 (`report_confirm.md`). With 5
  universes, a true δ = 0.10 and SD 0.05 give t ≈ 4.5 (Holm-adjusted p ≈ 0.02). With SD 0.08 the same δ fails Holm.
  So the design only has power for effects comfortably above the margin.

### R1-j — `feasible()` is looser than the sampler — **minor**

* `_special_ok` accepts known parts with any number of distinct roles (env.py:96-105). `_sample_special` draws only
  2–3 distinct (env.py:166-177).
* A universe could therefore pass `feasible()` while `novel_role` (or an extra-role world) cannot be sampled — the
  same failure mode as attempt 1's universe 2.
* **[critic run R1]** In 300 feasible synthetic laws (private RNG, same law rule as `make_universe`, no universe
  seed), 0 showed the mismatch. The risk is theoretical, but align the check: skip known parts with fewer than 2 or
  more than 3 distinct roles.

### R1-k — Is R1 tuned to attempt 1's results?

**No sign of tuning in most of the design.**

* Thresholds and margins are copied unchanged, and the confirmatory universes (4–8) are new.
* The IRA changes trace to critique findings that were committed *before* attempt 1's results: `c5922ab` at 21:00
  precedes `dfe2071` at 21:24.

**Changes made after seeing the results:**

* the staged design (disclosed);
* β re-selection (same rule);
* the `gru_narrow` budget cut (R1-g, undisclosed).

**Reporting requirement.** R1 is a *second* confirmatory attempt, on a revised agent. Attempt 1 remains the
confirmatory test of the original agent (H1 not supported) and must be reported as such. The two must not be pooled,
and R1 must not replace attempt 1.

### What R1 can and cannot establish (given R1-c and R1-f)

* **If stage 1 fails**, R1 replicates attempt 1's rejection with an agent that is able to probe.
* **If H1 holds against all three comparators**, R1 shows a structured-prior advantage over generic meta-learners,
  some of them trained on near-test task structure. No mechanism claim follows unless M4 (and M2, for concept
  creation) pass, and on dev evidence they are predicted to fail.
* **The principle-level claim still needs both:**
  * M4 to pass at the frozen β;
  * new concepts to actually be used in `novel_role`.

## Post-results critique (both attempts)

_Reviewed at `7ddfc80`: results `4ee56f1`, verdict `a35b629`, status updates `7ddfc80`. **[critic check]** means one of
the following, all done in the scratchpad (≈ 10 CPU-s, no training or evaluation):
* a recomputation from the raw world-level `jsonl` files with the critic's own code, not `u1.stats`;
* a re-run of the frozen or the fixed `stats.py` on a scratch copy of `results_r1/`._

**Statement.** The REJECT decision stands. **But the verdict text must change before it is final:**
* three wrong numbers corrected (P-2);
* three of the "claims that survive" removed or downgraded (P-3, P-4, P-5);
* "refuted" replaced by "not supported" for new-concept creation and accumulation (P-6);
* the integrity wording about deviation R1-X1 corrected (P-1).

### P-1 — Deviation R1-X1: verified, but one sentence must be reworded

**Only one code line changed after the freeze.** **[critic check]** The post-freeze changes are:
* one line in `stats.py`, the stage-2 presence check;
* an appended paragraph in the preregistration.

Every other hashed file matches `results_r1/FREEZE.json`.

**The fix cannot change any number.**
* **[critic check]** The frozen `stats.py` (sha `1ac1ea4a…`, as in the freeze) raises `IncompleteResults` on
  `u4/confirm/gru_broad__iid.jsonl` before it prints or writes anything.
* The fixed `stats.py` reproduces the committed `summary_confirm.json` **byte for byte**.
* The changed line only decides whether a comparator that was not run is skipped; it never enters a contrast.

**What must be reworded.** These statements are not accurate as written:
* "No result was printed or read" (`TRANSFER_RESULTS.md` §2);
* "before any R1 result was read" (`SCIENTIFIC_VERDICT.md` §6 and the preregistration's R1-X1 entry).

`cmd_eval` prints per-agent, per-condition means while the run is going — for example
`u4 confirm gru_narrow cue_reversed norm 0.621` in `confirm_logs/gru_narrow_u4.log`. The partial snapshot `1ea845b`
committed 22 such lines before the fix.

The true statement is also the sufficient one: *no contrast or test was computed before the fix, and the fix cannot
change any number.* Say that instead.

**The loud stage gate worked as designed.** `confirm_run.out` reads "STAGE 1 FAILED … not deciding stage 2".

### P-2 — Every preregistered R1 statistic reproduces; three factual errors

**What reproduces.** **[critic check]** From the raw files, all of the following match `report_confirm.md` and
`summary_confirm.json` to ±0.001:
* every H1 contrast — estimate, per-universe values, t, p and Holm p;
* M1–M5;
* A.

`gru_narrow` trained for 20,000 updates in all five universes (logs). The R1-D dev table matches `results_r1/u0/dev/`.

**Three errors.** None of them changes a decision.

| statement | where | correct value |
|---|---|---|
| "0 probes in **36,000** confirmatory steps" | VERDICT:51, TRANSFER:79 | Attempt 1 had **56,000** IRA confirmatory steps (2,800 worlds × 20), none of them a probe (**[critic check]**) |
| "a new concept in **1 of 500** unseen-role worlds" | VERDICT:76, TRANSFER:179, NEW_INTELLIGENCE_PRINCIPLE:9 | **2 of 500**, both in u4: `diagnostics_concepts.json` gives 0.02 for u4, and TRANSFER's own table says "2%" (**[critic check]**) |
| "K unchanged" | VERDICT:78 | True in attempt 1 (5→5, 8→8, 7→7). In attempt 2, universe 4 went **7→8** (**[critic check]**) |

### P-3 — §5.1 is false for attempt 2 — **veto**

The verdict says the meta-learner "transfers well to cue reversal … matched or beat the structured agent in 3 of 4
shifts in both attempts". In attempt 2:

* **Cue reversal.** `gru_narrow` fell from 0.936 (`iid`) to 0.733, and IRA was ahead pooled: +0.084, world-level CI
  [+0.057, +0.110], with +0.18 and +0.37 in u4 and u8. The test did not pass.
* **Overall.** The GRU matched or beat IRA in only **2 of 4** shifts (`recombined` −0.006, `novel_role` −0.044). It
  lost `role_switch`.
* **Unseen role.** The GRU itself dropped by 0.145 on `novel_role`.

`TRANSFER_RESULTS.md` §2.4 states this correctly. VERDICT §5.1 contradicts it, and so does the "transfers well"
wording of TRANSFER §1.4 when it is carried over to both attempts.

**Rewrite as:** "In attempt 1 it matched or beat IRA under 3 of 4 shifts; in attempt 2 under 2 of 4, with a large,
universe-dependent loss to IRA under cue reversal (u4, u8)."

An overclaim *against* the proposed agent must be corrected as strictly as one in its favour.

### P-4 — "Entrenchment" is not a finding — **veto §5.4** and the matching statements

This comparison was not preregistered, and the evidence points both ways. **[critic check]**, `ira` − `ira_nolibrary`
on `novel_role`:

* **Attempt 2:** −0.083, −0.206, +0.027, −0.036, +0.132 per universe. The pooled −0.033 is carried by u5.
* **Attempt 1:** +0.027, +0.068, +0.026. The library *helped* in 3 of 3 universes.
* **Dev** (one universe, paired over 50 seeds): 0.730 vs 0.785.

At most, say: "the learned library did not help on `novel_role` in attempt 2 (mixed across universes), although it
helped in attempt 1; entrenchment is a hypothesis, not a finding."

The same correction applies to:
* VERDICT §3 (iii), "knowledge entrenches";
* TRANSFER §2.4, "so knowledge entrenches";
* §7's premise "the library made the unseen concept harder". Keep the research suggestion, but not as established
  fact.

### P-5 — §5.2 is not a claim that survives — **veto** (move it to dev observations)

The claim rests on dev data only:
* one design universe;
* scratch-trained models (15,000 updates) whose training switch times included t = 10, so `role_switch` was
  in-distribution for them.

On dev `role_switch`, `gru_broad` was **not** ahead of IRA (0.673 vs 0.682); only `rel_broad` was (0.770).

Move §5.2 to "dev observations (not confirmatory, not held out)". VERDICT §2's sentence "On dev, the relational
meta-learner … exceeded IRA in four of five conditions, including the role switch" needs the same not-held-out
qualifier.

### P-6 — "Refuted" is too strong for (iii) and for accumulation; acceptable for (i) with its scope stated

* **(i) Information-gain action selection.** M4 is −0.016, with all five universes ≤ 0 (**[critic check]**:
  reverse-direction two-sided p ≈ 0.13). β = 1 is worse than β = 0.5 in every condition, pooled, by 0.047–0.091.
  * "Refuted in this environment family" is defensible if it adds "for this implementation: a BALD-style bonus, β
    selected on `iid`, 20-step episodes".
  * Write "at least as good", not "beats". §5.3 and TRANSFER §2.4 say greedy exploitation "beats paying for
    experiments", but the difference is not significant.
* **(iii) Opening new concepts.** The mechanism almost never fired (2 of 500 worlds).
  * With α = 1 against roughly 10³ memberships, the prior puts a fresh label 5–7 nats behind any established concept
    (critique F2).
  * The experiment cannot tell "opening concepts does not help" from "this implementation almost never opens them".
  * Status should be **"not supported; not exercised as implemented"**, not "refuted".
* **Accumulation.** A failed its margin twice. But attempt 1 showed a small positive effect (+0.027, p = 0.009,
  positive in 3 of 3 universes), and attempt 2 a null (+0.006).
  * "Not supported", plus "requires offline re-clustering during training" (D-2), is accurate. "Refuted" is not.
* **(ii) Cue-validity tracking.** M1 = +0.120 was **positive in all five universes** (+0.031, +0.084, +0.031, +0.005,
  +0.450). It exceeded the 0.05 margin in only two, and it failed Holm (p = 0.34).
  * Replace "inconsistent" and "carried by 2 of 5" with "same sign everywhere, very uneven in size".
  * The explanation "it helps when a reversed cue maps onto another known role's code" (VERDICT §3 (ii), TRANSFER
    §2.4) is a post-hoc hypothesis with no analysis in the repository. Label it as such, or test it. The test: in
    each universe, count the known roles whose complemented code equals another known role's code, and compare that
    count with M1 in that universe.

### P-7 — Minor wording

* **The condition that passed.** VERDICT §1 says that in attempt 2 "the condition that passed is not attributable to
  the principle". It *is* attributable to component (iv) of the stated principle — re-identification after
  contradiction — implemented as a fixed-threshold surprise rule, which is a known change-detection heuristic. Say
  exactly that.
* **Cue-validity tracking.** VERDICT §1's list of mechanisms omits it. Include it as "positive in direction, not
  established".
* **Scope.** The verdict sentence should say the hypothesis is rejected "as preregistered, in the one co-designed
  environment family tested".

### P-8 — Rejecting despite the skipped stage 2 is correct

* H1 needs IRA to beat *every* comparator (an intersection–union test). Failing against `gru_narrow` — 1 of 4 shifts,
  as in attempt 1 — is therefore enough for "not supported". The skipped comparators could only have made H1 harder.
* The skip was preregistered and carried out as written.
* What it leaves unknown is only how the broad and relational comparators would have done on confirmatory universes.
  Every statement about them must stay labelled as dev (see P-5).

### P-9 — Claims that survive, after the corrections above

1. **H1 failed in both attempts** (1 of 4 shift conditions each). REJECT, as preregistered, in this environment
   family.
2. **The role-switch advantage is the surprise-reset rule.** In both attempts (+0.173 / +0.214) it is explained by M3
   (+0.180 / +0.221). That is component (iv), implemented as a designer-set change-detection heuristic.
3. **The cross-world library helps in-distribution.** It improves `iid` return over within-world inference (M5
   +0.083 / +0.135). This is ordinary hierarchical Bayesian transfer.
4. **Information gain did not help.** In this environment family, at the selected β, information-gain-directed action
   selection was no better than greedy exploitation (M4 −0.016, all five universes ≤ 0). More weight on information
   gain (β = 1) was worse in every condition.
5. **Two mechanisms were not demonstrated:** creating new concepts (2 of 500 worlds) and additive accumulation
   (A +0.027 / +0.006).
6. **The flat RL² GRU, attempt by attempt:**
   * attempt 1: it matched or beat IRA under 3 of 4 shifts;
   * attempt 2: under 2 of 4, and under cue reversal it lost heavily in 2 of 5 universes;
   * both attempts: it failed under the mid-episode switch (post-switch ≈ 0.10).
7. **Novelty:** no new principle, algorithm or empirical finding. This stands.
