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
