# World modelling assumptions (sim/world, v0.1.0)

Everything here is an **abstract, uncalibrated modelling choice**. None of it is an empirical
claim about real people, labour markets, schools or families. Simulated outcomes may simply
reflect these assumptions; they are not estimates of real-world effects. Agents' "birth",
"death", "stress" and "energy" are computational state variables, not experiences.

## 1. Time
* One tick = `dt_days` (default 30). `year = tick * dt_days / 365.25`. An 80-year run at the
  default step is 974 ticks. Per-tick rates were specified for 30-day ticks and are scaled by
  `dt_days / 30` (episode counts) or `min(1, dt_days / 30)` (relaxation rates).
* Agents under 3 make no activity decision (they "rest" and receive caregiving); this is the
  event-based saving for idle agents. All other alive agents decide every tick.

## 2. Initial population and pre-history
* `young_families`: ~15% elder households (62–85), the rest couples aged 22–47 with children
  aged 0–15 distributed round-robin. `uniform_0_60` and `newborn_cohort` are alternatives.
* Traits (curiosity, diligence, sociability, risk_tolerance, patience), base learning
  parameters and value weights are random draws; no demographic attribute is modelled.
* **Pre-history warm-up:** agents who already exist at t=0 receive a condensed, seeded practice
  history (≤ 400 episodes, scaled by `min(age, 25) / 25`, chronological, school-like episodes at
  school ages and job-like episodes afterwards) so initial adults are not tabula rasa. It goes
  through `hpe.tasks.practice` and is applied identically to every policy class (a
  non-learning policy is unaffected). Constant `PREHISTORY_MAX_EPISODES`.
* Newborns inherit *initial conditions* only — traits, base parameters, values (parents' mean
  + noise). Their policy is freshly constructed: **no knowledge is inherited**
  (`inherited.knowledge_inherited = false`). Skill transmission can only happen through
  mechanisms (caregiving demonstrations, household resources → access).

## 3. Development and learning inputs
* **Perceptual age gate** (`enable_age_gate`): `hpe.tasks` zeroes features an agent cannot yet
  perceive (4 of 12 at birth, all 12 by age 10). Scripted, identical for every policy class.
* **Caregiving (ages 1 to school start):** each tick a household adult (parents first) solves
  `caregiver_episodes_per_tick` train items greedily; the child receives a `Demonstration` of
  the adult's action and its (noisy) bandit outcome, weighted by trust. A more skilled
  caregiver therefore transmits better examples — a mechanism, not a hard-coded effect.
* **School** (ages `school_start_age`..`school_end_age`, unless access is blocked):
  `school_episodes_per_tick` episodes rotating over the three training families; a
  `school_full_feedback_fraction` share is instructional (full feedback), the rest bandit.
* **Tutoring:** extra `tutoring_episodes_per_tick` *full-feedback* episodes on the child's
  weakest measured family, with item difficulty matched to the child's last measured accuracy.
  Tutoring is therefore an intervention *package* (more instruction + adaptive difficulty).
* **Self-study:** the agent chooses a family (`kind="practice"`), then gets
  `self_study_episodes_per_tick` bandit episodes (adult education under the midlife
  intervention: full-feedback, difficulty-matched).
* **Work:** `work_episodes_per_tick` bandit episodes on the job's family. Wage per worked tick =
  `wage_base * (0.5 + share correct) * dt/30` (×0.7 in a downturn). Unemployed adults who choose
  work do gig work at 30% pay on a rotating family.
* **Fatigue:** energy < 0.25 halves every episode count that tick.
* The barter family is never practised anywhere; its assessment is zero-shot transfer.

## 4. Social mechanisms
* Enrolled children meet one random schoolmate per tick, employees one coworker: edge
  interaction count +1 and slow trust drift toward 0.6.
* Agents who choose to socialise are matched in pairs at the plaza (50% chance to prefer a
  friend who is also there). Each meeting is a **stag-hunt trust game** (`kind="social"`;
  payoffs: both cooperate 1.0, cooperate vs withhold 0.0 / 0.4, both withhold 0.4) and an
  exchange of `peer_demonstrations_per_social_tick` demonstrations each way, weighted by trust.
  Trust moves 20% toward the partner's observed cooperation.
* Friendship forms when both directions have ≥ 3 interactions and trust ≥ 0.6.
* New hires get a mentor (the most skilled current holder of a same-family job) who gives one
  demonstration per worked tick for ~6 months.
* Strangers met at the plaza become internal "acquaintance" ties. The v0.1 contract enum has
  no such kind, so `World.edges()` omits them (contract-change request filed).

## 5. Economy
* Resources are pooled per household. Consumption per tick: `consumption_per_member` per adult,
  60% of that per child. Pensions (45% of `wage_base`) for agents past `retirement_age`;
  unemployment benefit (25%) for non-enrolled adults without a job who did not work this tick.
* **No debt:** household resources are floored at 0; unmet need raises members' stress.
* Job count = `n_jobs_per_100_agents` × population / 100, recomputed yearly, split evenly across
  the three workplaces/families. Yearly hiring ranks unemployed working-age adults by their
  latest *measured* held-out accuracy in the job's family (no extra assessment, no learning).
* Yearly separations: absence on more than half the year's ticks, 4% random layoff (12% in a
  downturn), retirement.

## 6. Lifecycle
* Stages: infant < 3, child < 12, adolescent < 18, adult < 65, elder.
* Mortality hazard per year `a * exp(b * age)` (Gompertz; ×3 while ill); one uniform is drawn
  per agent per tick from its own `life` stream (common random numbers); hard cap at
  `max_age_years`.
* Partnerships: (i) a plaza meeting between unrelated single adults aged 20–45 with mutual
  trust ≥ 0.6 and ≥ 4 interactions forms a partnership with probability 0.25/tick; (ii) a
  background partner search (hazard 0.2/year for singles aged 20–40) chooses an unrelated
  single within 8 years of age, weighting existing ties by trust. Mechanism (ii) keeps
  demography roughly policy-neutral so that population size does not confound comparisons of
  learning policies. A new couple forms a new household; each partner's dependent children (< 18) move with them, and each mover brings a per-capita share of the old household's resources. Children whose household loses its last adult stay in place (no adoption mechanism in v0.1) and receive no caregiving demonstrations.
* Births: couples aged 20–45 living together, ≤ 4 children, annual rate
  `annual_birth_rate_per_couple × clip(2 (1 − pop / max_population), 0, 1.5)`.
* Death: job freed, partner widowed; when a household empties its resources pass to the
  deceased's living children's households (else to an unclaimed commons pool).

## 7. Shocks (seeded, `shock_rate_per_year` Poisson)
* 50% illness (energy −0.4, stress +0.3, mortality ×3 for ~90 days), 25% flood (household
  loses half its resources), 25% downturn (wages ×0.7 and layoffs ×3 for a year).
* Values drift with experience: illness → health, flood/downturn → stability, friendship →
  relationships, successful study → learning (small increments, renormalised).

## 8. Interventions (change inputs/access only, never measured outcomes)
| kind | mechanism |
|---|---|
| `tutoring_access` | tutoring episodes as above (`params.episodes` overrides count) |
| `school_access` | `params.access` (default false) blocks enrolment for targeted agents |
| `midlife_education` | study becomes instructional adult education for targeted adults |
| `social_isolation` | no classmate/coworker/plaza/mentor contact; socialising meets family only |
| `misinformation` | targeted agents' demonstrations assert a wrong answer as correct (full feedback) |
| `cash_transfer` | `params.amount` to the household, `params.period` once / yearly / monthly |
| `caregiving_support` | doubles caregiver demonstrations for targeted young children |
| `cross_foster` | targeted newborns are raised in a random other household with adults; `parents` stays biological |

Assignment: explicit `agent_ids` / `household_ids`, otherwise a stateless seeded draw per
(intervention name, agent) compared with `target.fraction`, evaluated within `[age_min,
age_max]` and `[start_year, end_year)`.

## 9. Measurement
* Every `log.assessment_interval_years` (and at t=0) every alive agent aged ≥ 2 is assessed on
  the fixed held-out set of every family via `hpe.tasks.assess` (learning disabled). Hiring
  uses these measurements. Self-estimates are logged alongside, never used as skill.
* `employment_rate` is over non-enrolled agents aged 18 to retirement; `resource_gini` over
  non-empty households' (non-negative) resources.

## 10. Determinism
* RNG keys: per agent `act`, `learn`, `life`, `family`, `prehistory`; global `social/matching`,
  `shocks`, `market`, `map/homes`; stateless keyed draws for intervention assignment, baseline
  tutoring and birth trait noise. Agents are processed in sorted id order.
* `event_stream_hash` is a SHA-256 chain over each event's canonical JSON *without* `run_id`.
  Frame jitter uses a stateless integer hash, so requesting frames never changes dynamics.
