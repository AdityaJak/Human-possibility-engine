# Causal Decision Research — Phase III (Gate G)

_Director's report for the Causal Learning Researcher role, written 2026-11-04. It summarises what the
committed retrospective evidence can and cannot show, separates prediction from intervention effects
for the Phase III models, and specifies the prospective study Gate G requires. **No Gate G evaluation
was run in Phase III; none could be, because Gate G needs real people randomised to model-guided
decisions.**_

## 1. Prediction is not decision improvement

Every Phase III model (candidates F1–F5 and all comparators) estimates `p(next choice | the person's
history)`. A model that predicts choices well tells us **what a person is likely to do**, not **what
would happen if they did something else**. Three gaps separate the two:

1. **Counterfactual outcomes are unobserved.** Psych-101 records the outcome of the chosen option (and
   of others only where the experiment showed them). Predicting the choice says nothing about whether
   an alternative would have served the person's goals better.
2. **Behavioural fit can reward bias.** A model that reproduces a person's systematic mistakes (e.g.
   perseveration in a restless bandit) predicts them better than one that does not. Using it to
   recommend actions would reinforce the mistake.
3. **Advice changes behaviour.** Once a model's output is shown to the person, the data-generating
   process changes; the predictive fit measured on unadvised behaviour no longer applies.

Consequently Gate C (prediction) and Gate G (decision support) are independent. A Gate C pass would
justify using the model as a *description* of how people learn in these tasks; it would not justify
any advice to anyone. This is stated wherever Phase III results are shown (`PHASE3_RESEARCH_RESULTS.md`,
the dashboard).

## 2. What the committed retrospective evidence shows (Phase II, unchanged)

Source: `docs/phase2/CAUSAL_VALIDITY_REPORT.md`, artifacts under `reports/phase2/causal/`. Data:
HeartSteps V1 micro-randomised trial (CC-BY-4.0; Klasnja et al., 2019), 37 participants.

| Question | Result | Status |
|---|---|---|
| Is the estimator (WCLS, cluster-robust) valid at this scale? | bias ≤ 0.003; 95 % coverage 0.932–0.962; type I 0.044 over 500 simulated trials per scenario | validated by simulation |
| Do suggestions raise 30-minute step counts? | DR always − never on the test partition: +0.167 log-points [+0.018, +0.320] | confirmatory (C-002, with recorded deviations) |
| Is there reproducible heterogeneity across five pre-listed contexts? | weekend moderator on test: −0.088, p = 0.667 | not supported (C-001) |
| Does a dev-learned tailored policy beat always-send? | DR contextual − always: +0.021 [−0.107, +0.156] | no evidence of benefit from tailoring |

What this means for Phase III: at the one real randomised dataset available, the intervention helped
on average but **personalising when to intervene added nothing measurable** with 37 people. Power for
moderator effects of 0.20 was 50–74 %, so this is weak evidence of absence, not proof.

## 3. Could Phase III's datasets support causal claims? No.

Psych-101 studies randomise task parameters (reward schedules, options) but not advice or
interventions on the participant. Within them we can estimate how outcomes affect subsequent choices
(the task randomises outcomes), which is a causal statement about *learning*, not about the benefit of
any recommendation. We therefore make no causal decision-support claim from Psych-101 data.

## 4. Gate G protocol (designed, not executed)

**Question.** Do people who explore a decision with model-guided, uncertainty-aware scenario support
(the Future Self flow backed by a validated adaptive model) make choices that lead to better outcomes,
by their own pre-stated goals, than people with the same flow but without model guidance?

**Design.** Two-arm, individually randomised, prospective, pre-registered trial (or a micro-randomised
trial if the decisions recur, as in HeartSteps).

| Element | Specification |
|---|---|
| Population | consenting adults facing a recurring, low-stakes, reversible decision with a measurable outcome (e.g. a 6-week habit experiment chosen by the participant); exclusions: clinical, financial-advice, legal or crisis decisions |
| Arms | (A) Future Self flow with model-guided comparison of options and uncertainty; (B) identical flow and effort with the model's guidance replaced by a neutral structured-reflection prompt |
| Randomisation | 1:1, stratified by decision domain; allocation concealed; outcome assessor blind |
| Primary outcome | goal attainment on the participant's own pre-registered metric at 6 weeks, standardised within domain |
| Secondary | decision regret (Brehaut et al. 2003 scale), calibration of the participant's own forecasts, autonomy (participants rate whether they felt in control), adverse events |
| Sample size | to detect d = 0.25 at α = 0.05 (two-sided), power 0.8: ≈ 253 per arm; recruit 300 per arm for attrition |
| Analysis | intention-to-treat; pre-specified covariates; heterogeneity only as pre-listed secondary analyses with multiplicity control |
| Ethics | institutional ethics review before recruitment; informed consent; data minimisation; the right to withdraw and delete; no deception; a safety monitor with stopping rules for harm signals |
| Prerequisite | a model that has passed Gates A–C on held-out humans. Without it, arm A would deliver unvalidated guidance and the trial should not run |

**Pass condition (fixed here):** arm A beats arm B on the primary outcome with the 95 % CI excluding 0
and no increase in adverse events or loss of perceived autonomy.

## 5. Status

| Item | Status |
|---|---|
| Gate G | **not started** — requires the prospective study in §4, ethics approval and a Gates A–C model |
| Retrospective causal evidence | Phase II results stand (estimator valid; average suggestion effect positive; tailoring adds nothing measurable) |
| Prediction vs intervention separation | enforced in all Phase III reports and the dashboard copy |
