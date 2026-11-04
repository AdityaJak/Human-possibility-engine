# Claims policy

Every public statement about Human Possibility Engine (README, investor brief, UI copy, reports)
must fall in one of these categories and carry its evidence.

| Category | Allowed wording | Evidence required |
|---|---|---|
| Engineering fact | "The simulator runs 100 agents for 80 simulated years in N s on <hardware>." | A committed run manifest or benchmark record produced by code in this repo. |
| Synthetic research result | "In our synthetic world, the developmental learner's held-out accuracy exceeded the memory-only baseline by X (95% CI [a, b], 10 seeds)." | Preregistered experiment definition + raw logs + regenerated report. Always followed by the synthetic caveat. |
| Human-data result | "On held-out participants from <dataset>, model A's choice log-likelihood was …" | Licensed dataset with verified terms, documented split, committed evaluation report. |
| Hypothesis | "We hypothesise that …" | None, but must be labelled as hypothesis. |
| Illustration | "Illustrative scenario" | Must be visibly labelled; never contains numeric probabilities. |

## Prohibited unless the cited evidence exists (it does not today)

* "scientifically validated", "human-like intelligence", "causally accurate", "predicts your future",
  "digital twin of you", "10,000 agents running" (until a real 10k benchmark record exists),
  "deployed" (until a real deployment URL exists), any individualized success probability
  (e.g. "82% chance your startup succeeds").
* Treating a simulated effect as a real-world causal effect.
* Treating the simulator's own ground truth as independent human validation.
* Citing a paper, dataset, or URL that has not been checked to exist and say what we claim.

## Mandatory caveats

* Synthetic results: *"Simulated result from a synthetic world. Effects may simply reflect assumptions
  built into the simulator and are not estimates of real-world causal effects."*
* System-wide (until Gate 4 passes): *"Synthetic research prototype; not validated for predicting
  human futures."*
* Agents: simulated age, affect proxies, birth and death are computational abstractions; they do not
  imply consciousness, experienced emotion, or moral personhood.

## Negative results

Negative, null and inconclusive results are published with the same prominence as positive ones.
A failed preregistered test is reported as failed; the primary metric is never switched after
seeing data. Post-hoc analyses are labelled exploratory.
