# Claims and limitations

The authoritative, machine-generated list of claims is [`docs/EVIDENCE.md`](EVIDENCE.md)
(`python -m hpe.ledger`). This page explains how to read it and what is out of bounds.

## What we can claim today (with evidence)

1. **Engineering.** A 100-agent society runs 80 simulated years with no language model and replays
   bit-identically from its seed and from a mid-run checkpoint (`reports/runs/*/replay_verification.json`).
   Experiments regenerate their results byte-identically from raw logs, and a full from-scratch
   re-run of Exp 6 reproduced every raw outcome log (0 of 40 differed).
2. **Learning is real in the synthetic world.** Held-out skill (novel symbols never practised, learning
   disabled during assessment) depends on parameter learning and feedback: freezing learning removes
   +0.41 accuracy, withholding feedback removes +0.27 (`reports/experiments/exp_ablations`).
3. **The decisive comparison is mixed.** Against the strongest baseline (plain per-family RL) the
   developmental learner's within-family held-out accuracy is **not** better (H1 failed:
   +0.012, 95% CI [−0.008, +0.031]). It **is** better at adapting to an unseen task family that shares
   structure (Exp 6: +0.090 vs specialists, +0.049 vs plain RL), and it transfers **negatively** when the
   structure is not shared (−0.092) — a cost, not just a benefit.
4. **One real causal estimate.** From the HeartSteps V1 micro-randomised trial (CC-BY-4.0), our estimator
   gives +12.2% [−0.5%, +27.4%] for activity suggestions, consistent with the published +14%
   (`reports/human_validation/exp07b_causal_grounding`). This validates a pipeline, not our agents.
5. **Real human choices, mixed.** On held-out Psych-101 participants the developmental learner is the
   best of eight models on the Iowa Gambling Task (−0.110 nats vs the instance-based reference,
   95% CI [−0.140, −0.083]) but clearly worse than exemplar-memory models on the preregistered primary
   task, multiple-cue learning (+0.229 [+0.171, +0.289]). We may say the former only alongside the latter.

## What we cannot claim (and the UI never says)

* That agents are human-like, conscious, or feel anything. Affect variables are control proxies.
* That simulated intervention effects (e.g. tutoring) estimate real-world effects. They show what the
  simulator's mechanisms imply. The mechanism checks (a non-learning society shows no skill effect)
  establish only that effects flow through learning *inside the model*.
* That any result is validated for predicting a real person's future. Gate 4 (human validity of the
  developmental learner) is not passed; see `reports/human_validation/exp07_human_heldout`.
* That anything is *scientifically* established at 10,000 agents. What exists is an engineering
  benchmark: 10,000 agents ran one simulated year in 720.7 s with 5.1 GB peak RSS (`reports/scale/`).
  Agent count is not a result, and multi-year 10k runs have not been measured.
* Any individualised probability in Future Self. Scenarios are illustrative; calculations only use the
  person's own inputs; unknowns are shown as "cannot predict".

## Known limitations and threats to validity

| Threat | Where it bites | Mitigation / status |
|---|---|---|
| Rules in the task world are linear in raw features | A plain linear learner can learn every family; shared representation can only help via sample efficiency | Documented (ADR-0003); orthogonal control world |
| Scripted age gates (perception, plasticity, forgetting) | Lifespan skill curves partly reflect scripts, not learning | Applied identically to all policy classes; ablations + `no_age_gate` sensitivity; removing the plasticity schedule *improved* accuracy |
| Developmental learner hyperparameters tuned on task seeds 7 and 21 | Seed 7 is also an evaluation seed (0–9) | Independent audit recomputes primary contrasts excluding seed 7 (`reports/audit/`) |
| Baselines could be weak | Inflates H1 | plain_rl tuned on separate seeds 1001–1003; H1 compares against the strongest baseline chosen post hoc (conservative) |
| Society parameters are invented | All society experiment effect sizes | Sensitivity sweeps; synthetic disclaimer on every result |
| Psych-101 human data was used to train Centaur | Not a leakage risk for us, but context for comparisons | Stated in the report |
| HeartSteps public data has 37 of the paper's 44 participants; exact covariates unverified | Causal-grounding comparison | Stated; complete-case sensitivity reported |
| Single machine, CPU only | Scale numbers | Hardware and concurrent load disclosed with each benchmark |
