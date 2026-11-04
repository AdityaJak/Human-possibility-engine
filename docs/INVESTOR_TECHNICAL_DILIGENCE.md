# Technical diligence note

_For a technical investor or reviewer. 2–4 pages. Everything verifiable here is verifiable from a clean
checkout; `docs/INVESTOR_DEMO.md` gives the commands._

## 1. What this is

A research platform for one falsifiable hypothesis — that agents which **develop** (acquire skill from
experience under persistent state and social context) behave and learn more like people than agents
with a written persona or a memory store — and a consumer decision-support interface built on the
same scientific standards. Not an LLM wrapper: the simulator, the learners and every experiment run
with no language model and no API key (Tiers 1–2 in `ARCHITECTURE.md`).

## 2. Claimed novelty, and what is *not* claimed

* **Novel-symbol developmental benchmark.** Task families whose rules no pretrained model can know,
  with held-out symbols, a never-practised transfer family and an orthogonal negative-transfer
  control. This removes the "the LLM already knew it" confound that affects prompt-based agent societies.
* **Matched-budget policy classes under one protocol** (static persona, memory-only, plain RL,
  developmental) with common random numbers, preregistered hashes and analysis regenerated from raw logs.
* **Twin-society experiments with mechanism checks** (a non-learning society must show no effect).
* Not claimed: human-level anything; validated prediction of human futures; that simulated
  interventions estimate real effects; scale beyond 100 agents.

## 3. Current evidence (details and CIs in `RESEARCH_RESULTS.md`)

| Claim | Status | One-line evidence |
|---|---|---|
| Learning is real and generalises to unseen symbols | supported (synthetic) | freezing learning −0.41 accuracy; feedback −0.27 |
| Developmental ≫ persona and memory-only | supported (synthetic) | +0.41 and +0.08 held-out accuracy |
| Developmental > plain RL within a family (H1, ≥10%) | **not supported** | +0.012 [−0.008, +0.031]; worse under label noise |
| Shared representation transfers to an unseen family | supported (synthetic) | +0.09 AUC vs specialists; negative when structure differs |
| Tutoring effect flows through learning | supported (synthetic) | +0.029 [+0.022, +0.037]; 0.000 in a non-learning society |
| Causal estimator recovers a randomised human effect | supported with caveats | HeartSteps +12% vs published +14% |
| Agents predict held-out human choices | **mixed; preregistered primary failed** | IGT: best of 8 models (−0.11 nats vs instance-based); MCPL: worse than exemplar models (+0.23 nats) |
| Bit-identical replay of a 100-agent, 80-year society | supported (engineering) | from seed and from checkpoint |
| 1,000 / 10,000 agents | **not tested** | no benchmark record |

## 4. Threats to validity (and where we stand)

1. **Linear task rules.** A raw-feature linear learner is near-optimal within a family, which is the most
   plausible reason H1 failed. Next: non-linear, compositional and non-stationary families (NEXT_STEPS §1).
2. **Scripted age effects.** Perceptual gate, plasticity schedule and forgetting are assumptions applied to
   all classes; ablations show the plasticity schedule hurts. Lifespan curves are not human evidence.
3. **Tuning hygiene.** The developmental learner's defaults were chosen on task seeds 7 and 21; seed 7 is
   an evaluation seed. The independent audit recomputed primary contrasts without seed 7: verdicts unchanged.
4. **Invented society parameters.** Fertility, mortality, wages, schooling are abstract; sensitivity sweeps
   are reported, not calibration.
5. **Human data.** HeartSteps consistency rule is weak (CI includes 0); Psych-101 first pass favoured the
   reference models' parameter grids.
6. **Single-machine, shared-CPU timings.** All wall times disclose load.

## 5. Architecture and engineering quality

* Python 3.12+/numpy/pydantic simulation core; FastAPI; SQLite with namespaced tables and migrations;
  React + TypeScript dashboard and Future Self app; Docker Compose; CI with lint, types, contract and
  OpenAPI freshness checks, unit + integration tests against the real simulator, secrets scan,
  dependency audits, seeded smoke simulation with replay check.
* Determinism: SHA-256-keyed named RNG streams, sorted iteration, event-chain hashing, checkpoint
  resume verified bit-identical. Independent audit: 80/80 experiment jobs re-run bit-for-bit.
* Test suite: ~280 Python tests (+58 web) in the integrated branch; branch coverage 91% (audit).
* Safety in Future Self: consent gate, no-probability validator, crisis/high-stakes routing (two audit
  findings fixed and re-tested with the audit's own script), input limits, export/delete verified.

## 6. Costs

$0 API spend. ≈0.014 CPU-s per agent-year at 100 agents (≈$0.002 per 100-agent 80-year run at an assumed
$0.05/vCPU-h). Scaling cost is dominated by per-agent in-memory histories and per-episode Python overhead;
profiling and vectorisation are the first engineering items for 1k–10k agents.

## 7. Roadmap (funding-ready milestones, spec §30)

A (skill acquisition at 100 agents) — supported in synthetic world. B (beats baselines) — beats two of
three; **plain RL is the open problem**. C (human correspondence) — not yet. D (transfer) — supported in
synthetic world. E (prospective human study) — not started; requires ethics review. Detailed sequencing
in `NEXT_STEPS.md`.

## 8. Diligence checklist

- [ ] `make setup && make test` from a clean checkout (expect all green).
- [ ] `python -m hpe.evaluation analyze reports/experiments/exp_baselines_heldout` → `git diff` empty.
- [ ] `python -m hpe.cli verify-replay reports/runs/demo_developmental_100a_80y_s42` → `"passed": true`.
- [ ] Re-run Exp 6 to a temp dir and diff raw logs against the committed ones (0 differences expected).
- [ ] Read `reports/audit/AUDIT.md` (independent) and `docs/EVIDENCE.md` (generated ledger).
- [ ] Confirm no result number is hard-coded in `apps/web/src` (audit §7).
- [ ] Check `docs/DATA_SOURCES.md` licences against the upstream pages.
- [ ] Ask for the 1k/10k benchmark records; if absent, treat scale as unproven.
