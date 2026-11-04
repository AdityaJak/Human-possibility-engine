# Investor demo — 5–7 minute walkthrough

> Synthetic research prototype; not validated for predicting human futures.
> Every number below is read from a committed artifact (path given). The live pages compute
> them from the API, which reads the same artifacts — nothing is typed into the UI.

## Setup (one machine, no API keys, works offline)

```bash
make setup          # uv venv + pip install -e ".[dev,parquet]" + npm ci
make demo           # API on :8000 + built web app on :5173 (OFFLINE=1 for no-network mode)
open http://127.0.0.1:5173/demo
```

Offline mode serves **real stored outputs** produced by code in this repository
(`reports/runs/`, `reports/experiments/`, `reports/human_validation/`), labelled
"Stored real output" in the UI. Nothing in the demo is hand-authored data.

## The route (≈7 minutes)

| # | Minute | Page | What to say | What the investor sees (artifact) |
|---|---|---|---|---|
| 1 | 0:00–0:45 | `/` | People can't test how today's choices shape their future. We are building the science first, honestly: here is what is built and what is unproven. | Landing page with built-vs-unproven lists and the live claim counts from the Evidence ledger. |
| 2 | 0:45–2:00 | `/lab/society` | A 100-agent society, 80 simulated years, no language model anywhere in the loop. Agents are born into households, go to school, practise tasks built from novel symbols no model has ever seen, work, form ties, age and die. | Map replay of `reports/runs/demo_developmental_100a_80y_s42` (138,607 events; replay from seed and from the year-40 checkpoint are bit-identical: `replay_verification.json`). |
| 3 | 2:00–3:15 | agent inspector (pick any agent) | Skill is never self-reported. Once a simulated year we test each agent on 60 held-out symbols per task family with learning switched off. Here is its skill curve, the learned parameters that changed, and the never-practised family staying at chance. | Held-out skill curves with 95% intervals vs chance; parameter-change chart; sourced timeline with mechanism provenance. Population pattern (same run): mean held-out accuracy rises from near chance in infancy to a peak at 18–30 and declines in old age; the never-practised `barter` family stays at exactly 0.50 at every age. |
| 4 | 3:15–4:30 | `/lab/experiment` | Twin societies: identical populations and random numbers; only tutoring access differs. And a mechanism check: a society that cannot learn must show no tutoring effect. | `reports/experiments/exp02_twin_tutoring/` — per-seed dots, paired CI, sensitivity, assumptions & validity panel, raw-log download. |
| 5 | 4:30–6:00 | `/lab/learning`, `/lab/evidence` | The decisive question: does an agent that develops from experience beat a written persona, a memory-based agent and plain reinforcement learning? **On within-family skill it does not yet beat plain RL** — that preregistered test failed. Its measured advantage is transfer to a new task family, and it correctly transfers *negatively* when structure isn't shared. Real human data: our causal pipeline recovers the published HeartSteps effect. | H1 **failed**: +0.012 [−0.008, +0.031] vs plain RL (`reports/experiments/exp_baselines_heldout/result.json`). Exp6 transfer **passed**: +0.090 [+0.050, +0.123] vs specialists, +0.049 vs plain RL; orthogonal control −0.092 (`exp06_transfer`). HeartSteps V1: +12.2% [−0.5%, +27.4%] vs published +14% (`reports/human_validation/exp07b_causal_grounding`). |
| 6 | 6:00–7:00 | `/future/welcome` | The consumer expression of the science — kept separate from it. Consent first, your values, 2–4 options, illustrative scenarios, deterministic calculations from your own numbers, explicit "cannot predict", and small reversible experiments. No percentages, ever. | 10-step Future Self flow; every scenario labelled "Illustrative scenario"; evidence types colour-coded; crisis and high-stakes questions routed to professional help. |

## Reproduce one stated result in front of the investor (Gate 6)

```bash
# (a) regenerate the H1 result from raw logs — byte-identical result.json
python -m hpe.evaluation analyze reports/experiments/exp_baselines_heldout
git diff --stat reports/experiments/exp_baselines_heldout     # expect: no changes

# (b) replay the 100-agent, 80-year society from its seed and from the year-40 checkpoint (~3 min)
python -m hpe.cli verify-replay reports/runs/demo_developmental_100a_80y_s42   # "passed": true

# (c) re-run the transfer experiment from scratch (~2-6 min on 4 CPUs)
python -m hpe.evaluation run experiments/exp06_transfer.yaml --out /tmp/rerun --workers 3
```

## 2-minute screen recording — script and shot list (not a recording)

| Shot | Time | Screen | Voice-over |
|---|---|---|---|
| 1 | 0:00–0:12 | Landing hero, scroll to "Built / Unproven" | "Human Possibility Engine. We start with science you can check — including where it fails." |
| 2 | 0:12–0:35 | Live society map replay, scrub across decades | "A hundred software agents, eighty simulated years, no language model. Every event is logged and replays bit-for-bit." |
| 3 | 0:35–0:58 | Agent inspector: skill curves, parameter changes | "Skill is measured on symbols the agent has never seen, with learning switched off. Here are the parameters that changed." |
| 4 | 0:58–1:20 | Twin-society experiment: per-seed dots and CI | "Twin societies differ only in tutoring access. A society that cannot learn shows no effect — the mechanism check." |
| 5 | 1:20–1:42 | Proof of learning: H1 failed badge, transfer chart | "Our preregistered test against plain RL failed. Where development helps is learning new tasks faster. We publish both." |
| 6 | 1:42–2:00 | Future Self: consent → scenarios with 'Illustrative scenario' labels | "And for people: explore your options, see what can't be predicted, and take small reversible steps." |

Recording tips: 1440×900 window, light theme, `prefers-reduced-motion` off, hide bookmarks bar,
start the API with `HPE_REPORTS_DIR=reports` so only stored real outputs appear.
