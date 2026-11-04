# State-of-the-art benchmark — cognition/behaviour foundation models vs this repository

_Human-validation lane, Phase II. Two kinds of rows are kept strictly apart: **directly reproduced**
(run here, on our frozen dev/tune partitions, by committed code with committed logs) and **published
figure** (copied from a page we actually opened, with its own protocol). The two are never compared
as if they were the same number._

## 1. What exists publicly (verified on 2026-11-04; URLs are the pages opened)

| System | What it is | Weights / code | Licence | Verified source |
|---|---|---|---|---|
| **Centaur** (Binz et al.) | Llama-3.1-70B fine-tuned with low-rank adapters on Psych-101 (10.7 M choices, 160 experiments); predicts held-out participants, modified cover stories and new domains | `marcelbinz/Llama-3.1-Centaur-70B` (30 safetensors shards; also `-adapter`), `marcelbinz/Llama-3.1-RandomInit-70B` control; paper repo `marcelbinz/Centaur` says its code is outdated and points to `Zak-Hussain/LLM4BeSci` | `llama3.1` (model card) | https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B · https://arxiv.org/abs/2410.20268 · https://arxiv.org/html/2410.20268v3 (Nature version at doi 10.1038/s41586-025-09215-4 returned HTTP 429 to us) |
| **Minitaur** | Same recipe on Llama-3.1-8B; "captures human behavior close to its training distribution but generalizes less robustly" (paper, Methods) | `marcelbinz/Llama-3.1-Minitaur-8B` (4 shards, ≈16 GB bf16), `-adapter`, `RandomInit-8B`; third-party quantisations `HillPhelmuth/Llama-3.1-Minitaur-8B-GGUF`, `…-mlx-4Bit` (unverified fidelity) | `llama3.1` | https://huggingface.co/marcelbinz/Llama-3.1-Minitaur-8B |
| **Psych-201** | Successor dataset (208,021 rows; adds RTs, demographics, clinical questionnaires; flags `is_psych101`, `is_psych101_test`, `in_eval`) | dataset only | Apache-2.0 | https://huggingface.co/datasets/marcelbinz/Psych-201 |
| **Small Foundation Models of Human Cognition and Behaviour** (arXiv 2608.05224, Aug 2026) | 14 models, 135 M–14 B parameters, 4 architecture families, trained on Psych-101; abstract: "for in-distribution simulations, scale barely matters … 0.6B to 1B parameters suffice to match a 70B baseline on held[-out …]" | not checked for weights | — | http://export.arxiv.org/api/query?id_list=2608.05224 (abstract) |
| **Cross-Centaur** (arXiv 2610.23157, Oct 2026) | Cross-Psych-101: 591 participants, 261,551 decisions across 12 tasks linking the *same* participants across tasks; individual-level modelling | not checked | — | http://export.arxiv.org/api/query?id_list=2610.23157 (abstract) |
| **"Simile"** | Searched arXiv (`all:"Simile" AND all:"human behavior"`): **no result**. Nothing public that we could open documents such a model; treated as not publicly documented. | — | — | arXiv API query returned an empty feed |

## 2. Published figures (NOT our protocol)

Centaur's held-out protocol (paper, Results): each experiment split **90 % train / 10 % test by
participant**; goodness-of-fit = negative log-likelihood **averaged across responses**, model scoring
the choice tokens of a natural-language transcript. Averages over Psych-101: Centaur 0.44, base Llama
0.58 (t = −144.22). Table 1 of the arXiv HTML gives per-experiment held-out NLLs (Centaur | Llama |
domain-specific cognitive model). Rows below are the experiments that overlap our families; the mapping
from the paper's experiment names to Psych-101 file prefixes is our inference from the names, not
something the paper states.

| Paper experiment name (Table 1) | Centaur | Llama | cognitive model | Likely Psych-101 files (our inference) |
|---|---|---|---|---|
| Multiple-cue judgment | 1.1236 | 1.2818 | 1.9157 | collsiöö2023MCPL |
| Iowa gambling task | 0.8890 | 0.9880 | 1.1555 | steingroever2015data |
| Two-armed bandit | 0.2963 | 0.3829 | 0.4187 | gershman2018deconstructing |
| Horizon task | 0.4032 | 0.5237 | **0.3595** | wilson2014humans (cognitive model better than Centaur) |
| Weather prediction task | 0.5514 | 0.5749 | 0.6267 | speekenbrink2008learning |
| Probabilistic instrumental learning | 0.4937 | 0.5382 | **0.5047** | lefebvre2017behavioural |
| Decisions from description | 0.5336 | 0.7569 | 0.6120 | plonsky2018when (described gambles) |
| Decisions from experience | 0.3686 | 0.4339 | 0.5404 | — |
| Experiential-symbolic task | 0.4536 | 0.6983 | nan | garcia2023experiential |
| Multi-task reinforcement learning | 0.5672 | 0.6604 | 1.0424 | gershman2020reward (our `srl3`) — uncertain |
| CPC18 | 0.3390 | 0.4118 | 0.6607 | — |

Why these are **not directly comparable** to our numbers:

1. **Split.** 90/10 by participant with their seed vs our 40/20/40 dev/tune/test with seed 20261105.
   Different held-out people.
2. **Scoring unit.** Token-level NLL of a text choice (a multi-token answer like `<<somewhat low>>`
   can be scored across tokens; the paper reports per-response averages) vs our per-choice NLL over a
   fixed K-way categorical distribution. Even chance differs: our MCPL chance is ln 9 = 2.197; a
   language model's "chance" is undefined.
3. **Information.** The LLM reads the full instruction text and all prior trials as text; our models
   receive a 12-feature encoding and explicit feedback through a typed protocol.
4. **Forced trials.** We exclude instructed horizon-task trials from scoring; the paper's treatment is
   not stated in the text we opened.
5. **Lapse.** Our NLLs are lapse-mixed with a shared grid; the paper's are raw model likelihoods.

## 3. Can we reproduce them here? (4 cores, 15.7 GiB RAM, CPU only, no paid compute)

| Candidate | Feasible here? | Reasoning |
|---|---|---|
| Centaur-70B | **No.** | ≈140 GB bf16 weights; model card requires 2+ 80 GB GPUs. Not downloadable or runnable in this environment. |
| Minitaur-8B (official bf16) | **No.** | ≈16 GB bf16 exceeds RAM; no official quantised release; base Llama-3.1 licence acceptance is an account action. |
| Minitaur-8B third-party GGUF (`HillPhelmuth/…-GGUF`) | **Possible but not attempted.** | A 4-bit GGUF (~5 GB) would run under llama.cpp on CPU. Scoring one 100-trial IGT transcript (≈2.5 k tokens) is a full prompt evaluation per participant (≈1–3 min on 4 cores); a 5-participant pilot would take ≈15 min plus a 5 GB download and a compile of llama-cpp-python. We did not run it because (a) the quantisation is third-party and its fidelity to the published model is unverified, (b) the result would still differ from the paper's protocol (section 2), and (c) the lane's CPU budget was reserved for the equal-tuning pilot. The procedure is scripted in prose below so a later session can execute it. |
| 0.6–1 B "small foundation models" (arXiv 2608.05224) | **Unknown.** | We only opened the abstract; weights availability and licence were not verified. A 1 B model in 8-bit (~1 GB) would be CPU-feasible if released. |

Procedure for the untried Minitaur GGUF pilot: download `HillPhelmuth/Llama-3.1-Minitaur-8B-GGUF`
(Q4_K_M), `pip install llama-cpp-python`, for each dev participant build the Psych-101 transcript
prefix up to each choice, request `logits_all=True`, read the log-probability of the human's choice
tokens, average per choice; compare against `random` (ln K) and our models on the same participants.
Record wall time, RAM, and the quantisation's provenance in the run log.

## 4. Directly reproduced comparison on our protocol

The table is generated from the dev-pilot run by `scripts/phase2_sota_table.py` and stored at
`reports/phase2/human/p2_human_dev_pilot/sota_table.md`. It is reproduced verbatim in
`HUMAN_VALIDATION_V2.md` §6 together with the parse validation and grids. Rows there are all
**directly reproduced** on the tune partition (selection-optimistic, exploratory); no row in that table
is a published figure.

## 5. Where our architecture wins, loses, or is untested

Directly reproduced (tune partition, exploratory; details and CIs in `HUMAN_VALIDATION_V2.md` §6–§7):

* **Loses** to the ≤ 2 k-parameter GRU fitted on other participants in 8 of 9 families: MCPL
  (+0.445 nats), described gambles (+0.283), garcia (+0.194), bandit2 (+0.172), horizon (+0.169),
  srl3 (+0.113), IGT (+0.083), casino (+0.067); every paired 95 % CI excludes 0.
* **Loses** to classical cognitive models on cue-combination tasks (MCPL: instance-based; garcia:
  Bayesian rule learner) and to PVL-Delta on bandit2 and horizon.
* **Competitive** with classical cognitive models on IGT, srl3 and casino (second behind the GRU),
  and ahead of plain RL and memory-only in 7 of 9 families.
* **Tie** on the weather-prediction task (n = 5 tune participants; −0.007 nats vs the Bayesian rule
  learner, below the 0.02 margin).
* **Candidate A (hybrid exemplar, X-010)** also loses to the GRU on every pristine family (bandit2
  +0.196, garcia +0.178, described +0.269 nats; CIs exclude 0); it beats the exemplar and memory-only
  references everywhere and is level with PVL-Delta on garcia/described (`HUMAN_VALIDATION_V2.md` §6b).
* **Untested**: the test partition (locked; no freeze was granted), any second dataset, and any direct
  run of Centaur/Minitaur (section 3).

Against the published figures (section 2) no comparison is valid: different participants, scoring unit,
information and lapse treatment. Resource cost of our strongest baseline: the GRU fits in 2–20 s of
CPU per configuration (≤ 2 k parameters); the whole pilot — 9 families × 14 models × ≤ 24
configurations × 7 lapse values — used 3,744 CPU-seconds on 4 cores, 0 GPU, 0 API calls.
**No state-of-the-art claim is made.**
