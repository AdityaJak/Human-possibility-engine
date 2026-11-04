# ADR-0003: Novel-symbol task world and held-out measurement
Status: accepted (2026-11-04)

**Context.** Spec §12/§37: developmental competence must be measured by held-out tasks with novel rules unknown to any pretrained model; "an LLM prompted 'you are 5' is not a trained five-year-old".

**Decision.** `hpe.tasks.SymbolWorld` generates, per task seed, 96 random binary-feature symbols with nonsense names, split 64 train / 32 holdout. Hidden rules for four families (`sort` 3-way, `compare` 2-way pair, `signal` 2-way, `barter` 2-way pair) are linear readouts of one shared latent projection `z = A x` (k=3, d=12). `barter` is never practised during development: it is the transfer family. In `transfer_mode="orthogonal"`, `barter` uses a projection orthogonal to `A` (negative-transfer control). Practice only samples train symbols; assessment only uses a fixed, seed-determined set of holdout items. Accuracy credits ties fractionally, so a uniform policy scores exactly chance. Label noise corrupts feedback, never ground truth.

**Feasibility check (executed 2026-11-04, scratch script, seed 7, 5 learner seeds).** A plain online linear learner climbs from chance to 0.63–0.95 held-out accuracy within 400 episodes depending on family; the tasks are learnable, but there is headroom.

**Consequences.** Linear readouts mean a raw-feature linear learner can in principle learn every family; a shared representation can only help through sample efficiency. This makes H3 a genuine, falsifiable test rather than a built-in advantage.
