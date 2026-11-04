# ADR-0004: Policy classes and budget matching
Status: accepted (2026-11-04)

**Decision.** Five interchangeable classes implement one `Policy` protocol: `static_persona` (fixed trait-derived heuristic, no learning), `memory_only` (bounded episodic store + fixed retrieval rule), `plain_rl` (independent per-family online learner — the "simple RL baseline" the spec requires), `developmental` (proposed), `random` (reference). Baselines are owned by the evaluation lane, the proposed learner by the development lane, so neither author grades their own work. All classes receive identical observation and feedback streams in learning-lab experiments (common random numbers) and identical world inputs in society experiments; compute counters and wall time are reported per class.

**Consequences.** The investor's decisive question (§44) is answered by comparing `developmental` to the strongest of the other three, not to a strawman.
