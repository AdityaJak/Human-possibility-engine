# Scale benchmarks

Each `*.json` file is a `BenchmarkRecord` (see `hpe.contracts.api`) written by
`scripts/bench_scale.py` after actually running the simulator. Records disclose hardware and the
machine's load average before/after the run; numbers measured while other jobs were running are
slower than on a quiet machine and are labelled by their load average, not adjusted.

| Record | What it measures |
|---|---|
| `developmental_100a_2y_dt30_s0.json` | 100 agents, 2 simulated years, 30-day ticks, developmental learner |
| `static_persona_100a_2y_dt30_s0.json` | same configuration, non-learning static-persona baseline |

**Not yet measured:** 1,000-agent and 10,000-agent runs. Until a record with `status: completed`
exists here for those sizes, no claim about them may be made (docs/CLAIMS_POLICY.md). Run them on a
quiet machine with:

```bash
make bench-1k     # 1,000 agents x 10 years, timeout 2 h
make bench-10k    # 10,000 agents x 1 year, timeout 1 h, cProfile top-25 written next to the record
```

Timeouts and crashes are recorded too (`status: timeout|failed`) — a failed benchmark is a result.
What is measured: `world.run()` wall time with events counted in memory (not written to disk),
`decision_sample_rate=0`, no map frames, yearly held-out assessments, no paid model calls ($0).
