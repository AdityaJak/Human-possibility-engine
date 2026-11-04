# Scientific assumptions — and why "simulating a life" does not establish human fidelity

## What the simulator is

A deterministic, event-driven state machine. Agents are software processes with persistent state,
learnable parameters and bounded memory. Their "life" is a sequence of ticks in which they choose
activities, practise tasks with novel symbols, receive feedback, interact, earn abstract resources,
age, reproduce and are removed from the population by a modelled mortality hazard.

## Why that alone proves nothing about people

1. **Every mechanism is an assumption.** School gives more practice; tutoring gives instructional
   feedback; parents transmit skill through demonstrations. If tutored agents end up more skilled,
   that follows from the mechanisms we wrote, not from discovering something about humans.
2. **The simulator's ground truth is not independent.** An agent can match the world's hidden rules
   perfectly and say nothing about human cognition. Only held-out *human* data can test that (Gate 4).
3. **Believable narrative is not evidence.** A biography, a dialogue, or an LLM speaking as a child
   is not developmental competence. We measure competence only on held-out task items with
   learning disabled.
4. **Scripted age effects can masquerade as development.** Age gates (what an agent can perceive,
   how plastic its learning is) are scripted. Every policy class gets the same gates, and ablations
   switch them off, so that improvement attributable to learning can be separated from improvement
   attributable to the script.
5. **Agent count is not a result.** Running 10,000 agents is an engineering benchmark, not evidence
   of realism.
6. **Synthetic interventions are not causal estimates for the real world.** A twin-society
   experiment identifies the effect *of the simulator's mechanisms*. Real causal claims need
   randomised human data or explicit identification assumptions (Experiment 7).

## What it can legitimately show

* Whether a learning mechanism acquires skill that generalises to unseen symbols and task families
  under controlled, matched conditions, compared to strong non-developmental baselines.
* Whether simulated interventions act *through* a mechanism (e.g. no tutoring effect on skill for a
  non-learning policy) rather than through hard-coded outcomes.
* Reproducibility, cost, and failure modes of the machinery we would need for real-human research.

## Key modelling assumptions (non-exhaustive; each lane documents its own)

| Area | Assumption | Where | How it is checked |
|---|---|---|---|
| Tasks | Rules are linear readouts of a shared 3-D latent projection of 12 binary features | `hpe.tasks` | orthogonal control world; plain linear baseline can learn every family |
| Perception | Features visible grows linearly from 4 at age 0 to 12 at age 10 | `hpe.tasks` | `enable_age_gate=False` ablation |
| Learning | Plasticity higher in childhood, lower in old age; forgetting without practice | `hpe.development` | ablation flags |
| Society | Time step 30 days; school 6–18; Gompertz mortality; abstract wages | `hpe.world` | sensitivity sweeps |
| Affect | Stress/energy/motivation are scalar control proxies, not emotions | `hpe.world` | — (labelling) |
