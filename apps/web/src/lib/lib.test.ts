import { describe, expect, it } from "vitest";
import type { OutcomeObservation, Scenario } from "../types/contracts";
import { simulatedReply } from "./conversation";
import { curvesFromOutcomes, flattenNumeric, gini, perSeedValues, summarize } from "./stats";
import { describeEvent } from "./events";
import { isBaselineComparison, isIntervention } from "./experiments";

const scenario: Scenario = {
  scenario_id: "s",
  decision_id: "d",
  option_id: "o",
  title: "Startup",
  narrative: "In this illustrative scenario you build a prototype. You talk to customers.",
  uncertainties: ["whether customers will pay"],
  cannot_predict: ["whether the company succeeds"],
  reversible_experiments: [{ title: "Ten customer interviews", description: "Talk to ten potential customers.", duration_days: 14 }],
  calculations: [{ name: "Cash runway", formula: "savings / (expenses - income)", inputs: { savings: 6000, expenses: 2000, income: 500 }, result: 4, unit: "months", evidence_id: "e" }],
};

describe("template conversation", () => {
  it.each(["What should I try first?", "What are the biggest risks?", "What about money?", "Would I be happier?", "hello"])("discloses simulation for %s", (q) => {
    const r = simulatedReply(q, scenario, "Startup");
    expect(r).toMatch(/not your real future self/);
    expect(r).not.toMatch(/\d+\s?%/);
  });

  it("never chooses for the user", () => {
    expect(simulatedReply("Which one should I choose?", scenario, "Startup")).toMatch(/can't choose for you/);
  });

  it("uses only shown calculations for money questions", () => {
    expect(simulatedReply("Can I afford it?", scenario, "Startup")).toContain("savings / (expenses - income)");
  });

  it("redacts probabilities even if a scenario contained one", () => {
    const r = simulatedReply("hi", { ...scenario, narrative: "You have a 90% chance of success." }, "Startup");
    expect(r).not.toContain("90%");
  });
});

describe("stats helpers", () => {
  it("summarize computes mean and normal-approx interval", () => {
    const s = summarize([1, 2, 3]);
    expect(s.mean).toBe(2);
    expect(s.n).toBe(3);
    expect(s.lo).toBeLessThan(2);
    expect(s.hi).toBeGreaterThan(2);
  });

  it("curvesFromOutcomes averages within seed before across seeds", () => {
    const o: OutcomeObservation[] = [
      { metric: "acc", value: 0.2, arm: "a", seed: 1, age_years: 5 },
      { metric: "acc", value: 0.4, arm: "a", seed: 1, age_years: 5 },
      { metric: "acc", value: 0.9, arm: "a", seed: 2, age_years: 5 },
      { metric: "other", value: 9, arm: "a", seed: 1, age_years: 5 },
    ];
    const c = curvesFromOutcomes(o, "acc");
    const pt = c.get("a")?.[0];
    expect(pt?.n).toBe(2);
    expect(pt?.y).toBeCloseTo((0.3 + 0.9) / 2);
  });

  it("perSeedValues groups by arm and seed", () => {
    const o: OutcomeObservation[] = [
      { metric: "m", value: 1, arm: "control", seed: 0 },
      { metric: "m", value: 3, arm: "control", seed: 0 },
      { metric: "m", value: 5, arm: "t", seed: 0 },
    ];
    const p = perSeedValues(o, "m");
    expect(p.get("control")).toEqual([{ seed: "0", value: 2 }]);
  });

  it("flattenNumeric produces dot paths", () => {
    expect(flattenNumeric({ a: { b: 1, c: "x" }, d: 2, e: true })).toEqual({ "a.b": 1, d: 2 });
  });

  it("gini is 0 for equality and positive for inequality", () => {
    expect(gini([1, 1, 1])).toBeCloseTo(0);
    expect(gini([0, 0, 10])).toBeGreaterThan(0.5);
  });
});

describe("event descriptions", () => {
  it("describes assessments from payload only", () => {
    const d = describeEvent({
      seq: 1,
      run_id: "r",
      tick: 12,
      day: 360,
      type: "assessment",
      agent_id: "a1",
      payload: { family: "sort", accuracy: 0.61, ci_low: 0.5, ci_high: 0.7, n_items: 60 },
      provenance: { source: "evaluation", mechanism: "tasks.assess" },
    });
    expect(d).toContain("Held-out assessment");
    expect(d).toContain("0.61");
    expect(d).toContain("60 held-out items");
  });
});

describe("experiment classification", () => {
  const base = { experiment_id: "e", title: "t", hypothesis: "h", falsification_criterion: "f", control_arm: "a", seeds: [0], base_config: {}, analysis: { primary_metric: "m", primary_metric_definition: "d", comparison: "c" } };
  it("classifies baseline comparisons and interventions", () => {
    expect(isBaselineComparison({ ...base, arms: [{ name: "a", policy_class: "static_persona" }, { name: "b", policy_class: "developmental" }] })).toBe(true);
    const iv = { ...base, arms: [{ name: "a", policy_class: "developmental" as const }, { name: "b", policy_class: "developmental" as const, interventions: [{ name: "x", kind: "tutoring_access" as const }] }] };
    expect(isIntervention(iv)).toBe(true);
    expect(isBaselineComparison(iv)).toBe(false);
  });
});
