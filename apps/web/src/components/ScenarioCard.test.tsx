import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Scenario } from "../types/contracts";
import { ScenarioCard } from "./ScenarioCard";
import { REDACTION, findProbabilityClaims, redactProbabilityClaims } from "../lib/probabilityGuard";

const base: Scenario = {
  scenario_id: "s1",
  decision_id: "d1",
  option_id: "o1",
  title: "Take the new job",
  narrative: "In this scenario you spend more time commuting but learn a new codebase.",
  tradeoffs: [{ value_key: "income", direction: "likely_better", rationale: "higher stated salary" }],
  cannot_predict: ["long-term satisfaction"],
};

describe("ScenarioCard", () => {
  it("always renders the 'Illustrative scenario' label", () => {
    render(<ScenarioCard scenario={base} />);
    expect(screen.getAllByText(/illustrative scenario/i).length).toBeGreaterThan(0);
    expect(screen.getByText("Illustrative visualization")).toBeInTheDocument();
  });

  it("renders the label even when the API omits the label field", () => {
    const { label: _omit, ...noLabel } = { ...base, label: "illustrative scenario" as const };
    render(<ScenarioCard scenario={noLabel} />);
    expect(screen.getAllByText(/illustrative scenario/i).length).toBeGreaterThan(0);
  });

  it("never claims 'you will become'", () => {
    render(<ScenarioCard scenario={base} />);
    expect(document.body.textContent?.toLowerCase()).not.toContain("you will become");
  });

  it("flags and redacts unsupported percentage probabilities in narrative text", () => {
    render(<ScenarioCard scenario={{ ...base, narrative: "You have an 82% chance of success with this startup." }} />);
    expect(screen.getByTestId("probability-guard")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("82%");
    expect(document.body.textContent).toContain(REDACTION);
  });

  it("redacts probabilities inside trade-off rationales", () => {
    render(<ScenarioCard scenario={{ ...base, tradeoffs: [{ value_key: "stability", direction: "mixed", rationale: "a 1 in 3 chance of layoffs" }] }} />);
    expect(document.body.textContent).not.toMatch(/1 in 3 chance/);
  });

  it("shows no guard warning for clean text", () => {
    render(<ScenarioCard scenario={base} />);
    expect(screen.queryByTestId("probability-guard")).toBeNull();
  });
});

describe("probability guard", () => {
  it.each([
    "82% chance of success",
    "There is a 40 percent likelihood you regret it",
    "1 in 3 chance it fails",
    "the probability of success is 0.7",
    "p=0.82",
  ])("detects %s", (t) => {
    expect(findProbabilityClaims(t).length).toBeGreaterThan(0);
    expect(redactProbabilityClaims(t).text).toContain(REDACTION);
  });

  it.each(["You would work 3 years in this role", "Savings of 4,000 cover 5 months", "Option 2 of 3"])("ignores %s", (t) => {
    expect(findProbabilityClaims(t)).toEqual([]);
  });
});
