import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvidenceLegend, EvidenceTable } from "./Evidence";
import { EVIDENCE_META, FUTURE_SELF_LEGEND } from "../lib/evidence";

describe("EvidenceLegend", () => {
  it("lists all six Future Self evidence types with distinct styles", () => {
    render(<EvidenceLegend />);
    const legend = screen.getByRole("region", { name: /how to read evidence/i });
    for (const k of FUTURE_SELF_LEGEND) {
      const chip = within(legend).getByText(EVIDENCE_META[k].label);
      expect(chip).toHaveClass(`ev--${k}`);
    }
    expect(FUTURE_SELF_LEGEND).toEqual(["user_fact", "deterministic_calculation", "external_study", "assumption", "narrative", "insufficient_evidence"]);
  });

  it("does not offer simulated outcomes as Future Self evidence", () => {
    render(<EvidenceLegend />);
    expect(screen.queryByText(EVIDENCE_META.simulated_outcome.label)).toBeNull();
    expect(EVIDENCE_META.simulated_outcome.allowedInFutureSelf).toBe(false);
  });
});

describe("EvidenceTable", () => {
  it("renders insufficient-evidence records with the insufficient chip", () => {
    render(
      <EvidenceTable
        records={[
          { evidence_id: "e1", domain: "career", source_type: "assumption", claim: "Salary offer stays as stated", validation_status: "insufficient_evidence" },
          { evidence_id: "e2", domain: "career", source_type: "user_fact", claim: "Savings: 4000", validation_status: "unvalidated" },
        ]}
      />,
    );
    expect(screen.getByText("Insufficient evidence")).toHaveClass("ev--insufficient_evidence");
    expect(screen.getByText("Your fact")).toHaveClass("ev--user_fact");
  });

  it("shows an empty message", () => {
    render(<EvidenceTable records={[]} />);
    expect(screen.getByText(/no evidence records/i)).toBeInTheDocument();
  });
});
