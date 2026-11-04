import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ClaimRecord } from "../types/contracts";
import { ClaimsTable } from "./ClaimsTable";

const claims: ClaimRecord[] = [
  { claim_id: "c1", claim: "Claim A", status: "supported", evidence_kind: "synthetic_simulation", run_ids: ["r1", "r2"], metric: "acc", value: "0.1" },
  { claim_id: "c2", claim: "Claim B", status: "not_supported", evidence_kind: "synthetic_simulation", limitations: ["small n"] },
  { claim_id: "c3", claim: "Claim C", status: "blocked", evidence_kind: "human_dataset" },
  { claim_id: "c4", claim: "Claim D", status: "not_tested", evidence_kind: "none" },
  { claim_id: "c5", claim: "Claim E", status: "inconclusive", evidence_kind: "engineering" },
];

describe("ClaimsTable", () => {
  it("renders every claim with its status chip", () => {
    render(<ClaimsTable claims={claims} />);
    const rows = screen.getAllByTestId("claim-row");
    expect(rows).toHaveLength(5);
    for (const label of ["Supported", "Not supported", "Blocked", "Not tested", "Inconclusive"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(screen.getByText("2 runs")).toBeInTheDocument();
    expect(screen.getByText("small n")).toBeInTheDocument();
    expect(screen.getByText("Human dataset")).toBeInTheDocument();
  });

  it("filters by status", () => {
    render(<ClaimsTable claims={claims} />);
    const notSupported = screen.getAllByRole("button").find((b) => b.textContent?.includes("Not supported"));
    expect(notSupported).toBeDefined();
    fireEvent.click(notSupported as HTMLElement);
    expect(screen.getAllByTestId("claim-row")).toHaveLength(1);
    expect(screen.getByText("Claim B")).toBeInTheDocument();
  });
});
