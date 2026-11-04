import type { ClaimRecord } from "../types/contracts";

export const EVIDENCE_KIND_LABEL: Record<ClaimRecord["evidence_kind"], string> = {
  synthetic_simulation: "Synthetic simulation",
  human_dataset: "Human dataset",
  engineering: "Engineering",
  none: "No evidence",
};
