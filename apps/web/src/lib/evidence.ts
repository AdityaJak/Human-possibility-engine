import type { EvidenceSourceType, ValidationStatus } from "../types/contracts";

export type EvidenceKind = EvidenceSourceType | "insufficient_evidence";

export interface EvidenceMeta {
  label: string;
  description: string;
  /** Whether this kind of evidence may appear in the Future Self experience. */
  allowedInFutureSelf: boolean;
}

export const EVIDENCE_META: Record<EvidenceKind, EvidenceMeta> = {
  user_fact: {
    label: "Your fact",
    description: "Something you told us. We do not verify it.",
    allowedInFutureSelf: true,
  },
  deterministic_calculation: {
    label: "Calculation",
    description: "Arithmetic on your inputs with a shown formula. Correct only if the inputs are.",
    allowedInFutureSelf: true,
  },
  external_study: {
    label: "External study",
    description: "Published research about groups of people. Not validated for your individual situation.",
    allowedInFutureSelf: true,
  },
  assumption: {
    label: "Assumption",
    description: "A stated premise the scenario is conditioned on. Change it and the scenario changes.",
    allowedInFutureSelf: true,
  },
  narrative: {
    label: "Narrative",
    description: "Imaginative, illustrative text. Not evidence of anything.",
    allowedInFutureSelf: true,
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    description: "We explicitly cannot predict this. No number is shown.",
    allowedInFutureSelf: true,
  },
  simulated_outcome: {
    label: "Simulated outcome",
    description: "Output of the synthetic agent society. Research only — never used as evidence about your life.",
    allowedInFutureSelf: false,
  },
};

export const FUTURE_SELF_LEGEND: EvidenceKind[] = [
  "user_fact",
  "deterministic_calculation",
  "external_study",
  "assumption",
  "narrative",
  "insufficient_evidence",
];

/** An evidence record whose validation status is insufficient_evidence is displayed as such. */
export function evidenceKindOf(source: EvidenceSourceType, status?: ValidationStatus): EvidenceKind {
  if (status === "insufficient_evidence") return "insufficient_evidence";
  return source;
}

export const VALIDATION_LABEL: Record<ValidationStatus, string> = {
  validated: "Validated in a defined domain",
  unvalidated: "Not validated for this use",
  illustrative: "Illustrative only",
  insufficient_evidence: "Insufficient evidence to predict",
};
