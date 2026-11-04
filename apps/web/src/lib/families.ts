/**
 * Static task-family metadata mirrored from hpe.tasks.FAMILY_SPECS (task *definitions*, not
 * results). Used for labels and chance lines; assessment payloads that carry their own
 * `chance` value take precedence.
 */

export interface FamilyMeta {
  label: string;
  nActions: number;
  description: string;
  transfer: boolean;
}

export const FAMILY_META: Record<string, FamilyMeta> = {
  sort: { label: "Sort", nActions: 3, description: "Categorise a novel symbol into one of three hidden classes.", transfer: false },
  compare: { label: "Compare", nActions: 2, description: "Choose the more valuable of two novel symbols.", transfer: false },
  signal: { label: "Signal", nActions: 2, description: "Approach or avoid a novel symbol under a hidden safety rule.", transfer: false },
  barter: {
    label: "Barter (transfer)",
    nActions: 2,
    description: "Held-out family never practised during development: measures transfer.",
    transfer: true,
  },
};

export function familyLabel(f: string): string {
  return FAMILY_META[f]?.label ?? f;
}

export function chanceFor(f: string): number | undefined {
  const m = FAMILY_META[f];
  return m ? 1 / m.nActions : undefined;
}

export const POLICY_LABEL: Record<string, string> = {
  static_persona: "Static persona",
  memory_only: "Memory-only",
  plain_rl: "Plain RL",
  developmental: "Developmental learner",
  random: "Random",
};

export function policyLabel(p: string): string {
  return POLICY_LABEL[p] ?? p;
}

export const ACTIVITY_COLORS: Record<string, string> = {
  rest: "var(--s6)",
  study: "var(--s1)",
  school: "var(--s1)",
  work: "var(--s2)",
  socialize: "var(--s4)",
  tutoring: "var(--s3)",
};

export const SERIES_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"];

export function seriesColor(i: number): string {
  return SERIES_COLORS[i % SERIES_COLORS.length] ?? "var(--s1)";
}

/** Stable colour per policy class across all charts. */
export const POLICY_COLORS: Record<string, string> = {
  static_persona: "var(--s6)",
  memory_only: "var(--s2)",
  plain_rl: "var(--s4)",
  developmental: "var(--s3)",
  random: "var(--ink-3)",
};
