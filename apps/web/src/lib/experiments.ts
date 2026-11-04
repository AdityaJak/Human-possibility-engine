import type { ExperimentDefinition } from "../types/contracts";
import { POLICY_COLORS, seriesColor } from "./families";

export function policyClasses(def: ExperimentDefinition): string[] {
  return [...new Set(def.arms.map((a) => a.policy_class))];
}

export function isBaselineComparison(def: ExperimentDefinition): boolean {
  return policyClasses(def).length >= 2;
}

export function isIntervention(def: ExperimentDefinition): boolean {
  return policyClasses(def).length === 1 && def.arms.some((a) => (a.interventions?.length ?? 0) > 0 || Object.keys(a.world_overrides ?? {}).length > 0);
}

export function armColor(def: ExperimentDefinition | undefined, arm: string, i: number): string {
  if (def && isBaselineComparison(def)) {
    const a = def.arms.find((x) => x.name === arm);
    if (a && POLICY_COLORS[a.policy_class]) return POLICY_COLORS[a.policy_class] as string;
  }
  if (def && arm === def.control_arm) return "var(--ink-3)";
  return seriesColor(i);
}
