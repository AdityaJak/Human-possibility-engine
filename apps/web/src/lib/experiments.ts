import type { ExperimentDefinition } from "../types/contracts";
import { POLICY_COLORS, seriesColor } from "./families";

export function policyClasses(def: ExperimentDefinition): string[] {
  return [...new Set(def.arms.map((a) => a.policy_class))];
}

/** An intervention (twin-society) experiment: some arm applies a world intervention.
 *  Several policy classes may be present — e.g. a non-learning class used as a mechanism check.
 *  Learning-lab arms that differ only by world_overrides (feedback conditions) are not interventions. */
export function isIntervention(def: ExperimentDefinition): boolean {
  return def.arms.some((a) => (a.interventions?.length ?? 0) > 0);
}

/** A baseline comparison: several policy classes under identical conditions (no interventions). */
export function isBaselineComparison(def: ExperimentDefinition): boolean {
  return policyClasses(def).length >= 2 && !isIntervention(def);
}

export function armColor(def: ExperimentDefinition | undefined, arm: string, i: number): string {
  if (def && isBaselineComparison(def)) {
    const a = def.arms.find((x) => x.name === arm);
    if (a && POLICY_COLORS[a.policy_class]) return POLICY_COLORS[a.policy_class] as string;
  }
  if (def && arm === def.control_arm) return "var(--ink-3)";
  return seriesColor(i);
}
