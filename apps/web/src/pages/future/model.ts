import type { DecisionOption, ScenarioResponse, ValueWeight } from "../../types/contracts";

export const DEFAULT_VALUES: ValueWeight[] = [
  { key: "autonomy", label: "Autonomy", weight: 0.5 },
  { key: "income", label: "Income", weight: 0.5 },
  { key: "relationships", label: "Relationships", weight: 0.5 },
  { key: "stability", label: "Stability", weight: 0.5 },
  { key: "learning", label: "Learning & growth", weight: 0.5 },
  { key: "health", label: "Health", weight: 0.5 },
  { key: "time", label: "Free time", weight: 0.5 },
];

export interface Consent {
  /** required: understands scenarios are illustrative, and agrees to demo storage */
  demo: boolean;
  reflections: boolean;
  personalization: boolean;
  recordedAt?: string;
}

export interface DecisionDraft {
  question: string;
  options: DecisionOption[];
  constraints: Record<string, number | string>;
  horizonYears: number;
}

export interface PlannedStep {
  key: string;
  scenarioId: string;
  optionLabel: string;
  title: string;
  description: string;
  durationDays: number;
  checkIn: string;
  done: boolean;
}

export interface FutureState {
  userId: string;
  consent: Consent;
  values: ValueWeight[];
  draft: DecisionDraft;
  response: ScenarioResponse | null;
  plan: PlannedStep[];
}

export function emptyDraft(): DecisionDraft {
  return {
    question: "",
    options: [
      { option_id: "opt-1", label: "", kind: "current_path", description: "", inputs: {} },
      { option_id: "opt-2", label: "", kind: "new_job", description: "", inputs: {} },
    ],
    constraints: {},
    horizonYears: 3,
  };
}

export const STEPS = [
  { id: "welcome", label: "Welcome", needsConsent: false, needsScenarios: false },
  { id: "consent", label: "Consent", needsConsent: false, needsScenarios: false },
  { id: "north-star", label: "Your North Star", needsConsent: true, needsScenarios: false },
  { id: "decision", label: "Decision Studio", needsConsent: true, needsScenarios: false },
  { id: "futures", label: "Possible futures", needsConsent: true, needsScenarios: true },
  { id: "evidence", label: "Evidence & uncertainty", needsConsent: true, needsScenarios: true },
  { id: "plan", label: "Experiment planner", needsConsent: true, needsScenarios: true },
  { id: "journey", label: "Your journey", needsConsent: true, needsScenarios: false },
  { id: "conversation", label: "Future Self conversation", needsConsent: true, needsScenarios: true },
  { id: "privacy", label: "Privacy & data", needsConsent: false, needsScenarios: false },
] as const;

export type StepId = (typeof STEPS)[number]["id"];

export function stepIndex(id: string): number {
  return STEPS.findIndex((s) => s.id === id);
}
