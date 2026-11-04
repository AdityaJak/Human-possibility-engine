import type { EventType, ExperienceEvent } from "../types/contracts";
import { familyLabel } from "./families";

export const SIGNIFICANT_EVENT_TYPES: EventType[] = [
  "birth",
  "death",
  "school_enroll",
  "school_graduate",
  "job_start",
  "job_end",
  "partnership_formed",
  "shock",
  "intervention",
];

export const EVENT_GROUPS: Record<string, { label: string; types: EventType[]; color: string }> = {
  life: { label: "Life course", types: ["birth", "death", "stage_change", "partnership_formed", "inheritance"], color: "var(--s5)" },
  learning: {
    label: "Learning",
    types: ["school_enroll", "school_graduate", "tutoring_session", "learning_episode", "policy_update", "assessment", "mentorship"],
    color: "var(--s1)",
  },
  work: { label: "Work & resources", types: ["job_start", "job_end", "wage"], color: "var(--s2)" },
  social: { label: "Social", types: ["social_interaction", "friendship_formed"], color: "var(--s4)" },
  decisions: { label: "Decisions", types: ["decision"], color: "var(--s3)" },
  world: { label: "World", types: ["shock", "intervention", "checkpoint", "run_start", "run_end"], color: "var(--s6)" },
};

export function groupOf(t: EventType): string {
  for (const [k, g] of Object.entries(EVENT_GROUPS)) if (g.types.includes(t)) return k;
  return "world";
}

const LABEL: Partial<Record<EventType, string>> = {
  birth: "Born",
  death: "End of simulated life",
  stage_change: "Life stage changed",
  school_enroll: "Enrolled in school",
  school_graduate: "Left school",
  tutoring_session: "Tutoring session",
  learning_episode: "Practice episodes",
  policy_update: "Policy parameters updated",
  assessment: "Held-out assessment",
  decision: "Decision",
  job_start: "Started job",
  job_end: "Job ended",
  wage: "Wage",
  social_interaction: "Social interaction",
  friendship_formed: "Friendship formed",
  partnership_formed: "Partnership formed",
  mentorship: "Mentorship",
  shock: "Shock",
  intervention: "Intervention",
  inheritance: "Inheritance",
  checkpoint: "Checkpoint",
  run_start: "Run started",
  run_end: "Run ended",
};

function num(v: unknown, d = 2): string | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v ? v : undefined;
}

/** Human-readable one-line description of an event, using only fields present in its payload. */
export function describeEvent(e: ExperienceEvent): string {
  const p = (e.payload ?? {}) as Record<string, unknown>;
  const initialPopulation = e.type === "birth" && e.provenance?.mechanism === "world.init.population" && !e.other_ids?.length;
  const base = initialPopulation ? "Entered simulation (initial population)" : (LABEL[e.type] ?? e.type);
  const parts: string[] = [];
  const fam = str(p.family);
  switch (e.type) {
    case "assessment": {
      if (fam) parts.push(familyLabel(fam));
      const acc = num(p.accuracy);
      if (acc) parts.push(`accuracy ${acc}${p.ci_low !== undefined ? ` [${num(p.ci_low)}, ${num(p.ci_high)}]` : ""}`);
      if (typeof p.n_items === "number") parts.push(`${p.n_items} held-out items`);
      break;
    }
    case "learning_episode":
    case "tutoring_session": {
      if (fam) parts.push(familyLabel(fam));
      if (typeof p.n === "number") parts.push(`${p.n} episodes`);
      const mr = num(p.mean_reward);
      if (mr) parts.push(`mean reward ${mr}`);
      const src = str(p.source) ?? str(p.mode);
      if (src) parts.push(src);
      break;
    }
    case "job_start":
    case "job_end": {
      const job = str(p.job_id) ?? str(p.job) ?? str(p.title);
      if (job) parts.push(job);
      if (fam) parts.push(`uses ${familyLabel(fam)}`);
      break;
    }
    case "wage": {
      const w = num(p.amount ?? p.wage);
      if (w) parts.push(w);
      break;
    }
    case "shock":
    case "intervention": {
      const k = str(p.kind) ?? str(p.name);
      if (k) parts.push(k);
      break;
    }
    case "decision": {
      const a = str(p.activity) ?? str(p.action);
      if (a) parts.push(a);
      break;
    }
    case "stage_change": {
      const s = str(p.stage) ?? str(p.to);
      if (s) parts.push(`→ ${s}`);
      break;
    }
    default: {
      const k = str(p.kind) ?? str(p.reason);
      if (k) parts.push(k);
    }
  }
  if (e.other_ids && e.other_ids.length) parts.push(`with ${e.other_ids.slice(0, 3).join(", ")}${e.other_ids.length > 3 ? "…" : ""}`);
  return parts.length ? `${base}: ${parts.join(" · ")}` : base;
}
