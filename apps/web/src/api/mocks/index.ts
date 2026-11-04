/**
 * DEVELOPMENT-ONLY mock transport, loaded only when VITE_USE_MOCKS=1.
 *
 * Everything here is procedurally generated placeholder data so UI states can be developed
 * without the backend. It is NOT a research result, every label contains "MOCK", and the
 * app shows a persistent "MOCK DATA — not real results" banner whenever this is active.
 */

import type {
  AgentDetail,
  AgentFrame,
  AgentPage,
  BenchmarkRecord,
  ClaimRecord,
  DecisionRequest,
  EventPage,
  EvidenceRecord,
  ExperienceEvent,
  ExperimentDefinition,
  ExperimentInfo,
  FramesResponse,
  Health,
  OutcomeObservation,
  RunInfo,
  Scenario,
  ScenarioResponse,
  WorldMap,
  WorldTick,
} from "../../types/contracts";
import { ApiError } from "../client";
import type { RequestOptions, Transport } from "../client";

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MAP: WorldMap = {
  width: 100,
  height: 60,
  places: [
    ...Array.from({ length: 25 }, (_, i) => ({ place_id: `home-${i}`, kind: "home" as const, name: `MOCK home ${i}`, x: 8 + (i % 5) * 9, y: 8 + Math.floor(i / 5) * 10 })),
    { place_id: "school", kind: "school", name: "MOCK School", x: 62, y: 12 },
    { place_id: "work-a", kind: "workplace", name: "MOCK Sorting House", x: 82, y: 14 },
    { place_id: "work-b", kind: "workplace", name: "MOCK Market", x: 84, y: 40 },
    { place_id: "plaza", kind: "gathering", name: "MOCK Plaza", x: 64, y: 34 },
    { place_id: "tutor", kind: "tutoring_center", name: "MOCK Tutoring", x: 60, y: 52 },
    { place_id: "elder", kind: "elder_center", name: "MOCK Elder centre", x: 40, y: 54 },
  ],
};

const ACTS = ["rest", "study", "work", "socialize"];
const PLACE_FOR: Record<string, string> = { rest: "home", study: "school", work: "work-a", socialize: "plaza" };

let liveTick = 0;

function frameAt(tick: number, n = 60): AgentFrame[] {
  const r = rng(tick * 7919 + 13);
  return Array.from({ length: n }, (_, i) => {
    const act = ACTS[Math.floor(r() * ACTS.length)] ?? "rest";
    const home = MAP.places[i % 25];
    const placeId = act === "rest" ? home?.place_id ?? "home-0" : PLACE_FOR[act] ?? "plaza";
    const p = MAP.places.find((q) => q.place_id === placeId) ?? MAP.places[0];
    return {
      id: `mock-a${String(i).padStart(3, "0")}`,
      x: (p?.x ?? 0) + (r() - 0.5) * 5,
      y: (p?.y ?? 0) + (r() - 0.5) * 5,
      place_id: placeId,
      activity: act,
      age: (i * 1.3 + tick / 12) % 85,
      alive: true,
      household_id: `MOCK-h${i % 25}`,
    };
  });
}

function tickAt(tick: number, runId: string): WorldTick {
  const r = rng(tick + 1);
  return {
    run_id: runId,
    tick,
    day: tick * 30,
    year: (tick * 30) / 365,
    population: 60,
    births: r() < 0.1 ? 1 : 0,
    deaths: r() < 0.08 ? 1 : 0,
    mean_measured_accuracy: { sort: 0.3 + r() * 0.1, compare: 0.5 + r() * 0.1, signal: 0.5 + r() * 0.1, barter: 0.5 + r() * 0.05 },
    employment_rate: 0.5 + r() * 0.2,
    school_enrolment_rate: 0.8,
    resource_gini: 0.2 + r() * 0.1,
    mean_resources: 5 + r(),
    events_emitted: Math.floor(r() * 100),
    age_histogram: { "0-9": 10, "10-19": 12, "20-39": 18, "40-64": 14, "65+": 6 },
    wall_time_ms: 4 + r() * 2,
  };
}

function manifest(runId: string, mode: string) {
  return {
    run_id: runId,
    config: { seed: 0, n_agents: 60, years: 10, policy_class: "developmental" as const, label: `MOCK ${mode}` },
    config_hash: "MOCKMOCKMOCK",
    policy_class: "developmental" as const,
    policy_version: "mock",
  };
}

const RUNS: RunInfo[] = [
  { run_id: "MOCK-stored-run", status: "completed", mode: "stored", source: "stored_real_output", manifest: manifest("MOCK-stored-run", "stored") },
  { run_id: "MOCK-live-run", status: "paused", mode: "interactive", source: "live", manifest: manifest("MOCK-live-run", "interactive") },
];

function event(seq: number, tick: number, type: ExperienceEvent["type"], agent: string, payload: Record<string, unknown> = {}): ExperienceEvent {
  return { seq, run_id: "MOCK", tick, day: tick * 30, type, agent_id: agent, payload, provenance: { source: "world", mechanism: "MOCK.fixture" } };
}

function agentDetail(runId: string, id: string): AgentDetail {
  const r = rng(id.length * 31);
  const skill = ["sort", "compare", "signal", "barter"].flatMap((f, k) =>
    Array.from({ length: 12 }, (_, y) => {
      const acc = (k === 0 ? 0.33 : 0.5) + (f === "barter" ? 0.02 : 0.03) * y * r() + (r() - 0.5) * 0.04;
      return { tick: y * 12, age_years: y + 2, family: f, accuracy: acc, ci_low: acc - 0.1, ci_high: acc + 0.1, n_items: 60 };
    }),
  );
  return {
    state: {
      agent_id: id,
      run_id: runId,
      household_id: "MOCK-h1",
      birth_tick: 0,
      birth_cohort_year: 0,
      age_years: 13.2,
      stage: "adolescent",
      persona: "MOCK biography text — placeholder.",
      traits: { curiosity: 0.6, diligence: 0.4 },
      policy: { policy_class: "developmental", version: "mock" },
      capabilities: {
        sort: { family: "sort", measured_accuracy: 0.4, measured_ci_low: 0.3, measured_ci_high: 0.5, self_estimate_mean: 0.5, self_estimate_sd: 0.1, practice_episodes: 300 },
      },
      affect: { stress: 0.2, energy: 0.8, social_connection: 0.5, motivation: 0.6 },
    },
    timeline: [event(1, 0, "birth", id), event(2, 72, "school_enroll", id), event(3, 84, "assessment", id, { family: "sort", accuracy: 0.4, n_items: 60 })],
    skill_curve: skill,
    parameter_history: Array.from({ length: 12 }, (_, y) => ({ tick: y * 12, age_years: y + 2, summary: { "MOCK_param_norm": 0.1 * y * r(), n_updates: y * 100 } })),
    relationships: [{ src: id, dst: "mock-a002", kind: "friend", trust: 0.7, interactions: 12 }],
    recent_memories: [],
    latest_decision_explanation: { note: "MOCK explanation" },
  };
}

const MOCK_BASE = { seed: 0, n_agents: 20, years: 10 };

const EXP_BASELINE: ExperimentDefinition = {
  experiment_id: "MOCK-baselines",
  title: "MOCK baseline comparison (placeholder, not a result)",
  hypothesis: "MOCK hypothesis",
  falsification_criterion: "MOCK criterion",
  arms: [
    { name: "static", policy_class: "static_persona" },
    { name: "memory", policy_class: "memory_only" },
    { name: "plain", policy_class: "plain_rl" },
    { name: "dev", policy_class: "developmental" },
  ],
  control_arm: "static",
  seeds: [0, 1, 2, 3, 4],
  base_config: MOCK_BASE,
  analysis: { primary_metric: "holdout_accuracy_mean", primary_metric_definition: "MOCK", comparison: "MOCK" },
  validity_notes: ["MOCK note"],
};

const EXP_TWIN: ExperimentDefinition = {
  experiment_id: "MOCK-twin",
  title: "MOCK twin-society tutoring (placeholder, not a result)",
  hypothesis: "MOCK hypothesis",
  falsification_criterion: "MOCK criterion",
  arms: [
    { name: "control", policy_class: "developmental" },
    { name: "tutoring", policy_class: "developmental", interventions: [{ name: "MOCK tutoring", kind: "tutoring_access", target: { age_min: 5, age_max: 15, fraction: 1 } }] },
  ],
  control_arm: "control",
  seeds: [0, 1, 2, 3, 4],
  base_config: MOCK_BASE,
  analysis: { primary_metric: "holdout_accuracy_age18", primary_metric_definition: "MOCK", comparison: "MOCK" },
};

function outcomes(def: ExperimentDefinition): OutcomeObservation[] {
  const r = rng(def.experiment_id.length);
  return def.arms.flatMap((a, k) =>
    def.seeds.flatMap((s) =>
      Array.from({ length: 8 }, (_, t) => ({ arm: a.name, seed: s, metric: def.analysis.primary_metric, value: 0.5 + 0.01 * k * t + (r() - 0.5) * 0.05, age_years: t * 2, split: "holdout" as const })),
    ),
  );
}

function expInfo(def: ExperimentDefinition): ExperimentInfo {
  return {
    experiment_id: def.experiment_id,
    status: "completed",
    source: "live",
    definition: def,
    result: {
      experiment_id: def.experiment_id,
      preregistration_hash: "MOCK",
      status: "inconclusive",
      status_reason: "MOCK placeholder — no experiment has been run in mock mode.",
      arms: def.arms.map((a) => ({ arm: a.name, n_runs: def.seeds.length, metrics: { [def.analysis.primary_metric]: { mean: 0.5, sd: 0.02, ci_low: 0.48, ci_high: 0.52, n: 5 } } })),
      effects: def.arms.slice(1).map((a) => ({ metric: def.analysis.primary_metric, contrast: `${a.name} − ${def.control_arm} (MOCK)`, estimate: 0, ci_low: -0.02, ci_high: 0.02, n_units: 5, method: "MOCK" })),
      run_ids: ["MOCK-run-1"],
      limitations: ["MOCK limitation"],
    },
    outcomes_sample: outcomes(def),
  };
}

function scenarioFor(req: DecisionRequest): ScenarioResponse {
  const evidence: EvidenceRecord[] = [
    { evidence_id: "MOCK-ev-assume", domain: "MOCK", source_type: "assumption", claim: "MOCK assumption", validation_status: "illustrative" },
    { evidence_id: "MOCK-ev-calc", domain: "MOCK", source_type: "deterministic_calculation", claim: "MOCK calculation", validation_status: "unvalidated" },
  ];
  const scenarios: Scenario[] = req.options.map((o, i) => ({
    scenario_id: `MOCK-sc-${i}`,
    decision_id: "MOCK-decision",
    option_id: o.option_id,
    title: `MOCK scenario: ${o.label}`,
    label: "illustrative scenario",
    narrative: `MOCK narrative placeholder for “${o.label}”. Not generated by the real scenario engine.`,
    assumptions: [evidence[0] as EvidenceRecord],
    tradeoffs: req.values.slice(0, 3).map((v) => ({ value_key: v.key, direction: "unclear" as const, rationale: "MOCK rationale" })),
    calculations: [],
    uncertainties: ["MOCK uncertainty"],
    cannot_predict: ["MOCK — long-term satisfaction"],
    reversible_experiments: [{ title: "MOCK experiment", description: "MOCK description", duration_days: 14 }],
  }));
  return { decision_id: "MOCK-decision", scenarios, evidence, value_alignment_note: "MOCK value alignment note.", safety: null };
}

const CLAIMS: ClaimRecord[] = [
  { claim_id: "MOCK-1", claim: "MOCK claim — placeholder", status: "not_tested", evidence_kind: "none", limitations: ["MOCK"] },
  { claim_id: "MOCK-2", claim: "MOCK claim — placeholder", status: "inconclusive", evidence_kind: "synthetic_simulation", limitations: [] },
];

const BENCH: BenchmarkRecord[] = [];

function match(path: string, re: RegExp): string[] | null {
  const m = re.exec(path);
  return m ? m.slice(1).map(decodeURIComponent) : null;
}

export const mockTransport: Transport = async (method: string, path: string, opts: RequestOptions) => {
  await new Promise((r) => setTimeout(r, 120));
  let m: string[] | null;
  if (path === "/health") return { status: "ok", version: "MOCK", contract_version: "0.1.0", llm_provider: "none" } satisfies Health;
  if (path === "/runs" && method === "GET") return RUNS;
  if (path === "/runs" && method === "POST") return { ...RUNS[1], run_id: "MOCK-live-run" };
  if ((m = match(path, /^\/runs\/([^/]+)$/))) return RUNS.find((r) => r.run_id === m?.[0]) ?? RUNS[0];
  if ((m = match(path, /^\/runs\/([^/]+)\/step$/))) {
    const n = Number((opts.body as { ticks?: number } | undefined)?.ticks ?? 1);
    liveTick += n;
    return { ...tickAt(liveTick, m[0] ?? ""), frames: frameAt(liveTick) };
  }
  if ((m = match(path, /^\/runs\/([^/]+)\/control$/))) {
    if ((opts.body as { action?: string } | undefined)?.action === "reset") liveTick = 0;
    return RUNS[1];
  }
  if ((m = match(path, /^\/runs\/([^/]+)\/frames$/))) {
    const every = Number(opts.query?.every ?? 6);
    const ticks = Array.from({ length: 40 }, (_, i) => i * every);
    return { run_id: m[0] ?? "", map: MAP, ticks, frames: ticks.map((t) => frameAt(t)) } satisfies FramesResponse;
  }
  if ((m = match(path, /^\/runs\/([^/]+)\/timeseries$/))) {
    const len = m[0] === "MOCK-live-run" ? liveTick + 1 : 120;
    return { run_id: m[0], ticks: Array.from({ length: len }, (_, t) => tickAt(t, m?.[0] ?? "")) };
  }
  if ((m = match(path, /^\/runs\/([^/]+)\/agents$/))) {
    const q = String(opts.query?.q ?? "");
    const items = frameAt(0)
      .filter((f) => !q || f.id.includes(q))
      .map((f) => ({ agent_id: f.id, household_id: f.household_id, age_years: f.age, stage: "adult", alive: true, activity: f.activity, job_id: null, education_years: 8, resources: 3, mean_measured_accuracy: 0.5, n_friends: 2 }));
    return { items: items.slice(0, 40), total: items.length, offset: 0, limit: 40 } satisfies AgentPage;
  }
  if ((m = match(path, /^\/runs\/([^/]+)\/agents\/([^/]+)$/))) return agentDetail(m[0] ?? "", m[1] ?? "");
  if ((m = match(path, /^\/runs\/([^/]+)\/events$/))) {
    const items = [event(1, 3, "birth", "mock-a001"), event(2, 5, "job_start", "mock-a004", { job_id: "MOCK job" })];
    return { items, total: items.length, offset: 0, limit: 50 } satisfies EventPage;
  }
  if (path === "/experiments") return [expInfo(EXP_BASELINE), expInfo(EXP_TWIN)];
  if ((m = match(path, /^\/experiments\/([^/]+)\/results$/))) return expInfo(m[0] === "MOCK-twin" ? EXP_TWIN : EXP_BASELINE);
  if (path === "/research/claims") return CLAIMS;
  if (path === "/benchmarks") return BENCH;
  if (path === "/research/metrics") return { experiments: 0, runs: 0, benchmarks: 0, note: "MOCK" };
  if (path === "/scenarios" || path === "/decisions/scenarios") return scenarioFor(opts.body as DecisionRequest);
  if (path === "/decisions/reflect") return opts.body;
  if ((m = match(path, /^\/journey\/([^/]+)$/))) return { decisions: [{ decision_id: "MOCK-decision", question: "MOCK question" }], reflections: [] };
  if (path === "/privacy/consent") return opts.body;
  if (path === "/privacy/export") return { user_id: (opts.body as { user_id: string }).user_id, exported_at: new Date().toISOString(), records: { MOCK: [] } };
  if (path === "/privacy/delete") return { user_id: (opts.body as { user_id: string }).user_id, deleted: { MOCK_records: 0 } };
  if (path === "/evidence") return [];
  throw new ApiError(`MOCK: no fixture for ${method} ${path}`, 404, path);
};
