/**
 * Typed client for the HPE HTTP API (docs/INTERFACES.md §4).
 *
 * All response types come from the generated contracts in src/types/contracts.ts.
 * Shapes NOT covered by the contracts are declared here explicitly and documented as
 * assumptions (JourneyResponse, ResearchMetrics).
 */

import type {
  AgentDetail,
  AgentPage,
  BenchmarkRecord,
  ClaimRecord,
  ConsentRecord,
  DecisionRequest,
  EventPage,
  EvidenceRecord,
  ExperimentCreate,
  ExperimentInfo,
  FramesResponse,
  Health,
  PrivacyDeleteResult,
  PrivacyExport,
  Reflection,
  RunControl,
  RunCreate,
  RunInfo,
  ScenarioResponse,
  StepRequest,
  TimeSeriesResponse,
  WorldTick,
} from "../types/contracts";
import { apiBaseUrl, isMockMode, isStaticMode } from "./config";
import { staticDownloadUrl } from "./static/get";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  readonly url: string;

  constructor(message: string, status: number, url: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
    this.detail = detail;
  }

  /** True when the server could not be reached at all (network / CORS / server down). */
  get isNetwork(): boolean {
    return this.status === 0;
  }
}

export type QueryValue = string | number | boolean | null | undefined | ReadonlyArray<string | number>;
export type Query = Record<string, QueryValue>;

export interface RequestOptions {
  query?: Query;
  body?: unknown;
  signal?: AbortSignal;
}

/** Encodes arrays as repeated keys (FastAPI list query convention): ?types=a&types=b */
export function buildQuery(query?: Query): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const key of Object.keys(query).sort()) {
    const value = query[key];
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const v of value) params.append(key, String(v));
    } else {
      params.append(key, String(value));
    }
  }
  const s = params.toString();
  return s ? `?${s}` : "";
}

function describeDetail(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg, type}]
    const msgs = detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : undefined))
      .filter(Boolean);
    if (msgs.length) return msgs.join("; ");
  }
  return undefined;
}

export type Transport = (method: string, path: string, opts: RequestOptions) => Promise<unknown>;

const fetchTransport: Transport = async (method, path, opts) => {
  const url = `${apiBaseUrl()}${path}${buildQuery(opts.query)}`;
  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: opts.body !== undefined ? { "Content-Type": "application/json", Accept: "application/json" } : { Accept: "application/json" },
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: opts.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`Cannot reach the API at ${apiBaseUrl()}. Is the backend running?`, 0, url, String(err));
  }
  const text = await res.text();
  let parsed: unknown = undefined;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (!res.ok) {
    const detail = parsed && typeof parsed === "object" && "detail" in parsed ? (parsed as { detail: unknown }).detail : parsed;
    const msg = describeDetail(detail) ?? `${res.status} ${res.statusText || "error"}`;
    throw new ApiError(msg, res.status, url, detail);
  }
  return parsed;
};

let transportOverride: Transport | null = null;

/** Test hook: replace the transport (e.g. with a stub). Pass null to restore. */
export function setTransport(t: Transport | null): void {
  transportOverride = t;
}

async function resolveTransport(): Promise<Transport> {
  if (transportOverride) return transportOverride;
  if (isMockMode()) {
    const mod = await import("./mocks");
    return mod.mockTransport;
  }
  if (isStaticMode()) {
    const mod = await import("./static/transport");
    return mod.staticTransport;
  }
  return fetchTransport;
}

export async function request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
  const t = await resolveTransport();
  return (await t(method, path, opts)) as T;
}

/** Assumed shape of GET /journey/{user_id} (not in the frozen contracts). */
export interface JourneyResponse {
  decisions: Array<Record<string, unknown>>;
  reflections: Reflection[];
}

/** GET /research/phase3 — Phase III results exactly as committed (hpe.population.core.dashboard). */
export interface Phase3Model {
  report_nll: number;
  ci: [number, number];
  n: number;
  top1: number;
  ece: number;
  quarters: number[];
  select_nll: number | null;
  params: number | null;
  role: string;
  lapse: number;
  grid_edges: string[];
  fit_cpu_s: number;
  causality_violations: string[];
  excluded?: boolean;
}

export interface Phase3Contrast {
  first: string;
  second: string;
  kind: string;
  estimate: number;
  ci_low: number;
  ci_high: number;
  n: number;
}

export interface Phase3Family {
  role: string;
  best_comparator: string | null;
  models: Record<string, Phase3Model>;
  contrasts: Phase3Contrast[];
}

export interface Phase3Experiment {
  experiment_id: string;
  lane: string;
  title: string;
  hypothesis: string;
  report_partition: string;
  path: string;
  preregistration_hash: string | null;
  families: Record<string, Phase3Family>;
}

export interface Phase3Response {
  disclaimer: string;
  experiments: Phase3Experiment[];
  registry: {
    exploratory: Array<Record<string, string>>;
    confirmatory: Array<Record<string, string>>;
    abandoned: Array<Record<string, string>>;
    amendments: Array<Record<string, string>>;
  };
  verdict: {
    decision: string;
    summary: string;
    gates: Record<string, { status: string; note?: string }>;
  } | null;
}

/** Assumed shape of GET /research/metrics: {experiments, runs, benchmarks} summary. */
export type ResearchMetrics = Record<string, unknown>;

export interface AgentQuery {
  offset?: number;
  limit?: number;
  q?: string;
  alive?: boolean;
}

export interface EventParams {
  agent_id?: string;
  types?: string[];
  tick_from?: number;
  tick_to?: number;
  offset?: number;
  limit?: number;
}

export interface FrameParams {
  tick_from?: number;
  tick_to?: number;
  every?: number;
}

const enc = encodeURIComponent;

export const api = {
  health: (signal?: AbortSignal) => request<Health>("GET", "/health", { signal }),

  // ---- runs ---------------------------------------------------------------------
  listRuns: (signal?: AbortSignal) => request<RunInfo[]>("GET", "/runs", { signal }),
  createRun: (body: RunCreate) => request<RunInfo>("POST", "/runs", { body }),
  getRun: (id: string, signal?: AbortSignal) => request<RunInfo>("GET", `/runs/${enc(id)}`, { signal }),
  stepRun: (id: string, body: StepRequest) => request<WorldTick>("POST", `/runs/${enc(id)}/step`, { body }),
  controlRun: (id: string, body: RunControl) => request<RunInfo>("POST", `/runs/${enc(id)}/control`, { body }),
  listAgents: (id: string, q: AgentQuery = {}, signal?: AbortSignal) =>
    request<AgentPage>("GET", `/runs/${enc(id)}/agents`, { query: { ...q }, signal }),
  getAgent: (id: string, agentId: string, signal?: AbortSignal) =>
    request<AgentDetail>("GET", `/runs/${enc(id)}/agents/${enc(agentId)}`, { signal }),
  listEvents: (id: string, p: EventParams = {}, signal?: AbortSignal) =>
    request<EventPage>("GET", `/runs/${enc(id)}/events`, { query: { ...p }, signal }),
  getFrames: (id: string, p: FrameParams = {}, signal?: AbortSignal) =>
    request<FramesResponse>("GET", `/runs/${enc(id)}/frames`, { query: { ...p }, signal }),
  getTimeseries: (id: string, signal?: AbortSignal) =>
    request<TimeSeriesResponse>("GET", `/runs/${enc(id)}/timeseries`, { signal }),
  downloadUrl: (id: string) => (isStaticMode() ? staticDownloadUrl(id) : `${apiBaseUrl()}/runs/${enc(id)}/download`),

  // ---- experiments --------------------------------------------------------------
  listExperiments: (signal?: AbortSignal) => request<ExperimentInfo[]>("GET", "/experiments", { signal }),
  createExperiment: (body: ExperimentCreate) => request<ExperimentInfo>("POST", "/experiments", { body }),
  runExperiment: (id: string) => request<ExperimentInfo>("POST", `/experiments/${enc(id)}/run`),
  getExperimentResults: (id: string, signal?: AbortSignal) =>
    request<ExperimentInfo>("GET", `/experiments/${enc(id)}/results`, { signal }),

  // ---- research -----------------------------------------------------------------
  researchMetrics: (signal?: AbortSignal) => request<ResearchMetrics>("GET", "/research/metrics", { signal }),
  claims: (signal?: AbortSignal) => request<ClaimRecord[]>("GET", "/research/claims", { signal }),
  benchmarks: (signal?: AbortSignal) => request<BenchmarkRecord[]>("GET", "/benchmarks", { signal }),
  phase3: (signal?: AbortSignal) => request<Phase3Response>("GET", "/research/phase3", { signal }),

  // ---- future self --------------------------------------------------------------
  scenarios: (body: DecisionRequest) => request<ScenarioResponse>("POST", "/scenarios", { body }),
  reflect: (body: Reflection) => request<Reflection>("POST", "/decisions/reflect", { body }),
  evidence: (signal?: AbortSignal) => request<EvidenceRecord[]>("GET", "/evidence", { signal }),
  evidenceById: (id: string, signal?: AbortSignal) => request<EvidenceRecord>("GET", `/evidence/${enc(id)}`, { signal }),
  journey: (userId: string, signal?: AbortSignal) => request<JourneyResponse>("GET", `/journey/${enc(userId)}`, { signal }),

  // ---- privacy ------------------------------------------------------------------
  consent: (body: ConsentRecord) => request<ConsentRecord>("POST", "/privacy/consent", { body }),
  exportData: (userId: string) => request<PrivacyExport>("POST", "/privacy/export", { body: { user_id: userId } }),
  deleteData: (userId: string) => request<PrivacyDeleteResult>("POST", "/privacy/delete", { body: { user_id: userId } }),
};

export type Api = typeof api;
