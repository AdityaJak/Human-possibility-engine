/**
 * Static GET resolution (lane static-site).
 *
 * `staticGet(path, opts)` returns what the live API returns for `GET ${path}` on the stored real
 * outputs. Files live under `staticApiBase()` and are written by `scripts/export_static_api.py`,
 * which runs the real FastAPI app in-process. Parameterised endpoints are stored once in their
 * fullest form and the query is applied here, with the same semantics as the API handlers
 * (apps/api/hpe_api/routers/runs.py):
 *
 *   /runs/{id}/agents      q (substring of agent or household id), alive, stage, offset, limit (200)
 *   /runs/{id}/events      agent_id, types, tick_from, tick_to, offset, limit (200)
 *   /runs/{id}/frames      tick_from, tick_to, every (1), limit (500)
 *   /runs/{id}/timeseries  every (1)
 */

import { ApiError, type QueryValue, type RequestOptions } from "../client";
import { staticApiBase } from "../config";
import type { AgentPage, AgentSummary, EventPage, ExperienceEvent, FramesResponse, TimeSeriesResponse } from "../../types/contracts";

type Query = Record<string, QueryValue>;

function num(q: Query, key: string, fallback: number): number {
  const v = q[key];
  if (v === undefined || v === null || v === "") return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function optNum(q: Query, key: string): number | undefined {
  const v = q[key];
  if (v === undefined || v === null || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function str(q: Query, key: string): string | undefined {
  const v = q[key];
  if (v === undefined || v === null || v === "") return undefined;
  return Array.isArray(v) ? v.map(String).join(",") : String(v);
}

function list(q: Query, key: string): string[] | undefined {
  const v = q[key];
  if (v === undefined || v === null || v === "") return undefined;
  const parts = (Array.isArray(v) ? v.map(String) : String(v).split(",")).map((s) => s.trim()).filter(Boolean);
  return parts.length ? parts : undefined;
}

async function loadJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const url = `${staticApiBase()}${path}.json`;
  let res: Response;
  try {
    res = await fetch(url, { signal, headers: { Accept: "application/json" } });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`Cannot load the stored data file ${url}.`, 0, url, String(err));
  }
  if (!res.ok) {
    const msg = res.status === 404 ? `No stored data for GET ${path} on this static deployment.` : `${res.status} ${res.statusText || "error"}`;
    throw new ApiError(msg, res.status, url);
  }
  return res.json();
}

const AGENTS_RE = /^\/runs\/([^/]+)\/agents$/;
const EVENTS_RE = /^\/runs\/([^/]+)\/events$/;
const FRAMES_RE = /^\/runs\/([^/]+)\/frames$/;
const SERIES_RE = /^\/runs\/([^/]+)\/timeseries$/;

export function filterAgents(page: AgentPage, q: Query): AgentPage {
  let agents: AgentSummary[] = page.items;
  const needle = str(q, "q")?.toLowerCase();
  if (needle) agents = agents.filter((a) => a.agent_id.toLowerCase().includes(needle) || a.household_id.toLowerCase().includes(needle));
  const alive = q.alive;
  if (alive === true || alive === false || alive === "true" || alive === "false") {
    const want = alive === true || alive === "true";
    agents = agents.filter((a) => a.alive === want);
  }
  const stage = str(q, "stage");
  if (stage) agents = agents.filter((a) => String(a.stage) === stage);
  agents = [...agents].sort((a, b) => (a.agent_id < b.agent_id ? -1 : a.agent_id > b.agent_id ? 1 : 0));
  const offset = Math.max(0, num(q, "offset", 0));
  const limit = Math.max(1, num(q, "limit", 200));
  return { items: agents.slice(offset, offset + limit), total: agents.length, offset, limit };
}

export function filterEvents(page: EventPage, q: Query): EventPage {
  let items: ExperienceEvent[] = page.items;
  const agentId = str(q, "agent_id");
  if (agentId) items = items.filter((e) => e.agent_id === agentId);
  const types = list(q, "types");
  if (types) {
    const set = new Set(types);
    items = items.filter((e) => set.has(e.type));
  }
  const from = optNum(q, "tick_from");
  const to = optNum(q, "tick_to");
  if (from !== undefined) items = items.filter((e) => e.tick >= from);
  if (to !== undefined) items = items.filter((e) => e.tick <= to);
  const offset = Math.max(0, num(q, "offset", 0));
  const limit = Math.max(1, num(q, "limit", 200));
  return { items: items.slice(offset, offset + limit), total: items.length, offset, limit };
}

/** Native tick stride of a pre-rendered frames file (1 when every tick is present). */
export function nativeStride(ticks: readonly number[]): number {
  if (ticks.length < 2) return 1;
  return Math.max(1, (ticks[1] as number) - (ticks[0] as number));
}

export function filterFrames(resp: FramesResponse, q: Query): FramesResponse {
  const from = optNum(q, "tick_from");
  const to = optNum(q, "tick_to");
  const every = Math.max(1, Math.floor(num(q, "every", 1)));
  const limit = Math.max(1, num(q, "limit", 500));
  // The file may hold every k-th tick; `every` is applied relative to that stride so a request for
  // every 6th tick on a stride-3 file yields ticks 0, 6, 12, … exactly as the API would.
  const step = Math.max(1, Math.round(every / nativeStride(resp.ticks)));
  const idx: number[] = [];
  resp.ticks.forEach((t, i) => {
    if ((from === undefined || t >= from) && (to === undefined || t <= to)) idx.push(i);
  });
  const chosen = idx.filter((_, k) => k % step === 0).slice(0, limit);
  return { ...resp, ticks: chosen.map((i) => resp.ticks[i] as number), frames: chosen.map((i) => resp.frames[i] ?? []) };
}

export function filterTimeseries(resp: TimeSeriesResponse, q: Query): TimeSeriesResponse {
  const every = Math.max(1, Math.floor(num(q, "every", 1)));
  if (every === 1) return resp;
  return { ...resp, ticks: resp.ticks.filter((_, i) => i % every === 0) };
}

export async function staticGet(path: string, opts: RequestOptions): Promise<unknown> {
  const q: Query = opts.query ?? {};
  const data = await loadJson(path, opts.signal);
  if (AGENTS_RE.test(path)) return filterAgents(data as AgentPage, q);
  if (EVENTS_RE.test(path)) return filterEvents(data as EventPage, q);
  if (FRAMES_RE.test(path)) return filterFrames(data as FramesResponse, q);
  if (SERIES_RE.test(path)) return filterTimeseries(data as TimeSeriesResponse, q);
  return data;
}

/** URL of the pre-rendered raw replay log for a run (gzip JSONL, same bytes the API streams). */
export function staticDownloadUrl(runId: string): string {
  return `${staticApiBase()}/runs/${encodeURIComponent(runId)}/download.jsonl.gz`;
}
