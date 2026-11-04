/**
 * Future Self in the browser — lane `pyodide-futures`.
 *
 * For the paths below, `futuresTransport` returns exactly what the live API returns (same Pydantic
 * models, serialised the same way), by running the repository's own Python engine (`hpe.futures`,
 * through `packages/hpe/futures/bridge.py`) in Pyodide. User-private records (consent, decisions,
 * reflections, exported data) live only in this browser's storage and are deleted by
 * POST /privacy/delete. No network call carries user data anywhere.
 *
 *   POST /scenarios, POST /decisions/scenarios   → bridge.generate(DecisionRequest) → ScenarioResponse
 *   POST /decisions/reflect                      → Reflection (requires prior consent, as the API)
 *   GET  /journey/{user_id}                      → { user_id, decisions, reflections }
 *   GET  /evidence, GET /evidence/{id}           → the static evidence library (from the engine)
 *   POST /privacy/consent | /privacy/export | /privacy/delete
 */

import type { ConsentRecord, DecisionRequest, EvidenceRecord, Reflection, ScenarioResponse } from "../../types/contracts";
import { ApiError, type RequestOptions } from "../client";
import { getBridge, pythonError } from "./pyodide";
import * as store from "./storage";

/** GET /evidence stays with the pre-rendered static JSON (lane static-site); it must not boot the engine. */
const FUTURES_PREFIXES = ["/scenarios", "/decisions/", "/journey/", "/privacy/"] as const;

export function isFuturesPath(_method: string, path: string): boolean {
  return FUTURES_PREFIXES.some((p) => path === p.replace(/\/$/, "") || path.startsWith(p));
}

function bad(status: number, message: string, path: string, detail?: unknown): ApiError {
  return new ApiError(message, status, path, detail ?? { detail: message });
}

function body<T>(opts: RequestOptions, path: string): T {
  if (opts.body === undefined || opts.body === null) throw bad(422, "request body required", path);
  return opts.body as T;
}

async function generate(path: string, opts: RequestOptions): Promise<ScenarioResponse> {
  const req = body<DecisionRequest>(opts, path);
  if (!req.consent_given) {
    throw bad(400, "consent_given must be true: the Future Self demo needs explicit consent before any processing", path);
  }
  const bridge = await getBridge();
  let raw: string;
  try {
    raw = bridge.generate(JSON.stringify(req));
  } catch (e) {
    const { type, message } = pythonError(e);
    if (type === "ConsentRequired") throw bad(400, message, path);
    if (type === "ValidationError") throw bad(422, message, path);
    throw bad(500, `scenario engine error: ${message}`, path, { python: type });
  }
  const resp = JSON.parse(raw) as ScenarioResponse;
  store.addDecision(req, resp);
  return resp;
}

/** The engine's static evidence library (what GET /evidence lists); exposed for tests and tooling. */
export async function evidenceLibrary(): Promise<EvidenceRecord[]> {
  const bridge = await getBridge();
  return JSON.parse(bridge.evidenceCatalog()) as EvidenceRecord[];
}

export async function futuresTransport(method: string, path: string, opts: RequestOptions): Promise<unknown> {
  const url = new URL(path, "http://static.local");
  const p = url.pathname;

  if (method === "POST" && (p === "/scenarios" || p === "/decisions/scenarios")) return generate(p, opts);

  if (method === "POST" && p === "/decisions/reflect") {
    const r = body<Reflection>(opts, p);
    try {
      return store.addReflection({ ...r, created_at: r.created_at ?? new Date().toISOString() });
    } catch (e) {
      if (e instanceof store.StoreError) throw bad(e.status, e.message, p);
      throw e;
    }
  }

  if (method === "GET" && p.startsWith("/journey/")) {
    return store.journey(decodeURIComponent(p.slice("/journey/".length)));
  }

  if (method === "POST" && p === "/privacy/consent") return store.consent(body<ConsentRecord>(opts, p));
  if (method === "POST" && p === "/privacy/export") return store.exportData(body<{ user_id: string }>(opts, p).user_id);
  if (method === "POST" && p === "/privacy/delete") return store.deleteData(body<{ user_id: string }>(opts, p).user_id);

  throw bad(405, `${method} ${p} is not available on the static deployment`, p, { static_mode: true });
}
