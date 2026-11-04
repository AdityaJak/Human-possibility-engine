/**
 * Future Self in the browser — implemented by lane `pyodide-futures`.
 *
 * Contract: for the paths below, `futuresTransport` returns exactly what the live API returns
 * (same Pydantic models, serialised the same way), by running the repository's own Python engine
 * (`hpe.futures`, through `packages/hpe/futures/bridge.py`) in Pyodide. User-private records
 * (consent, decisions, reflections, exported data) live only in this browser's storage and are
 * deleted by POST /privacy/delete. No network call carries user data anywhere.
 *
 *   POST /scenarios, POST /decisions/scenarios   → bridge.generate(DecisionRequest) → ScenarioResponse
 *   POST /decisions/reflect                      → Reflection (requires prior consent, as the API)
 *   GET  /journey/{user_id}                      → { user_id, decisions, reflections }
 *   POST /privacy/consent | /privacy/export | /privacy/delete
 */

import type { RequestOptions } from "../client";

const FUTURES_PREFIXES = ["/scenarios", "/decisions/", "/journey/", "/privacy/"] as const;

export function isFuturesPath(_method: string, path: string): boolean {
  return FUTURES_PREFIXES.some((p) => path === p.replace(/\/$/, "") || path.startsWith(p));
}

export async function futuresTransport(method: string, path: string, _opts: RequestOptions): Promise<unknown> {
  throw new Error(`futuresTransport not implemented yet (lane pyodide-futures): ${method} ${path}`);
}
