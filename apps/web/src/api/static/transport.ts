/**
 * Static-deployment transport (VITE_API_MODE=static). FROZEN SEAM — owned by the orchestrator.
 *
 *  - GET  → `./get.ts`     (lane static-site): JSON pre-rendered from committed stored real outputs.
 *  - Future Self paths → `./futures.ts` (lane pyodide-futures): the repository's Python engine
 *    (hpe.futures) executed in the browser via Pyodide; user-private records in browser storage.
 *  - Anything that would start a simulation/experiment or mutate server state → 405.
 *
 * Nothing here is a mock: every GET payload is produced by the real API code from real artifacts,
 * and scenarios come from the same Python code the API runs.
 */

import { ApiError, type RequestOptions, type Transport } from "../client";
import { futuresTransport, isFuturesPath } from "./futures";
import { staticGet } from "./get";

export const UNAVAILABLE_IN_STATIC_MODE = [
  "POST /runs",
  "POST /runs/{id}/step",
  "POST /runs/{id}/control",
  "POST /experiments",
  "POST /experiments/{id}/run",
] as const;

export const STATIC_MODE_MESSAGE =
  "Not available on this static deployment, which serves stored real outputs read-only. " +
  "Run `make demo` locally or deploy the API (docs/DEPLOYMENT.md) for live simulation.";

export const staticTransport: Transport = async (method: string, path: string, opts: RequestOptions) => {
  if (isFuturesPath(method, path)) return futuresTransport(method, path, opts);
  if (method === "GET") return staticGet(path, opts);
  throw new ApiError(STATIC_MODE_MESSAGE, 405, path, { static_mode: true, method });
};
