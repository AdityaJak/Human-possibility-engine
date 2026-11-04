/**
 * Static GET resolution — implemented by lane `static-site`.
 *
 * Contract: `staticGet(path, opts)` returns the same JSON the live API would return for
 * `GET ${path}` on the stored real outputs. Files live under `staticApiBase()` and are produced
 * by `scripts/export_static_api.py` (which runs the real FastAPI app in-process). Query strings
 * are not part of the file name; the exporter pre-renders the fullest useful variant of each
 * parameterised endpoint (see the exporter for the exact mapping).
 */

import type { RequestOptions } from "../client";

export async function staticGet(path: string, _opts: RequestOptions): Promise<unknown> {
  throw new Error(`staticGet not implemented yet (lane static-site): GET ${path}`);
}
