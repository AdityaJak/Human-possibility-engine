/**
 * Lazy Pyodide loader for the in-browser Future Self engine (lane pyodide-futures).
 *
 * Loads nothing until `getBridge()` is first called. Then, once per page load:
 *   1. injects the Pyodide script from the CDN,
 *   2. loads numpy + pydantic from the Pyodide package index,
 *   3. installs this repository's own wheel (`hpe-<version>-py3-none-any.whl`, served next to the
 *      site) with `deps=False` — Pyodide ships numpy 2.0.x while the wheel's metadata pins >=2.1;
 *      the engine only needs numpy for stable hashing, which works on 2.0 —
 *   4. imports `hpe.futures.bridge`.
 * Progress is published to subscribers for the UI. Errors become ApiError(503).
 */

import { ApiError } from "../client";

export const PYODIDE_VERSION = "0.27.7";
export const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
/** Built by `uv build --wheel -o apps/web/public/wheels` (distribution name from pyproject). */
export const HPE_WHEEL = "human_possibility_engine-0.1.0-py3-none-any.whl";

export type EngineStage = "idle" | "runtime" | "packages" | "wheel" | "import" | "ready" | "error";

export interface EngineProgress {
  stage: EngineStage;
  message: string;
  startedAt: number;
  elapsedMs: number;
  error?: string;
}

/** Minimal surface of the Pyodide API that this module uses (keeps the real package out of tests). */
export interface PyodideLike {
  loadPackage(names: string[] | string, options?: { messageCallback?: (m: string) => void }): Promise<unknown>;
  pyimport(name: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  globals: { get(name: string): unknown };
}

export interface Bridge {
  /** DecisionRequest JSON → ScenarioResponse JSON (raises on consent/validation errors). */
  generate(requestJson: string, decisionId?: string | null): string;
  evidenceCatalog(): string;
}

type LoadPyodideFn = (opts: { indexURL: string }) => Promise<PyodideLike>;

const listeners = new Set<(p: EngineProgress) => void>();
let progress: EngineProgress = { stage: "idle", message: "Not loaded", startedAt: 0, elapsedMs: 0 };
let bridgePromise: Promise<Bridge> | null = null;

/** Test seam: provide a fake `loadPyodide` instead of injecting the CDN script. */
let loaderOverride: LoadPyodideFn | null = null;
export function setPyodideLoader(fn: LoadPyodideFn | null): void {
  loaderOverride = fn;
  bridgePromise = null;
  publish({ stage: "idle", message: "Not loaded", startedAt: 0, elapsedMs: 0 });
}

export function engineProgress(): EngineProgress {
  return progress;
}

export function subscribeEngine(fn: (p: EngineProgress) => void): () => void {
  listeners.add(fn);
  fn(progress);
  return () => void listeners.delete(fn);
}

function publish(p: Partial<EngineProgress> & { stage: EngineStage; message: string }): void {
  const startedAt = p.startedAt ?? progress.startedAt;
  progress = { ...progress, ...p, startedAt, elapsedMs: startedAt ? Date.now() - startedAt : 0 };
  for (const fn of listeners) fn(progress);
}

function wheelUrl(): string {
  const base = ((import.meta.env.BASE_URL as string | undefined) ?? "/").replace(/\/+$/, "");
  return `${base}/wheels/${HPE_WHEEL}`;
}

function injectScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "1") return resolve();
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () => reject(new Error(`failed to load ${src}`)));
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.addEventListener("load", () => {
      s.dataset.loaded = "1";
      resolve();
    });
    s.addEventListener("error", () => reject(new Error(`failed to load ${src}`)));
    document.head.appendChild(s);
  });
}

async function cdnLoader(): Promise<LoadPyodideFn> {
  await injectScript(`${PYODIDE_INDEX_URL}pyodide.js`);
  const fn = (globalThis as unknown as { loadPyodide?: LoadPyodideFn }).loadPyodide;
  if (!fn) throw new Error("loadPyodide is not defined after loading the Pyodide script");
  return fn;
}

async function boot(): Promise<Bridge> {
  const startedAt = Date.now();
  try {
    publish({ stage: "runtime", message: "Downloading the Python runtime (once, about 20 MB)…", startedAt });
    const loadPyodide = loaderOverride ?? (await cdnLoader());
    const py = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

    publish({ stage: "packages", message: "Loading numpy and pydantic…" });
    await py.loadPackage(["numpy", "pydantic", "micropip"]);

    publish({ stage: "wheel", message: "Installing the Human Possibility Engine package…" });
    const url = new URL(wheelUrl(), typeof location !== "undefined" ? location.href : "http://localhost/").toString();
    await py.runPythonAsync(
      `import micropip\nawait micropip.install(${JSON.stringify(url)}, deps=False)\n`,
    );

    publish({ stage: "import", message: "Starting the scenario engine…" });
    const mod = py.pyimport("hpe.futures.bridge") as {
      generate: (requestJson: string, decisionId?: string | null) => string;
      evidence_catalog: () => string;
    };
    const bridge: Bridge = {
      generate: (requestJson, decisionId) => mod.generate(requestJson, decisionId ?? null),
      evidenceCatalog: () => mod.evidence_catalog(),
    };
    publish({ stage: "ready", message: "Scenario engine ready" });
    return bridge;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    publish({ stage: "error", message: "The scenario engine could not be loaded", error: msg });
    bridgePromise = null; // allow a retry on the next call
    throw new ApiError(`Scenario engine unavailable in this browser: ${msg}`, 503, "/scenarios", { stage: progress.stage });
  }
}

/** The bridge, booting the runtime on first use. Concurrent callers share one boot. */
export function getBridge(): Promise<Bridge> {
  if (!bridgePromise) bridgePromise = boot();
  return bridgePromise;
}

/** Extract the Python exception class name and message from a Pyodide error. */
export function pythonError(e: unknown): { type: string; message: string } {
  const text = e instanceof Error ? e.message : String(e);
  const lines = text.trim().split("\n");
  const last = lines[lines.length - 1] ?? text;
  const m = /^([A-Za-z_][\w.]*):\s*(.*)$/.exec(last);
  const type = m?.[1] ?? "Error";
  return m ? { type: type.split(".").pop() ?? type, message: m[2] ?? "" } : { type: "Error", message: last };
}
