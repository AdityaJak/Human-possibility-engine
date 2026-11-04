/**
 * Small descriptive helpers used to *display* raw observations returned by the API.
 * The authoritative inferential statistics (bootstrap CIs, effect sizes, tests) come from
 * the backend's ExperimentResult; anything computed here is labelled as a client-side summary.
 */

import type { OutcomeObservation } from "../types/contracts";

export interface Summary {
  mean: number;
  sd: number;
  n: number;
  lo: number;
  hi: number;
}

export function summarize(values: number[]): Summary {
  const xs = values.filter((v) => Number.isFinite(v));
  const n = xs.length;
  if (n === 0) return { mean: NaN, sd: NaN, n: 0, lo: NaN, hi: NaN };
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
  const half = n > 1 ? (1.96 * sd) / Math.sqrt(n) : 0;
  return { mean, sd, n, lo: mean - half, hi: mean + half };
}

export interface CurvePoint {
  x: number;
  y: number;
  lo?: number;
  hi?: number;
  n?: number;
}

/**
 * Build per-arm curves of `metric` against age (or tick) from raw observations: average within
 * (arm, seed, x) first, then summarise across seeds so the unit of analysis is the seed.
 */
export function curvesFromOutcomes(
  outcomes: OutcomeObservation[],
  metric: string,
  xKey: "age_years" | "tick" = "age_years",
): Map<string, CurvePoint[]> {
  const byArm = new Map<string, Map<number, Map<string, number[]>>>();
  for (const o of outcomes) {
    if (o.metric !== metric) continue;
    const x = o[xKey];
    if (x === null || x === undefined) continue;
    const arm = o.arm ?? "—";
    const xr = Math.round(x * 100) / 100;
    const seed = String(o.seed ?? o.run_id ?? "");
    let xs = byArm.get(arm);
    if (!xs) byArm.set(arm, (xs = new Map()));
    let seeds = xs.get(xr);
    if (!seeds) xs.set(xr, (seeds = new Map()));
    const arr = seeds.get(seed) ?? [];
    arr.push(o.value);
    seeds.set(seed, arr);
  }
  const out = new Map<string, CurvePoint[]>();
  for (const [arm, xs] of byArm) {
    const pts: CurvePoint[] = [];
    for (const [x, seeds] of [...xs.entries()].sort((a, b) => a[0] - b[0])) {
      const perSeed = [...seeds.values()].map((v) => v.reduce((a, b) => a + b, 0) / v.length);
      const s = summarize(perSeed);
      pts.push({ x, y: s.mean, lo: s.lo, hi: s.hi, n: s.n });
    }
    out.set(arm, pts);
  }
  return out;
}

/** Per-seed values of a metric for each arm (one dot per seed). */
export function perSeedValues(outcomes: OutcomeObservation[], metric: string): Map<string, Array<{ seed: string; value: number }>> {
  const acc = new Map<string, Map<string, number[]>>();
  for (const o of outcomes) {
    if (o.metric !== metric) continue;
    const arm = o.arm ?? "—";
    const seed = String(o.seed ?? o.run_id ?? "");
    let m = acc.get(arm);
    if (!m) acc.set(arm, (m = new Map()));
    const arr = m.get(seed) ?? [];
    arr.push(o.value);
    m.set(seed, arr);
  }
  const out = new Map<string, Array<{ seed: string; value: number }>>();
  for (const [arm, m] of acc) {
    out.set(
      arm,
      [...m.entries()]
        .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))
        .map(([seed, vs]) => ({ seed, value: vs.reduce((a, b) => a + b, 0) / vs.length })),
    );
  }
  return out;
}

export function metricsIn(outcomes: OutcomeObservation[]): string[] {
  return [...new Set(outcomes.map((o) => o.metric))].sort();
}

/** Flatten nested numeric leaves of an object into dot paths ({a:{b:1}} → {"a.b":1}). */
export function flattenNumeric(obj: unknown, prefix = "", out: Record<string, number> = {}, depth = 0): Record<string, number> {
  if (depth > 4 || obj === null || obj === undefined) return out;
  if (typeof obj === "number" && Number.isFinite(obj)) {
    out[prefix || "value"] = obj;
    return out;
  }
  if (typeof obj === "boolean") {
    return out;
  }
  if (typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      flattenNumeric(v, prefix ? `${prefix}.${k}` : k, out, depth + 1);
    }
  }
  return out;
}

export function gini(values: number[]): number {
  const xs = values.filter((v) => Number.isFinite(v) && v >= 0).sort((a, b) => a - b);
  const n = xs.length;
  if (n === 0) return NaN;
  const sum = xs.reduce((a, b) => a + b, 0);
  if (sum === 0) return 0;
  let cum = 0;
  for (let i = 0; i < n; i++) cum += (i + 1) * (xs[i] ?? 0);
  return (2 * cum) / (n * sum) - (n + 1) / n;
}
