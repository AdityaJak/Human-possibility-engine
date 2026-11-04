export type Scale = ((v: number) => number) & { domain: [number, number]; range: [number, number]; ticks: (n?: number) => number[] };

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain[0] === domain[1] ? [domain[0] - 1, domain[1] + 1] : domain;
  const [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  f.domain = [d0, d1];
  f.range = range;
  f.ticks = (n = 5) => niceTicks(d0, d1, n);
  return f;
}

export function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const span = hi - lo;
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const out: number[] = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export function extent(values: number[], pad = 0): [number, number] {
  const xs = values.filter((v) => Number.isFinite(v));
  if (!xs.length) return [0, 1];
  let lo = Math.min(...xs);
  let hi = Math.max(...xs);
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return `${(v / 1000).toFixed(a >= 10000 ? 0 : 1)}k`;
  if (a >= 10 || Number.isInteger(v)) return String(Math.round(v * 100) / 100);
  return v.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}
