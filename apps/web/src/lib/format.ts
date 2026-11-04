export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function fmtFixed(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

/** Proportion in [0,1] displayed as a share of items (e.g. accuracy), NOT a probability claim. */
export function fmtShare(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Math.round(v).toLocaleString();
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 120) return `${seconds.toFixed(1)} s`;
  if (seconds < 7200) return `${(seconds / 60).toFixed(1)} min`;
  return `${(seconds / 3600).toFixed(2)} h`;
}

export function fmtCI(lo: number | null | undefined, hi: number | null | undefined, digits = 2): string {
  if (lo === null || lo === undefined || hi === null || hi === undefined) return "";
  return `[${fmtFixed(lo, digits)}, ${fmtFixed(hi, digits)}]`;
}

export function titleCase(s: string): string {
  return s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function shortId(id: string, n = 10): string {
  return id.length > n ? `${id.slice(0, n)}…` : id;
}

/** Simulated year → "Year 12, day 84" */
export function fmtSimTime(year: number | undefined, day?: number): string {
  if (year === undefined) return "—";
  const y = Math.floor(year);
  if (day === undefined) return `Year ${y}`;
  const doy = Math.max(0, Math.round(day - y * 365));
  return `Year ${y} · day ${doy}`;
}
