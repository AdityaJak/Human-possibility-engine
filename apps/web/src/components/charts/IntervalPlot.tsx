import { extent, fmtTick, linear } from "./scale";

export interface IntervalRow {
  id: string;
  label: string;
  estimate: number;
  lo: number;
  hi: number;
  color?: string;
  dots?: number[];
  note?: string;
}

interface Props {
  rows: IntervalRow[];
  title: string;
  xLabel?: string;
  zeroLine?: number | null;
  domain?: [number, number];
  format?: (v: number) => string;
}

const W = 640;
const M = { top: 10, right: 20, bottom: 36, left: 170 };
const ROW = 34;

/**
 * Horizontal interval plot: estimate ± interval per row, optional per-unit dots (e.g. one dot
 * per seed) and a reference line (0 for effect estimates, chance for accuracies).
 */
export function IntervalPlot({ rows, title, xLabel, zeroLine = 0, domain, format = fmtTick }: Props) {
  const H = M.top + M.bottom + Math.max(1, rows.length) * ROW;
  const vals = rows.flatMap((r) => [r.lo, r.hi, r.estimate, ...(r.dots ?? [])]);
  if (zeroLine !== null && zeroLine !== undefined) vals.push(zeroLine);
  const xd = domain ?? extent(vals, 0.08);
  const x = linear(xd, [M.left, W - M.right]);
  const ticks = x.ticks(6);
  const summary = rows.map((r) => `${r.label}: ${format(r.estimate)} [${format(r.lo)}, ${format(r.hi)}]`).join("; ");
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. ${summary}`}>
        <title>{title}</title>
        <g className="grid">
          {ticks.map((t) => (
            <line key={t} x1={x(t)} x2={x(t)} y1={M.top} y2={H - M.bottom} />
          ))}
        </g>
        {zeroLine !== null && zeroLine !== undefined && (
          <line className="refline" x1={x(zeroLine)} x2={x(zeroLine)} y1={M.top} y2={H - M.bottom} />
        )}
        <g className="axis">
          <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
          {ticks.map((t) => (
            <text key={t} x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
              {format(t)}
            </text>
          ))}
          {xLabel && (
            <text x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
              {xLabel}
            </text>
          )}
        </g>
        {rows.map((r, i) => {
          const cy = M.top + i * ROW + ROW / 2;
          const c = r.color ?? "var(--ink)";
          return (
            <g key={r.id}>
              <text x={M.left - 10} y={cy + 4} textAnchor="end" style={{ fill: "var(--ink-2)", fontSize: 12 }}>
                {r.label.length > 26 ? `${r.label.slice(0, 25)}…` : r.label}
              </text>
              {(r.dots ?? []).map((d, j) => (
                <circle key={j} cx={x(d)} cy={cy + ((j % 5) - 2) * 2.2} r={3} fill={c} opacity={0.35} />
              ))}
              <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={c} strokeWidth={3} strokeLinecap="round" />
              <circle cx={x(r.estimate)} cy={cy} r={5.5} fill="var(--bg-elev)" stroke={c} strokeWidth={3} />
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
