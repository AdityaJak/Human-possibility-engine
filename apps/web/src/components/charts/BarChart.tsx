import { fmtTick, linear } from "./scale";

export interface BarDatum {
  label: string;
  value: number;
  color?: string;
}

interface Props {
  data: BarDatum[];
  title: string;
  height?: number;
  yLabel?: string;
  format?: (v: number) => string;
}

const W = 640;
const M = { top: 10, right: 10, bottom: 34, left: 40 };

export function BarChart({ data, title, height = 180, yLabel, format = fmtTick }: Props) {
  const H = height;
  const max = Math.max(1e-9, ...data.map((d) => d.value));
  const y = linear([0, max], [H - M.bottom, M.top]);
  const bw = (W - M.left - M.right) / Math.max(1, data.length);
  const summary = data.map((d) => `${d.label}: ${format(d.value)}`).join(", ");
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}. ${summary}`}>
      <title>{title}</title>
      <g className="grid">
        {y.ticks(4).map((t) => (
          <line key={t} x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
        ))}
      </g>
      <g className="axis">
        {y.ticks(4).map((t) => (
          <text key={t} x={M.left - 6} y={y(t) + 4} textAnchor="end">
            {format(t)}
          </text>
        ))}
        {yLabel && (
          <text transform={`translate(10,${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
            {yLabel}
          </text>
        )}
      </g>
      {data.map((d, i) => {
        const x0 = M.left + i * bw + bw * 0.12;
        const h = H - M.bottom - y(d.value);
        return (
          <g key={d.label}>
            <rect x={x0} y={y(d.value)} width={bw * 0.76} height={Math.max(0, h)} rx={3} fill={d.color ?? "var(--lab)"} />
            <text x={x0 + bw * 0.38} y={H - M.bottom + 14} textAnchor="middle">
              {d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function Sparkline({ values, color = "var(--lab)", label }: { values: number[]; color?: string; label: string }) {
  const W2 = 120;
  const H2 = 28;
  const xs = values.filter((v) => Number.isFinite(v));
  if (xs.length < 2) return <span className="muted">—</span>;
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const x = linear([0, xs.length - 1], [2, W2 - 2]);
  const y = linear([lo, hi === lo ? lo + 1 : hi], [H2 - 3, 3]);
  const d = xs.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  return (
    <svg width={W2} height={H2} viewBox={`0 0 ${W2} ${H2}`} role="img" aria-label={`${label} trend`}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.8} />
    </svg>
  );
}
