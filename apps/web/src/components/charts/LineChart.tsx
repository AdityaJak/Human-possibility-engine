import type { MouseEvent } from "react";
import { useId, useMemo, useState } from "react";
import { extent, fmtTick, linear } from "./scale";

export interface LinePoint {
  x: number;
  y: number;
  lo?: number;
  hi?: number;
}

export interface LineSeries {
  id: string;
  label: string;
  color: string;
  points: LinePoint[];
  dashed?: boolean;
}

export interface RefLine {
  y: number;
  label: string;
}

interface Props {
  series: LineSeries[];
  title: string;
  xLabel?: string;
  yLabel?: string;
  yDomain?: [number, number];
  xDomain?: [number, number];
  refLines?: RefLine[];
  height?: number;
  showBands?: boolean;
  formatY?: (v: number) => string;
}

const W = 640;
const M = { top: 14, right: 18, bottom: 40, left: 52 };

export function LineChart({ series, title, xLabel, yLabel, yDomain, xDomain, refLines = [], height = 260, showBands = true, formatY = fmtTick }: Props) {
  const clipId = useId();
  const [hover, setHover] = useState<number | null>(null);
  const H = height;
  const all = series.flatMap((s) => s.points);
  const xs = xDomain ?? extent(all.map((p) => p.x));
  const ys = yDomain ?? extent([...all.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]), ...refLines.map((r) => r.y)], 0.08);
  const x = linear(xs, [M.left, W - M.right]);
  const y = linear(ys, [H - M.bottom, M.top]);
  const xticks = x.ticks(6);
  const yticks = y.ticks(5);

  const xsSorted = useMemo(() => [...new Set(all.map((p) => p.x))].sort((a, b) => a - b), [all]);

  const summary = series
    .map((s) => {
      const last = s.points[s.points.length - 1];
      return last ? `${s.label}: last value ${formatY(last.y)} at ${fmtTick(last.x)}` : `${s.label}: no data`;
    })
    .join("; ");

  function onMove(e: MouseEvent<SVGSVGElement>) {
    if (!xsSorted.length) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = xsSorted[0] ?? 0;
    for (const v of xsSorted) if (Math.abs(x(v) - px) < Math.abs(x(best) - px)) best = v;
    setHover(best);
  }

  return (
    <figure style={{ margin: 0 }}>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title}. ${summary}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <title>{title}</title>
        <defs>
          <clipPath id={clipId}>
            <rect x={M.left} y={M.top} width={W - M.left - M.right} height={H - M.top - M.bottom} />
          </clipPath>
        </defs>
        <g className="grid">
          {yticks.map((t) => (
            <line key={t} x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
          ))}
        </g>
        <g className="axis">
          <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
          {xticks.map((t) => (
            <g key={t} transform={`translate(${x(t)},${H - M.bottom})`}>
              <line y2={4} />
              <text y={16} textAnchor="middle">
                {fmtTick(t)}
              </text>
            </g>
          ))}
          {yticks.map((t) => (
            <text key={t} x={M.left - 8} y={y(t) + 4} textAnchor="end">
              {formatY(t)}
            </text>
          ))}
          {xLabel && (
            <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle">
              {xLabel}
            </text>
          )}
          {yLabel && (
            <text transform={`translate(13,${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
              {yLabel}
            </text>
          )}
        </g>
        <g clipPath={`url(#${clipId})`}>
          {refLines.map((r) => (
            <g key={r.label}>
              <line className="refline" x1={M.left} x2={W - M.right} y1={y(r.y)} y2={y(r.y)} />
              <text x={W - M.right - 4} y={y(r.y) - 4} textAnchor="end">
                {r.label}
              </text>
            </g>
          ))}
          {showBands &&
            series.map((s) => {
              const pts = s.points.filter((p) => p.lo !== undefined && p.hi !== undefined && Number.isFinite(p.lo) && Number.isFinite(p.hi));
              if (pts.length < 2) return null;
              const top = pts.map((p) => `${x(p.x)},${y(p.hi as number)}`).join(" L");
              const bot = [...pts]
                .reverse()
                .map((p) => `${x(p.x)},${y(p.lo as number)}`)
                .join(" L");
              return <path key={`band-${s.id}`} d={`M${top} L${bot} Z`} fill={s.color} opacity={0.14} />;
            })}
          {series.map((s) => {
            const pts = s.points.filter((p) => Number.isFinite(p.y));
            if (!pts.length) return null;
            const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(" ");
            return (
              <g key={s.id}>
                <path d={d} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dashed ? "6 4" : undefined} />
                {pts.length <= 40 && pts.map((p) => <circle key={p.x} cx={x(p.x)} cy={y(p.y)} r={2.6} fill={s.color} />)}
              </g>
            );
          })}
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={H - M.bottom} stroke="var(--ink-3)" strokeWidth={1} />}
        </g>
        {hover !== null && (
          <g transform={`translate(${Math.min(x(hover) + 8, W - 190)},${M.top + 4})`} pointerEvents="none">
            <rect width={182} height={16 + series.length * 15} rx={6} fill="var(--bg-elev)" stroke="var(--line-strong)" />
            <text x={8} y={13} style={{ fontWeight: 700 }}>
              {xLabel ? `${xLabel}: ` : ""}
              {fmtTick(hover)}
            </text>
            {series.map((s, i) => {
              const p = s.points.find((q) => q.x === hover);
              return (
                <text key={s.id} x={8} y={28 + i * 15}>
                  <tspan fill={s.color}>■ </tspan>
                  {s.label.slice(0, 18)}: {p ? formatY(p.y) : "—"}
                  {p && p.lo !== undefined && p.hi !== undefined ? ` [${formatY(p.lo)}, ${formatY(p.hi)}]` : ""}
                </text>
              );
            })}
          </g>
        )}
      </svg>
      <figcaption className="legend">
        {series.map((s) => (
          <span key={s.id}>
            <i style={{ background: s.color }} aria-hidden="true" />
            {s.label}
          </span>
        ))}
        {refLines.map((r) => (
          <span key={r.label}>
            <i style={{ background: "var(--ink-3)", height: 2 }} aria-hidden="true" />
            {r.label}
          </span>
        ))}
        {showBands && series.some((s) => s.points.some((p) => p.lo !== undefined)) && <span className="muted">Shaded: interval shown with each point</span>}
      </figcaption>
    </figure>
  );
}
