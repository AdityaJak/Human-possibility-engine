import type { MouseEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentFrame, Place, WorldMap } from "../types/contracts";
import { ACTIVITY_COLORS } from "../lib/families";

export type ColorBy = "activity" | "age";

interface Props {
  map: WorldMap;
  frames: AgentFrame[];
  colorBy?: ColorBy;
  selectedId?: string | null;
  onSelect?: (id: string) => void;
  height?: number;
}

const PLACE_STYLE: Record<Place["kind"], { color: string; size: number; label: string }> = {
  home: { color: "--line-strong", size: 5, label: "Home" },
  school: { color: "--s1", size: 14, label: "School" },
  workplace: { color: "--s2", size: 13, label: "Workplace" },
  gathering: { color: "--s4", size: 13, label: "Gathering place" },
  tutoring_center: { color: "--s3", size: 11, label: "Tutoring centre" },
  elder_center: { color: "--s6", size: 11, label: "Elder centre" },
};

const AGE_BANDS: Array<{ max: number; label: string; color: string }> = [
  { max: 3, label: "0–2", color: "--s5" },
  { max: 12, label: "3–11", color: "--s2" },
  { max: 18, label: "12–17", color: "--s3" },
  { max: 65, label: "18–64", color: "--s1" },
  { max: 1000, label: "65+", color: "--s4" },
];

function cssVar(name: string, fallback = "#888"): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function resolve(c: string): string {
  const m = /^var\((--[\w-]+)\)$/.exec(c);
  return m ? cssVar(m[1] as string) : c;
}

function agentColor(f: AgentFrame, colorBy: ColorBy): string {
  if (colorBy === "age") {
    const band = AGE_BANDS.find((b) => f.age < b.max) ?? AGE_BANDS[AGE_BANDS.length - 1];
    return cssVar(band?.color ?? "--s1");
  }
  return resolve(ACTIVITY_COLORS[f.activity ?? ""] ?? "var(--ink-3)");
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Canvas map of the synthetic town. Agent positions animate between consecutive frames
 * (disabled under prefers-reduced-motion). Click an agent dot to select it.
 */
export function SocietyMap({ map, frames, colorBy = "activity", selectedId, onSelect, height = 420 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const prevRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const shownRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const [width, setWidth] = useState(800);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(Math.max(280, Math.floor(w)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const aspect = map.height > 0 && map.width > 0 ? map.height / map.width : 0.6;
  const cssH = Math.min(height, Math.round(width * aspect)) || height;
  const scale = Math.min(width / (map.width || 1), cssH / (map.height || 1));
  const offX = (width - (map.width || 1) * scale) / 2;
  const offY = (cssH - (map.height || 1) * scale) / 2;

  const alive = useMemo(() => frames.filter((f) => f.alive), [frames]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext("2d");
    } catch {
      ctx = null;
    }
    if (!ctx) return;
    const c = ctx;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.height = `${cssH}px`;

    const from = new Map(shownRef.current);
    const target = new Map(alive.map((f) => [f.id, { x: f.x, y: f.y }]));
    const reduced = prefersReducedMotion();
    const duration = reduced ? 0 : 280;
    const start = performance.now();
    let raf = 0;

    const toPx = (x: number, y: number) => [offX + x * scale, offY + y * scale] as const;

    const draw = (t: number) => {
      const k = duration === 0 ? 1 : Math.min(1, (t - start) / duration);
      const ease = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.clearRect(0, 0, width, cssH);
      c.fillStyle = cssVar("--bg-sunken", "#eef1f4");
      c.fillRect(0, 0, width, cssH);

      // places
      c.textBaseline = "middle";
      c.font = "600 11px Inter, system-ui, sans-serif";
      for (const p of map.places) {
        const st = PLACE_STYLE[p.kind] ?? PLACE_STYLE.home;
        const [px, py] = toPx(p.x, p.y);
        const s = st.size * Math.max(0.7, Math.min(1.4, scale / 6));
        c.fillStyle = cssVar(st.color);
        c.globalAlpha = p.kind === "home" ? 0.65 : 0.22;
        c.beginPath();
        if (p.kind === "gathering") c.arc(px, py, s * 1.2, 0, Math.PI * 2);
        else if (typeof c.roundRect === "function") c.roundRect(px - s, py - s * 0.8, s * 2, s * 1.6, 4);
        else c.rect(px - s, py - s * 0.8, s * 2, s * 1.6);
        c.fill();
        c.globalAlpha = 1;
        if (p.kind !== "home") {
          c.strokeStyle = cssVar(st.color);
          c.lineWidth = 1.5;
          c.stroke();
          c.fillStyle = cssVar("--ink-2", "#3a4a5a");
          c.fillText(p.name, px + s * 1.25, py - s * 0.9);
        }
      }

      // agents
      const shown = new Map<string, { x: number; y: number }>();
      for (const f of alive) {
        const tgt = target.get(f.id) ?? { x: f.x, y: f.y };
        const src = from.get(f.id) ?? tgt;
        const x = src.x + (tgt.x - src.x) * ease;
        const y = src.y + (tgt.y - src.y) * ease;
        shown.set(f.id, { x, y });
        const [px, py] = toPx(x, y);
        const r = f.age < 12 ? 2.6 : f.age >= 65 ? 3.4 : 3.6;
        c.beginPath();
        c.arc(px, py, r, 0, Math.PI * 2);
        c.fillStyle = agentColor(f, colorBy);
        c.fill();
        if (f.id === selectedId) {
          c.beginPath();
          c.arc(px, py, r + 5, 0, Math.PI * 2);
          c.strokeStyle = cssVar("--ink", "#0f1d2a");
          c.lineWidth = 2;
          c.stroke();
        }
      }
      shownRef.current = shown;
      if (k < 1) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    prevRef.current = target;
    return () => cancelAnimationFrame(raf);
  }, [alive, map, width, cssH, scale, offX, offY, colorBy, selectedId]);

  function onClick(e: MouseEvent<HTMLCanvasElement>) {
    if (!onSelect) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: string | null = null;
    let bestD = 12 * 12;
    for (const [id, p] of shownRef.current) {
      const dx = offX + p.x * scale - mx;
      const dy = offY + p.y * scale - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = id;
      }
    }
    if (best) onSelect(best);
  }

  const counts = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of alive) m.set(f.activity ?? "—", (m.get(f.activity ?? "—") ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [alive]);

  return (
    <div>
      <div className="map-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: cssH }}
          onClick={onClick}
          role="img"
          aria-label={`Map of the synthetic town with ${map.places.length} places and ${alive.length} living agents. ${counts
            .map(([a, n]) => `${n} ${a}`)
            .join(", ")}. Use the agent search list to select an agent with the keyboard.`}
        />
      </div>
      <div className="legend" aria-label="Map legend">
        {colorBy === "activity"
          ? ["study", "work", "socialize", "rest"].map((a) => (
              <span key={a}>
                <i style={{ background: ACTIVITY_COLORS[a], height: 10, width: 10, borderRadius: 5 }} aria-hidden="true" />
                {a}
              </span>
            ))
          : AGE_BANDS.map((b) => (
              <span key={b.label}>
                <i style={{ background: `var(${b.color})`, height: 10, width: 10, borderRadius: 5 }} aria-hidden="true" />
                age {b.label}
              </span>
            ))}
        {(["school", "workplace", "gathering", "tutoring_center", "elder_center"] as const).map((k) => (
          <span key={k}>
            <i style={{ background: `var(${PLACE_STYLE[k].color})`, opacity: 0.5, height: 10, width: 14 }} aria-hidden="true" />
            {PLACE_STYLE[k].label}
          </span>
        ))}
      </div>
    </div>
  );
}
