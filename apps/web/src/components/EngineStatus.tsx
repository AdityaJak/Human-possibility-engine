/**
 * Progress of the in-browser scenario engine (static deployment only).
 * Reads `subscribeEngine` from the Pyodide loader; renders nothing once the engine is ready
 * and nothing at all outside static mode.
 */

import { useEffect, useState } from "react";
import { isStaticMode } from "../api/config";
import { engineProgress, subscribeEngine, type EngineProgress } from "../api/static/pyodide";

export function EngineStatus({ active }: { active: boolean }) {
  const [p, setP] = useState<EngineProgress>(engineProgress());
  useEffect(() => subscribeEngine(setP), []);
  if (!isStaticMode() || !active) return null;
  if (p.stage === "idle" || p.stage === "ready") return null;
  const seconds = Math.round(p.elapsedMs / 1000);
  return (
    <div className={`notice ${p.stage === "error" ? "notice--bad" : ""}`} role="status" aria-live="polite" data-testid="engine-status">
      <strong>{p.stage === "error" ? "Scenario engine unavailable" : "Loading the scenario engine in your browser (about 20 MB, once)…"}</strong>
      <div className="muted">
        {p.message}
        {p.stage !== "error" && seconds > 0 ? ` · ${seconds}s` : ""}
        {p.error ? ` — ${p.error}` : ""}
      </div>
      {p.stage !== "error" && (
        <div className="muted" style={{ fontSize: "0.85rem", marginTop: 4 }}>
          This static site has no server: the same Python engine the API uses runs here, in your browser. Your answers never leave this device.
        </div>
      )}
    </div>
  );
}
