import type { ReactNode } from "react";
import { isMockMode, isStaticMode } from "../api/config";

export function SyntheticBadge() {
  return (
    <span className="badge badge--lab" title="Everything on this page comes from a synthetic agent simulation.">
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
        <circle cx="3" cy="6" r="2" fill="currentColor" />
        <circle cx="9" cy="3" r="2" fill="currentColor" />
        <circle cx="9" cy="9" r="2" fill="currentColor" />
      </svg>
      Synthetic simulation — not real people
    </span>
  );
}

export function IllustrativeBadge({ compact = false }: { compact?: boolean }) {
  return (
    <span className="badge badge--future" title="An illustrative scenario conditioned on stated assumptions. Not a prediction.">
      {compact ? "Illustrative" : "Illustrative scenario"}
    </span>
  );
}

export function LabBanner({ children }: { children?: ReactNode }) {
  return (
    <div className="zone-banner zone-banner--lab" role="note" aria-label="Research Lab disclosure">
      <SyntheticBadge />
      <p>
        {children ??
          "Research Lab: agents are software processes in a synthetic world. Simulated age, affect proxies, birth and death are computational abstractions — not people, feelings or real outcomes."}
      </p>
    </div>
  );
}

export function FutureBanner({ children }: { children?: ReactNode }) {
  return (
    <div className="zone-banner zone-banner--future" role="note" aria-label="Future Self disclosure">
      <IllustrativeBadge />
      <p>
        {children ??
          "Future Self is a decision-exploration tool. It shows illustrative scenarios conditioned on your stated assumptions. It does not predict your future and shows no individualized probabilities."}
      </p>
    </div>
  );
}

/** Shown only on the static (GitHub Pages) deployment: real stored outputs, read-only, no backend. */
export function StaticModeBadge({ enabled }: { enabled?: boolean }) {
  const on = enabled ?? isStaticMode();
  if (!on) return null;
  return (
    <span
      className="chip chip--lab"
      data-testid="static-mode-badge"
      title="This deployment has no backend: every number is pre-rendered from the committed stored real outputs by the real API code, and Future Self runs the repository's own Python engine in your browser. Live simulations need the API (make demo)."
    >
      Static deployment · stored real outputs
    </span>
  );
}

/** Persistent banner shown only when VITE_USE_MOCKS=1. */
export function MockBanner({ enabled }: { enabled?: boolean }) {
  const on = enabled ?? isMockMode();
  if (!on) return null;
  return (
    <div className="mock-banner" role="alert" data-testid="mock-banner">
      MOCK DATA — not real results. Placeholder fixtures for UI development only (VITE_USE_MOCKS=1).
    </div>
  );
}

export function SourceTag({ source }: { source?: string | null }) {
  if (source === "stored_real_output") {
    return (
      <span className="chip chip--lab" title="Precomputed run loaded from reports/, produced by the real simulator code.">
        Stored real output
      </span>
    );
  }
  if (source === "live") return <span className="chip chip--neutral">Live run</span>;
  return null;
}
