import { Suspense, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api/client";
import { apiBaseUrl, isMockMode } from "../api/config";
import { CONTRACT_VERSION } from "../types/contracts";
import type { Health } from "../types/contracts";
import { MockBanner, StaticModeBadge } from "./Badges";
import { Loading } from "./AsyncState";

const LAB_LINKS: Array<{ to: string; label: string }> = [
  { to: "/lab/society", label: "Live society" },
  { to: "/lab/learning", label: "Proof of learning" },
  { to: "/lab/experiment", label: "Twin-society experiment" },
  { to: "/lab/evidence", label: "Evidence" },
  { to: "/lab/benchmarks", label: "Benchmarks" },
];

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--ink)" />
      <path d="M8 24 C12 18 13 14 16 10 M16 10 C19 14 20 18 24 24 M16 10 L16 6" stroke="var(--lab)" strokeWidth="2.4" fill="none" strokeLinecap="round" />
      <circle cx="8" cy="24" r="2.2" fill="#f2b45a" />
      <circle cx="24" cy="24" r="2.2" fill="var(--future)" />
      <circle cx="16" cy="6" r="2.2" fill="var(--lab)" />
    </svg>
  );
}

function ApiStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    const ctrl = new AbortController();
    api
      .health(ctrl.signal)
      .then((h) => setHealth(h))
      .catch(() => {
        if (!ctrl.signal.aborted) setDown(true);
      });
    return () => ctrl.abort();
  }, []);
  if (down)
    return (
      <span>
        API: <span className="chip chip--bad">unreachable</span> <code>{apiBaseUrl()}</code>
      </span>
    );
  if (!health) return <span>API: checking…</span>;
  const mismatch = health.contract_version !== CONTRACT_VERSION;
  return (
    <span>
      API: <span className={`chip chip--${health.status === "ok" ? "ok" : "warn"}`}>{health.status}</span> v{health.version} · contract {health.contract_version}
      {mismatch && <span className="chip chip--warn"> web expects {CONTRACT_VERSION}</span>} · language model: {health.llm_provider ?? "none"}
    </span>
  );
}

export function Layout() {
  const loc = useLocation();
  const inLab = loc.pathname.startsWith("/lab");
  useEffect(() => {
    // move focus to main content on route change for screen-reader users
    const main = document.getElementById("main");
    if (main && loc.hash === "") main.focus({ preventScroll: true });
    window.scrollTo?.(0, 0);
  }, [loc.pathname, loc.hash]);

  return (
    <div className="app">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <header className="site-header">
        <MockBanner />
        <div className="site-header__inner">
          <NavLink to="/" className="brand" aria-label="Human Possibility Engine home">
            <Logo />
            <span>
              Human Possibility Engine
              <small>Synthetic research prototype</small>
            </span>
          </NavLink>
          <nav className="nav" aria-label="Primary">
            <NavLink to="/lab/society" className={({ isActive }) => `zone-lab ${isActive || inLab ? "active" : ""}`}>
              Research Lab
            </NavLink>
            <NavLink to="/future" className={({ isActive }) => `zone-future ${isActive ? "active" : ""}`}>
              Future Self
            </NavLink>
            <NavLink to="/demo">Investor demo</NavLink>
          </nav>
        </div>
        {inLab && (
          <nav className="subnav" aria-label="Research Lab">
            {LAB_LINKS.map((l) => (
              <NavLink key={l.to} to={l.to}>
                {l.label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>
      <main id="main" tabIndex={-1} style={{ outline: "none" }}>
        <Suspense fallback={<Loading label="Loading page…" />}>
          <Outlet />
        </Suspense>
      </main>
      <footer className="site-footer">
        <div className="site-footer__inner">
          <strong>Synthetic research prototype; not validated for predicting human futures.</strong>
          <span>
            Research Lab results come from a synthetic simulation and may simply reflect assumptions built into it. Future Self shows illustrative scenarios, never
            individualized probabilities, and never uses simulation results as evidence about a person.
          </span>
          <span>
            <ApiStatus /> <StaticModeBadge /> {isMockMode() && <span className="badge badge--mock">Mock mode</span>}
          </span>
        </div>
      </footer>
    </div>
  );
}
