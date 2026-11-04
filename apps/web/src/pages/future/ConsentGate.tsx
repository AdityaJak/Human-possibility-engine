import type { ReactNode } from "react";
import { Link } from "react-router-dom";

/** Blocks a Future Self step until the required consent has been given. */
export function ConsentGate({ consented, children }: { consented: boolean; children: ReactNode }) {
  if (consented) return <>{children}</>;
  return (
    <section className="state" role="region" aria-labelledby="gate-title" data-testid="consent-gate">
      <h2 id="gate-title">Consent needed first</h2>
      <p>This step uses the values and decision details you enter. We need your explicit consent before collecting anything.</p>
      <Link className="btn btn--future" to="/future/consent">
        Review consent
      </Link>
    </section>
  );
}

export function NeedsScenarios({ has, children }: { has: boolean; children: ReactNode }) {
  if (has) return <>{children}</>;
  return (
    <section className="state">
      <h2>No scenarios yet</h2>
      <p>Frame a decision in the Decision Studio first.</p>
      <Link className="btn btn--future" to="/future/decision">
        Open Decision Studio
      </Link>
    </section>
  );
}
