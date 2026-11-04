import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/hooks";
import { ClaimStatusChip } from "../components/StatusChip";
import type { ClaimRecord } from "../types/contracts";

const STATUS_ORDER: ClaimRecord["status"][] = ["supported", "not_supported", "inconclusive", "not_tested", "blocked"];

function ClaimsSnapshot() {
  const { data, error, loading } = useApi((s) => api.claims(s), []);
  if (loading) return <p className="muted">Loading the current evidence ledger…</p>;
  if (error)
    return (
      <p className="muted">
        The evidence ledger is served by the API, which is not reachable right now. Nothing on this page claims a result on its own.
      </p>
    );
  if (!data || !data.length) return <p className="muted">No claims have been recorded yet.</p>;
  const counts = new Map<string, number>();
  for (const c of data) counts.set(c.status, (counts.get(c.status) ?? 0) + 1);
  return (
    <div className="row" aria-label="Claim status counts">
      {STATUS_ORDER.filter((s) => counts.get(s)).map((s) => (
        <span key={s} className="row" style={{ gap: 6 }}>
          <ClaimStatusChip status={s} /> <strong>{counts.get(s)}</strong>
        </span>
      ))}
      <Link to="/lab/evidence">Open the full ledger →</Link>
    </div>
  );
}

export function Landing() {
  return (
    <div className="stack" style={{ gap: 28 }}>
      <section className="hero" aria-labelledby="hero-title">
        <div className="eyebrow">Human Possibility Engine</div>
        <h1 id="hero-title">Better decisions for your future self — grounded in evidence, not certainty theater.</h1>
        <p className="lede">
          People cannot reliably evaluate how today's choices shape their futures. We are building the science first: a synthetic society of software agents that{" "}
          <em>acquire</em> measurable skills through learning, controlled experiments against honest baselines, and a consumer tool that explores choices without
          pretending to predict them.
        </p>
        <div className="row" style={{ marginTop: 18 }}>
          <Link to="/lab/society" className="btn btn--lab">
            Enter the Research Lab
          </Link>
          <Link to="/future" className="btn btn--future">
            Try Future Self
          </Link>
          <Link to="/demo" className="btn">
            Investor walkthrough (≈6 min)
          </Link>
        </div>
      </section>

      <section className="grid grid-2" aria-label="Two separate zones">
        <article className="card" style={{ borderTop: "5px solid var(--lab)" }}>
          <div className="eyebrow">Research Lab · synthetic</div>
          <h2>Do agents that develop from experience learn better?</h2>
          <p>
            Agents live from simulated birth to later life in households, schools and workplaces. Their skill is measured only on <strong>held-out</strong> novel-symbol
            tasks with learning switched off — never from biographies, confidence or prose.
          </p>
          <ul>
            <li>Same inputs and budget for four agent classes: static persona, memory-only, plain RL, developmental learner.</li>
            <li>Twin societies cloned from one population, one randomized intervention, paired seeds.</li>
            <li>Negative and null results are shown with the same prominence as positive ones.</li>
          </ul>
          <Link to="/lab/learning">See the baseline comparison →</Link>
        </article>
        <article className="card" style={{ borderTop: "5px solid var(--future)" }}>
          <div className="eyebrow">Future Self · illustrative</div>
          <h2>Explore choices. Inspect assumptions. Take a reversible next step.</h2>
          <p>
            Compare two to four paths against your own values. Every number traces to your inputs, a shown calculation, or a cited source. Where we cannot predict, we say
            so.
          </p>
          <ul>
            <li>No individualized probabilities — ever. No "you will become".</li>
            <li>Nothing from the synthetic society is used as evidence about your life.</li>
            <li>Consent first; export or delete your demo data at any time.</li>
          </ul>
          <Link to="/future">Start with your North Star →</Link>
        </article>
      </section>

      <section className="card" aria-labelledby="honest-title">
        <h2 id="honest-title">What is built — and what remains unproven</h2>
        <div className="grid grid-2">
          <div>
            <h3>Built (engineering, inspectable in the repository)</h3>
            <ul>
              <li>Deterministic, replayable developmental-agent society simulator that runs with no API keys.</li>
              <li>Novel-symbol task world with held-out symbols and an unseen transfer family.</li>
              <li>Learning policies and baselines behind one shared contract; preregistered experiment runner.</li>
              <li>This dashboard and the Future Self prototype.</li>
            </ul>
          </div>
          <div>
            <h3>Unproven (open research questions)</h3>
            <ul>
              <li>Whether any simulated effect corresponds to real human development.</li>
              <li>Whether shared representations transfer to genuinely new task families.</li>
              <li>Whether decision support improves real outcomes — this needs prospective human studies.</li>
              <li>Validated personal forecasting of any kind.</li>
            </ul>
          </div>
        </div>
        <h3 style={{ marginTop: 10 }}>Current evidence ledger</h3>
        <ClaimsSnapshot />
      </section>

      <section className="notice notice--info" aria-label="Agents are not people">
        <p>
          <strong>Agents are not people.</strong> Simulated age, birth, death and affect variables are computational abstractions. They do not imply consciousness, experienced
          emotion or moral personhood.
        </p>
      </section>
    </div>
  );
}
