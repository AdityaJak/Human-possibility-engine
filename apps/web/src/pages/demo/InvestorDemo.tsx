import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import { ClaimStatusChip } from "../../components/StatusChip";
import { SourceTag } from "../../components/Badges";

interface DemoStep {
  id: string;
  title: string;
  minutes: number;
  narration: string[];
  show: string[];
  links: Array<{ to: string; label: string }>;
  caveat?: string;
}

const STEPS: DemoStep[] = [
  {
    id: "problem",
    title: "The problem and our standard of honesty",
    minutes: 0.75,
    narration: [
      "People can't reliably evaluate how today's choices shape their future. Existing tools offer generic advice, confident-sounding forecasts, or entertaining avatars — none with demonstrated causal accuracy.",
      "We are building the science first. Everything you'll see is either measured by code in the repository, or explicitly labelled as unproven.",
    ],
    show: ["Landing page: the 'built vs unproven' panel and the live evidence-ledger counts."],
    links: [{ to: "/", label: "Open landing page" }],
  },
  {
    id: "society",
    title: "A live developmental society",
    minutes: 1.25,
    narration: [
      "This is a synthetic town of software agents: households, a school, workplaces, a plaza. Agents are born, go to school, work, form relationships, age and reach the end of their simulated lives.",
      "It runs on a laptop CPU with no language-model calls. Pause, step, reset or switch the seed — the same seed reproduces the same event stream.",
    ],
    show: ["Press Play, then Pause. Step one year. Point at employment, age structure and the events feed.", "Replay a stored real run with the scrubber."],
    links: [{ to: "/lab/society", label: "Open Live society" }],
    caveat: "Agents are not people. Simulated age, affect and death are abstractions.",
  },
  {
    id: "agent",
    title: "One agent's learned skills",
    minutes: 1.25,
    narration: [
      "Pick any agent. Its timeline is sourced: every entry shows the event sequence number and the code mechanism that produced it.",
      "Its skill curve is measured on held-out symbols it has never seen in practice, with learning switched off. Below it: the learned parameters actually changing over its life. A biography or a chat transcript would not count as learning.",
    ],
    show: ["Search an agent → Inspect.", "Skill curves vs the chance line; parameter-change chart; self-estimate vs measured accuracy."],
    links: [{ to: "/lab/society", label: "Choose an agent in Live society" }],
  },
  {
    id: "twin",
    title: "Twin-society experiment",
    minutes: 1.25,
    narration: [
      "We clone one initial population into control and treatment, share random streams, and randomize one intervention — here, tutoring access. Each seed is a replication.",
      "We show per-seed outcomes, paired differences, the preregistered bootstrap interval, sensitivity checks and a mechanism check with a non-learning agent class.",
    ],
    show: ["Per-seed dots and the paired difference.", "The Assumptions & validity panel; download a raw run log."],
    links: [{ to: "/lab/experiment", label: "Open Twin-society experiment" }],
    caveat: "Simulated effects may simply reflect assumptions built into the world. They are not real-world causal estimates.",
  },
  {
    id: "baselines",
    title: "Baseline comparison, reproducibility and caveats",
    minutes: 1,
    narration: [
      "The decisive question: does an agent that develops from experience predict and learn better than a written-biography agent, a memory-only agent, or plain RL — under the same inputs and budget?",
      "Whatever the answer is, it is shown here, including negative results. Every claim on the Evidence page links to run ids and a config; human-data validation is listed as its own gate.",
    ],
    show: ["Proof of learning: curves, effects table, transfer and ablations.", "Evidence page: status chips, limitations."],
    links: [
      { to: "/lab/learning", label: "Open Proof of learning" },
      { to: "/lab/evidence", label: "Open Evidence" },
      { to: "/lab/benchmarks", label: "Open Benchmarks" },
    ],
  },
  {
    id: "future",
    title: "Future Self — the consumer expression of the science",
    minutes: 1,
    narration: [
      "Future Self helps a person think through one real decision: values first, then two to four options, then illustrative scenarios conditioned on explicit assumptions.",
      "Every number is either the user's own input, a shown calculation, or a cited source. Where we cannot predict, it says so — no success percentages. It ends with small reversible experiments, not a verdict.",
    ],
    show: ["Consent → North Star → Decision Studio (use the job/startup example) → Possible futures → Evidence & uncertainty → Experiment planner."],
    links: [{ to: "/future/welcome", label: "Open Future Self" }],
    caveat: "Nothing from the synthetic society is used as evidence about a person's life.",
  },
];

const TOTAL = STEPS.reduce((a, s) => a + s.minutes, 0);

function LiveFacts() {
  const runs = useApi((s) => api.listRuns(s), []);
  const claims = useApi((s) => api.claims(s), []);
  const health = useApi((s) => api.health(s), []);
  return (
    <section className="card card--flat" aria-label="Live system facts">
      <h2 style={{ fontSize: "1rem" }}>What the running system reports right now</h2>
      <ul style={{ margin: 0, paddingLeft: 18, fontSize: "0.9rem" }}>
        <li>
          API: {health.data ? `${health.data.status}, contract ${health.data.contract_version}, language model: ${health.data.llm_provider ?? "none"}` : health.error ? "not reachable" : "checking…"}
        </li>
        <li>
          Runs available: {runs.data ? runs.data.length : runs.error ? "—" : "…"}
          {runs.data && runs.data.some((r) => r.source === "stored_real_output") && (
            <>
              {" "}
              (incl. <SourceTag source="stored_real_output" />)
            </>
          )}
        </li>
        <li>
          Claims ledger:{" "}
          {claims.data
            ? (["supported", "not_supported", "inconclusive", "not_tested", "blocked"] as const)
                .map((s) => [s, claims.data?.filter((c) => c.status === s).length ?? 0] as const)
                .filter(([, n]) => n > 0)
                .map(([s, n]) => (
                  <span key={s} style={{ marginRight: 8 }}>
                    <ClaimStatusChip status={s} /> {n}
                  </span>
                ))
            : claims.error
              ? "—"
              : "…"}
        </li>
      </ul>
    </section>
  );
}

export default function InvestorDemo() {
  const [i, setI] = useState(0);
  const step = STEPS[i] as DemoStep;
  const go = useCallback((d: number) => setI((x) => Math.max(0, Math.min(STEPS.length - 1, x + d))), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "ArrowRight") go(1);
      if (e.key === "ArrowLeft") go(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go]);

  const elapsed = STEPS.slice(0, i).reduce((a, s) => a + s.minutes, 0);

  return (
    <div className="stack">
      <header className="page-head">
        <div className="eyebrow">Investor walkthrough · about {Math.round(TOTAL)} minutes</div>
        <h1>What we built, what the agents learned, and what remains unproven</h1>
        <p>
          A guided route through the running system. Use the step list or <span className="kbd">←</span> <span className="kbd">→</span>. Every result shown in the linked pages comes
          from the live API; this page contains narration only and no research numbers.
        </p>
      </header>
      <div className="split" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.6fr)" }}>
        <nav aria-label="Walkthrough steps">
          <ol className="demo-steps">
            {STEPS.map((s, k) => (
              <li key={s.id}>
                <button type="button" onClick={() => setI(k)} aria-current={k === i ? "step" : undefined}>
                  <strong>{k + 1}</strong>
                  <span>{s.title}</span>
                  <span className="muted" style={{ fontSize: "0.8rem" }}>
                    {s.minutes} min
                  </span>
                </button>
              </li>
            ))}
          </ol>
          <div style={{ marginTop: 14 }}>
            <LiveFacts />
          </div>
        </nav>
        <article className="card" aria-labelledby="demo-step-title" aria-live="polite">
          <div className="eyebrow">
            Step {i + 1} of {STEPS.length} · {elapsed.toFixed(1)}–{(elapsed + step.minutes).toFixed(1)} min
          </div>
          <h2 id="demo-step-title">{step.title}</h2>
          <h3 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--ink-3)" }}>Say</h3>
          {step.narration.map((p) => (
            <p key={p}>{p}</p>
          ))}
          <h3 style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--ink-3)" }}>Show</h3>
          <ul>
            {step.show.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
          {step.caveat && (
            <p className="notice">
              <strong>Caveat to state out loud:</strong> {step.caveat}
            </p>
          )}
          <div className="row" style={{ marginTop: 14 }}>
            {step.links.map((l) => (
              <Link key={l.to + l.label} to={l.to} className="btn btn--primary">
                {l.label} →
              </Link>
            ))}
          </div>
          <div className="row" style={{ marginTop: 20 }}>
            <button type="button" className="btn" onClick={() => go(-1)} disabled={i === 0}>
              ← Previous
            </button>
            <span className="spacer" />
            <button type="button" className="btn" onClick={() => go(1)} disabled={i === STEPS.length - 1}>
              Next →
            </button>
          </div>
        </article>
      </div>
      <section className="card card--flat" aria-labelledby="repro-title">
        <h2 id="repro-title" style={{ fontSize: "1.05rem" }}>
          Reproduce a stated result yourself
        </h2>
        <pre className="json" tabIndex={0}>{`make setup        # Python + web dependencies, no API keys
make experiments  # rerun preregistered experiments from seeds
make demo         # API + web app locally`}</pre>
        <p className="muted" style={{ fontSize: "0.84rem", marginTop: 8 }}>
          Compare the regenerated <code>reports/experiments/*/result.json</code> with the values on the Evidence page.
        </p>
      </section>
    </div>
  );
}
