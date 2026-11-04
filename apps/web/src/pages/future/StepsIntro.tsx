import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { ErrorState } from "../../components/AsyncState";
import { useFuture } from "./context";
import type { Consent } from "./model";

export function Welcome() {
  return (
    <div className="stack">
      <section className="card">
        <h2>Understand your choices. Explore possible futures. Take the next meaningful step.</h2>
        <p>
          Future Self helps you think through one concrete decision. You tell us what matters to you and the options you are weighing. We show illustrative scenarios
          conditioned on explicit assumptions, simple calculations from the numbers you provide, and small reversible experiments you could try.
        </p>
        <div className="grid grid-2">
          <div>
            <h3>What it can do</h3>
            <ul>
              <li>Lay out trade-offs against values you choose and can edit at any time.</li>
              <li>Show calculations (e.g. cash runway) with formulas and your inputs.</li>
              <li>Name what is unknown, and where prediction is not possible.</li>
              <li>Suggest safe, reversible ways to learn more before deciding.</li>
            </ul>
          </div>
          <div>
            <h3>What it will not do</h3>
            <ul>
              <li>Predict your future or tell you who “you will become”.</li>
              <li>Show success probabilities about you — we have no validated model that could justify one.</li>
              <li>Decide for you, or give medical, legal, financial or relationship verdicts.</li>
              <li>Use results from our synthetic agent society as evidence about your life.</li>
            </ul>
          </div>
        </div>
      </section>
      <section className="card card--flat">
        <h3>Privacy in this demo</h3>
        <p>
          You are identified only by a random pseudonymous id. Nothing is collected until you consent. Do not enter health details, names or other sensitive information — none
          is needed. You can export or delete everything from the Privacy step.
        </p>
        <Link to="/future/consent" className="btn btn--future">
          Continue to consent
        </Link>
      </section>
    </div>
  );
}

export function ConsentStep() {
  const { state, update } = useFuture();
  const navigate = useNavigate();
  const [understand, setUnderstand] = useState(state.consent.demo);
  const [store, setStore] = useState(state.consent.demo);
  const [reflections, setReflections] = useState(state.consent.reflections);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error | null>(null);
  const ready = understand && store;

  async function submit() {
    setBusy(true);
    setErr(null);
    try {
      const now = new Date().toISOString();
      await api.consent({ user_id: state.userId, purpose: "future_self_demo", granted: true, created_at: now });
      await api.consent({ user_id: state.userId, purpose: "reflection_storage", granted: reflections, created_at: now });
      const consent: Consent = { demo: true, reflections, personalization: state.consent.personalization, recordedAt: now };
      update({ consent });
      navigate("/future/north-star");
    } catch (e) {
      setErr(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card" aria-labelledby="consent-title">
      <h2 id="consent-title">Your consent</h2>
      <p>
        Pseudonymous demo id: <code>{state.userId}</code>
      </p>
      <div className="stack" style={{ gap: 12 }}>
        <label className="check">
          <input type="checkbox" checked={understand} onChange={(e) => setUnderstand(e.target.checked)} />
          <span>
            <strong>Required.</strong> I understand that scenarios are illustrative, conditioned on stated assumptions, and are not predictions of my future.
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={store} onChange={(e) => setStore(e.target.checked)} />
          <span>
            <strong>Required.</strong> I agree that the values and decision details I enter are stored on this demo server under my pseudonymous id so scenarios can be generated.
            I will not enter sensitive personal information.
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={reflections} onChange={(e) => setReflections(e.target.checked)} />
          <span>Optional. Store my later reflections (what I chose and what happened) so I can review my journey.</span>
        </label>
      </div>
      <div className="row" style={{ marginTop: 16 }}>
        <button type="button" className="btn btn--future" disabled={!ready || busy} onClick={() => void submit()}>
          {busy ? "Recording consent…" : "I consent — continue"}
        </button>
        {!ready && <small>Both required boxes must be checked to continue.</small>}
        <Link to="/future/welcome" className="btn btn--ghost">
          Not now
        </Link>
      </div>
      {err && (
        <div style={{ marginTop: 12 }}>
          <ErrorState error={err} title="Consent could not be recorded" onRetry={() => void submit()} />
          <p className="muted">Without a recorded consent the flow stays locked.</p>
        </div>
      )}
    </section>
  );
}

export function NorthStar() {
  const { state, update } = useFuture();
  const values = state.values;

  function move(i: number, dir: -1 | 1) {
    const j = i + dir;
    if (j < 0 || j >= values.length) return;
    const next = [...values];
    const a = next[i];
    const b = next[j];
    if (!a || !b) return;
    next[i] = b;
    next[j] = a;
    update({ values: next });
  }

  function setWeight(i: number, w: number) {
    update({ values: values.map((v, k) => (k === i ? { ...v, weight: w } : v)) });
  }

  return (
    <section className="card" aria-labelledby="ns-title">
      <h2 id="ns-title">Your North Star</h2>
      <p>
        Rank what matters most to you for this decision (top = most important) and set how much each matters. You can come back and change this at any time; scenarios are
        re-evaluated against your current values.
      </p>
      <ol className="values-list" aria-label="Your values, in rank order">
        {values.map((v, i) => (
          <li key={v.key}>
            <span className="rank" aria-hidden="true">
              {i + 1}
            </span>
            <label htmlFor={`w-${v.key}`}>{v.label}</label>
            <input
              id={`w-${v.key}`}
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={v.weight}
              onChange={(e) => setWeight(i, Number(e.target.value))}
              aria-valuetext={`${v.label} importance ${Math.round(v.weight * 10)} out of 10`}
            />
            <span className="mono" aria-hidden="true">
              {Math.round(v.weight * 10)}/10
            </span>
            <span className="row" style={{ gap: 4 }}>
              <button type="button" className="btn btn--sm" onClick={() => move(i, -1)} disabled={i === 0} aria-label={`Move ${v.label} up`}>
                ↑
              </button>
              <button type="button" className="btn btn--sm" onClick={() => move(i, 1)} disabled={i === values.length - 1} aria-label={`Move ${v.label} down`}>
                ↓
              </button>
            </span>
          </li>
        ))}
      </ol>
      <div className="row" style={{ marginTop: 16 }}>
        <Link to="/future/decision" className="btn btn--future">
          Frame a decision
        </Link>
      </div>
    </section>
  );
}
