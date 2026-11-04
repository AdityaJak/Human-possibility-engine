import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import type { PrivacyDeleteResult } from "../../types/contracts";
import { Async, ErrorState } from "../../components/AsyncState";
import { IllustrativeBadge } from "../../components/Badges";
import { CONVERSATION_DISCLOSURE, SUGGESTED_QUESTIONS, simulatedReply } from "../../lib/conversation";
import { titleCase } from "../../lib/format";
import { useFuture } from "./context";
import type { PlannedStep } from "./model";

function addDays(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() + Math.max(1, Math.round(days)));
  return d.toISOString().slice(0, 10);
}

export function ExperimentPlanner() {
  const { state, update } = useFuture();
  const res = state.response;
  const optionLabel = (id: string) => state.draft.options.find((o) => o.option_id === id)?.label ?? id;
  const all = useMemo(
    () =>
      (res?.scenarios ?? []).flatMap((s) =>
        (s.reversible_experiments ?? []).map((e, i) => ({ key: `${s.scenario_id}:${i}`, scenarioId: s.scenario_id, optionLabel: optionLabel(s.option_id), e })),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [res, state.draft.options],
  );
  const planned = new Map(state.plan.map((p) => [p.key, p]));

  function toggle(item: (typeof all)[number]) {
    if (planned.has(item.key)) {
      update({ plan: state.plan.filter((p) => p.key !== item.key) });
    } else {
      const step: PlannedStep = {
        key: item.key,
        scenarioId: item.scenarioId,
        optionLabel: item.optionLabel,
        title: item.e.title,
        description: item.e.description,
        durationDays: item.e.duration_days,
        checkIn: addDays(item.e.duration_days),
        done: false,
      };
      update({ plan: [...state.plan, step] });
    }
  }

  function patch(key: string, p: Partial<PlannedStep>) {
    update({ plan: state.plan.map((x) => (x.key === key ? { ...x, ...p } : x)) });
  }

  return (
    <div className="stack">
      <section className="card">
        <h2>Small, reversible experiments</h2>
        <p>
          Before committing to a path, you can often learn a lot cheaply and safely. Choose any you would genuinely try. You decide; nothing here is done on your behalf.
        </p>
        {all.length === 0 ? (
          <p className="muted">The scenarios did not include experiments.</p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0, display: "grid", gap: 10 }}>
            {all.map((item) => (
              <li key={item.key} className="card card--flat" style={{ padding: 12 }}>
                <label className="check">
                  <input type="checkbox" checked={planned.has(item.key)} onChange={() => toggle(item)} />
                  <span>
                    <strong>{item.e.title}</strong> <span className="muted">· for “{item.optionLabel}” · ~{item.e.duration_days} days · {titleCase(item.e.reversibility ?? "fully_reversible")}</span>
                    <div>{item.e.description}</div>
                    {item.e.learns_about && item.e.learns_about.length > 0 && <div className="muted" style={{ fontSize: "0.84rem" }}>Learns about: {item.e.learns_about.join(", ")}</div>}
                    {item.e.cost_note && <div className="muted" style={{ fontSize: "0.84rem" }}>Cost: {item.e.cost_note}</div>}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section className="card" aria-labelledby="plan-title">
        <h2 id="plan-title">My plan & check-ins</h2>
        {state.plan.length === 0 ? (
          <p className="muted">No experiments selected yet.</p>
        ) : (
          <div className="table-wrap" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  <th>Experiment</th>
                  <th>Check-in date</th>
                  <th>Done</th>
                </tr>
              </thead>
              <tbody>
                {state.plan.map((p) => (
                  <tr key={p.key}>
                    <td>
                      <strong>{p.title}</strong>
                      <div className="muted" style={{ fontSize: "0.82rem" }}>
                        {p.optionLabel}
                      </div>
                    </td>
                    <td>
                      <label htmlFor={`ci-${p.key}`} className="sr-only">
                        Check-in date for {p.title}
                      </label>
                      <input id={`ci-${p.key}`} type="date" value={p.checkIn} onChange={(e) => patch(p.key, { checkIn: e.target.value })} />
                    </td>
                    <td>
                      <input type="checkbox" checked={p.done} onChange={(e) => patch(p.key, { done: e.target.checked })} aria-label={`Mark ${p.title} done`} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="muted" style={{ fontSize: "0.84rem", marginTop: 8 }}>
          Your plan is kept in this browser only. At a check-in, record what happened in Your Journey.
        </p>
        <Link to="/future/journey" className="btn btn--future">
          Go to Your Journey
        </Link>
      </section>
    </div>
  );
}

export function Journey() {
  const { state, update } = useFuture();
  const journey = useApi((s) => api.journey(state.userId, s), [state.userId]);
  const [chosen, setChosen] = useState<string>("");
  const [what, setWhat] = useState("");
  const [withValues, setWithValues] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error | null>(null);
  const decisionId = state.response?.decision_id;

  async function enableReflections() {
    try {
      await api.consent({ user_id: state.userId, purpose: "reflection_storage", granted: true, created_at: new Date().toISOString() });
      update((s) => ({ consent: { ...s.consent, reflections: true } }));
    } catch (e) {
      setErr(e instanceof Error ? e : new Error(String(e)));
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!decisionId) return;
    setBusy(true);
    setErr(null);
    try {
      await api.reflect({
        user_id: state.userId,
        decision_id: decisionId,
        chosen_option_id: chosen || null,
        what_happened: what,
        values_changed: withValues ? state.values : [],
      });
      setWhat("");
      journey.reload();
    } catch (ex) {
      setErr(ex instanceof Error ? ex : new Error(String(ex)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <section className="card" aria-labelledby="refl-title">
        <h2 id="refl-title">Record a reflection</h2>
        <p className="muted">
          A single experience can't show that a choice caused an outcome — but writing down what happened, and how your priorities shifted, helps you decide the next step.
        </p>
        {!state.consent.reflections ? (
          <div className="notice">
            <p>You have not consented to storing reflections.</p>
            <button type="button" className="btn btn--sm btn--future" onClick={() => void enableReflections()}>
              Allow reflection storage
            </button>
          </div>
        ) : !decisionId ? (
          <p className="muted">
            Frame a decision first in the <Link to="/future/decision">Decision Studio</Link>.
          </p>
        ) : (
          <form className="stack" onSubmit={submit} style={{ gap: 12 }}>
            <div className="field">
              <label htmlFor="chosen">What did you choose (so far)?</label>
              <select id="chosen" value={chosen} onChange={(e) => setChosen(e.target.value)}>
                <option value="">Not decided yet</option>
                {state.draft.options.map((o) => (
                  <option key={o.option_id} value={o.option_id}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="what">What happened? What did you learn?</label>
              <textarea id="what" value={what} onChange={(e) => setWhat(e.target.value)} />
            </div>
            <label className="check">
              <input type="checkbox" checked={withValues} onChange={(e) => setWithValues(e.target.checked)} />
              <span>Save my current North Star values with this reflection, so I can see how they change.</span>
            </label>
            <div className="row">
              <button type="submit" className="btn btn--future" disabled={busy || (!what.trim() && !chosen)}>
                {busy ? "Saving…" : "Save reflection"}
              </button>
            </div>
          </form>
        )}
        {err && <ErrorState error={err} title="Could not save" />}
      </section>
      <section className="card" aria-labelledby="journey-title">
        <h2 id="journey-title">Your journey</h2>
        <Async {...journey} loadingLabel="Loading your journey…">
          {(j) => (
            <div className="stack">
              <div>
                <h3>Decisions explored ({j.decisions.length})</h3>
                {j.decisions.length === 0 ? (
                  <p className="muted">None yet.</p>
                ) : (
                  <ul>
                    {j.decisions.map((d, i) => (
                      <li key={String(d.decision_id ?? i)}>
                        {String(d.question ?? d.decision_id ?? "Decision")}
                        {d.created_at ? <span className="muted"> · {String(d.created_at).slice(0, 10)}</span> : null}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div>
                <h3>Reflections ({j.reflections.length})</h3>
                {j.reflections.length === 0 ? (
                  <p className="muted">None yet.</p>
                ) : (
                  <ol className="timeline" tabIndex={0} aria-label="Reflections">
                    {j.reflections.map((r, i) => (
                      <li key={`${r.decision_id}-${i}`}>
                        <span className="age">{r.created_at ? r.created_at.slice(0, 10) : "—"}</span>
                        <span className="dot" style={{ background: "var(--future)" }} aria-hidden="true" />
                        <span>
                          {r.chosen_option_id ? `Chose: ${state.draft.options.find((o) => o.option_id === r.chosen_option_id)?.label ?? r.chosen_option_id}. ` : ""}
                          {r.what_happened}
                          {r.values_changed && r.values_changed.length > 0 && (
                            <div className="prov">Top values then: {[...r.values_changed].slice(0, 3).map((v) => v.label).join(", ")}</div>
                          )}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            </div>
          )}
        </Async>
      </section>
    </div>
  );
}

interface Msg {
  from: "user" | "sim";
  text: string;
}

export function Conversation() {
  const { state } = useFuture();
  const scenarios = state.response?.scenarios ?? [];
  const [sid, setSid] = useState(scenarios[0]?.scenario_id ?? "");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const scenario = scenarios.find((s) => s.scenario_id === sid) ?? scenarios[0];
  const optionLabel = state.draft.options.find((o) => o.option_id === scenario?.option_id)?.label ?? scenario?.title ?? "";
  const labels = Object.fromEntries(state.values.map((v) => [v.key, v.label]));

  function ask(text: string) {
    if (!scenario || !text.trim()) return;
    const reply = simulatedReply(text, scenario, optionLabel, labels);
    setMsgs((m) => [...m, { from: "user", text: text.trim() }, { from: "sim", text: reply }]);
    setQ("");
  }

  return (
    <section className="card" aria-labelledby="conv-title">
      <h2 id="conv-title">Future Self conversation (optional)</h2>
      <div className="notice notice--bad" role="note" data-testid="conversation-disclosure" style={{ position: "sticky", top: 70, zIndex: 2 }}>
        <p>
          <strong>{CONVERSATION_DISCLOSURE}</strong>
        </p>
      </div>
      <div className="row" style={{ margin: "12px 0" }}>
        <label htmlFor="conv-sc">Talk with the scenario for</label>
        <select
          id="conv-sc"
          value={scenario?.scenario_id ?? ""}
          onChange={(e) => {
            setSid(e.target.value);
            setMsgs([]);
          }}
          style={{ width: "auto" }}
        >
          {scenarios.map((s) => (
            <option key={s.scenario_id} value={s.scenario_id}>
              {state.draft.options.find((o) => o.option_id === s.option_id)?.label ?? s.title}
            </option>
          ))}
        </select>
        <IllustrativeBadge />
      </div>
      <div className="chat" tabIndex={0} aria-live="polite" aria-label="Conversation">
        {msgs.length === 0 && <p className="muted">Ask a question, or pick one below.</p>}
        {msgs.map((m, i) => (
          <div key={i} className={`bubble bubble--${m.from}`}>
            {m.from === "sim" && <span className="bubble__tag">Simulated · illustrative</span>}
            {m.text}
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        {SUGGESTED_QUESTIONS.map((s) => (
          <button key={s} type="button" className="btn btn--sm" onClick={() => ask(s)}>
            {s}
          </button>
        ))}
      </div>
      <form
        className="row"
        style={{ marginTop: 10 }}
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
      >
        <label htmlFor="conv-q" className="sr-only">
          Your question
        </label>
        <input id="conv-q" type="text" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask the illustrative scenario…" style={{ flex: 1, minWidth: 200 }} />
        <button type="submit" className="btn btn--future" disabled={!q.trim()}>
          Ask
        </button>
      </form>
      <p className="muted" style={{ fontSize: "0.82rem", marginTop: 8 }}>
        Replies are assembled from fixed templates and the scenario's own content. No language model is used and nothing is stored.
      </p>
    </section>
  );
}

export function Privacy() {
  const { state, update, reset } = useFuture();
  const navigate = useNavigate();
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<Error | null>(null);
  const [deleted, setDeleted] = useState<PrivacyDeleteResult | null>(null);
  const [confirm, setConfirm] = useState(false);

  async function doExport() {
    setBusy("export");
    setErr(null);
    try {
      const data = await api.exportData(state.userId);
      const blob = new Blob([JSON.stringify({ server: data, browser: { values: state.values, draft: state.draft, plan: state.plan } }, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hpe-future-self-${state.userId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(null);
    }
  }

  async function doDelete() {
    setBusy("delete");
    setErr(null);
    try {
      const r = await api.deleteData(state.userId);
      setDeleted(r);
      reset();
      setConfirm(false);
    } catch (e) {
      setErr(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(null);
    }
  }

  async function togglePersonalization(on: boolean) {
    setBusy("personalization");
    setErr(null);
    try {
      await api.consent({ user_id: state.userId, purpose: "personalization", granted: on, created_at: new Date().toISOString() });
      update((s) => ({ consent: { ...s.consent, personalization: on } }));
    } catch (e) {
      setErr(e instanceof Error ? e : new Error(String(e)));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="stack">
      <section className="card" aria-labelledby="priv-title">
        <h2 id="priv-title">Privacy & your data</h2>
        <ul>
          <li>
            Pseudonymous demo id: <code>{state.userId}</code>
          </li>
          <li>Consent to demo storage: {state.consent.demo ? `given${state.consent.recordedAt ? ` (${state.consent.recordedAt.slice(0, 10)})` : ""}` : "not given"}</li>
          <li>Reflection storage: {state.consent.reflections ? "allowed" : "off"}</li>
          <li>Personalization: {state.consent.personalization ? "on" : "off"} (this demo does not personalize beyond the values you set)</li>
          <li>Your data is kept separately from synthetic simulation data and is never used to train or evaluate agents.</li>
        </ul>
        <div className="row">
          <button type="button" className="btn" onClick={() => void doExport()} disabled={busy !== null}>
            {busy === "export" ? "Exporting…" : "Export my data (JSON)"}
          </button>
          <label className="check" style={{ alignItems: "center" }}>
            <input type="checkbox" checked={state.consent.personalization} onChange={(e) => void togglePersonalization(e.target.checked)} disabled={busy !== null || !state.consent.demo} />
            <span>Allow personalization</span>
          </label>
        </div>
      </section>
      <section className="card" aria-labelledby="del-title" style={{ borderLeft: "5px solid var(--bad)" }}>
        <h2 id="del-title">Delete everything</h2>
        <p>Deletes your consent records, decisions and reflections on the server, and clears this browser's Future Self data. This cannot be undone.</p>
        {!confirm ? (
          <button type="button" className="btn btn--danger" onClick={() => setConfirm(true)} disabled={busy !== null}>
            Delete my data…
          </button>
        ) : (
          <div className="row">
            <button type="button" className="btn btn--danger" onClick={() => void doDelete()} disabled={busy !== null}>
              {busy === "delete" ? "Deleting…" : "Yes, permanently delete"}
            </button>
            <button type="button" className="btn btn--ghost" onClick={() => setConfirm(false)}>
              Cancel
            </button>
          </div>
        )}
        {deleted && (
          <div className="notice notice--info" role="status" style={{ marginTop: 12 }}>
            <p>
              Deleted for <code>{deleted.user_id}</code>: {Object.entries(deleted.deleted).map(([k, v]) => `${v} ${k}`).join(", ") || "no server records"}. Browser data cleared; you
              now have a new pseudonymous id.
            </p>
            <button type="button" className="btn btn--sm" onClick={() => navigate("/future/welcome")}>
              Start over
            </button>
          </div>
        )}
      </section>
      {err && <ErrorState error={err} title="Privacy request failed" />}
    </div>
  );
}
