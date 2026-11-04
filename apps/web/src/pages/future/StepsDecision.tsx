import { useState } from "react";
import type { FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import type { Calculation, DecisionOption, DecisionRequest } from "../../types/contracts";
import { ErrorState } from "../../components/AsyncState";
import { EvidenceChip, EvidenceLegend, EvidenceTable } from "../../components/Evidence";
import { IllustrativeBadge } from "../../components/Badges";
import { GuardedText, ScenarioCard } from "../../components/ScenarioCard";
import { SafetyNotice } from "../../components/SafetyNotice";
import { evidenceKindOf } from "../../lib/evidence";
import { fmtNum, titleCase } from "../../lib/format";
import { useFuture } from "./context";
import { emptyDraft } from "./model";
import type { DecisionDraft } from "./model";

const KINDS: Array<{ value: NonNullable<DecisionOption["kind"]>; label: string }> = [
  { value: "current_path", label: "Stay on current path" },
  { value: "new_job", label: "New job" },
  { value: "learn_skill", label: "Learn a skill" },
  { value: "start_project", label: "Start a project / company" },
  { value: "education", label: "Education" },
  { value: "other", label: "Other" },
];

const INPUTS: Array<{ key: string; label: string; hint: string }> = [
  { key: "monthly_income", label: "Monthly income", hint: "after tax, your currency" },
  { key: "monthly_expenses", label: "Monthly expenses", hint: "your currency" },
  { key: "savings", label: "Savings available", hint: "your currency" },
  { key: "upfront_cost", label: "Up-front cost", hint: "e.g. course fee" },
  { key: "hours_per_week", label: "Hours per week", hint: "time commitment" },
];

const EXAMPLE: DecisionDraft = {
  question: "Should I take a higher-paying job, continue in my current role, or attempt a startup?",
  options: [
    { option_id: "opt-1", label: "Continue in my current role", kind: "current_path", description: "", inputs: {} },
    { option_id: "opt-2", label: "Take the higher-paying job", kind: "new_job", description: "", inputs: {} },
    { option_id: "opt-3", label: "Attempt a startup", kind: "start_project", description: "", inputs: {} },
  ],
  constraints: {},
  horizonYears: 3,
};

export function DecisionStudio() {
  const { state, update } = useFuture();
  const navigate = useNavigate();
  const draft = state.draft;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error | null>(null);

  const setDraft = (d: Partial<DecisionDraft>) => update((s) => ({ draft: { ...s.draft, ...d } }));
  const setOption = (i: number, o: Partial<DecisionOption>) => setDraft({ options: draft.options.map((x, k) => (k === i ? { ...x, ...o } : x)) });

  function setInput(i: number, key: string, raw: string) {
    const opt = draft.options[i];
    if (!opt) return;
    const inputs = { ...(opt.inputs ?? {}) };
    if (raw === "" || Number.isNaN(Number(raw))) delete inputs[key];
    else inputs[key] = Number(raw);
    setOption(i, { inputs });
  }

  function addOption() {
    if (draft.options.length >= 4) return;
    const n = draft.options.length + 1;
    setDraft({ options: [...draft.options, { option_id: `opt-${n}-${Date.now() % 10000}`, label: "", kind: "other", description: "", inputs: {} }] });
  }

  function removeOption(i: number) {
    if (draft.options.length <= 2) return;
    setDraft({ options: draft.options.filter((_, k) => k !== i) });
  }

  const valid = draft.question.trim().length > 3 && draft.options.length >= 2 && draft.options.every((o) => o.label.trim().length > 0);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!valid) return;
    setBusy(true);
    setErr(null);
    const req: DecisionRequest = {
      user_id: state.userId,
      consent_given: state.consent.demo,
      question: draft.question.trim(),
      options: draft.options.map((o) => ({ ...o, label: o.label.trim() })),
      values: state.values,
      constraints: draft.constraints,
      horizon_years: draft.horizonYears,
    };
    try {
      const res = await api.scenarios(req);
      update({ response: res, plan: [] });
      navigate("/future/futures");
    } catch (ex) {
      setErr(ex instanceof Error ? ex : new Error(String(ex)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="stack" onSubmit={submit} aria-labelledby="ds-title">
      <section className="card">
        <div className="card__header">
          <h2 id="ds-title">Decision Studio</h2>
          <div className="row">
            <button type="button" className="btn btn--sm" onClick={() => update({ draft: { ...EXAMPLE, options: EXAMPLE.options.map((o) => ({ ...o })) } })}>
              Use the job / startup example
            </button>
            <button type="button" className="btn btn--sm btn--ghost" onClick={() => update({ draft: emptyDraft() })}>
              Clear
            </button>
          </div>
        </div>
        <div className="field">
          <label htmlFor="q">What decision are you weighing?</label>
          <textarea id="q" value={draft.question} onChange={(e) => setDraft({ question: e.target.value })} placeholder="e.g. Should I learn data analysis or ask for a new role at work?" />
        </div>
        <div className="grid grid-2" style={{ marginTop: 12 }}>
          <div className="field">
            <label htmlFor="horizon">Horizon to think about (years)</label>
            <input id="horizon" type="number" min={0.5} max={10} step={0.5} value={draft.horizonYears} onChange={(e) => setDraft({ horizonYears: Number(e.target.value) || 1 })} />
          </div>
          <div className="field">
            <label htmlFor="c-runway">
              Minimum savings runway you'd accept <small>(months, optional)</small>
            </label>
            <input
              id="c-runway"
              type="number"
              min={0}
              value={typeof draft.constraints.min_runway_months === "number" ? draft.constraints.min_runway_months : ""}
              onChange={(e) => {
                const c = { ...draft.constraints };
                if (e.target.value === "") delete c.min_runway_months;
                else c.min_runway_months = Number(e.target.value);
                setDraft({ constraints: c });
              }}
            />
          </div>
          <div className="field">
            <label htmlFor="c-hours">
              Maximum hours per week you can commit <small>(optional)</small>
            </label>
            <input
              id="c-hours"
              type="number"
              min={0}
              max={100}
              value={typeof draft.constraints.max_hours_per_week === "number" ? draft.constraints.max_hours_per_week : ""}
              onChange={(e) => {
                const c = { ...draft.constraints };
                if (e.target.value === "") delete c.max_hours_per_week;
                else c.max_hours_per_week = Number(e.target.value);
                setDraft({ constraints: c });
              }}
            />
          </div>
          <div className="field">
            <label htmlFor="c-notes">
              Other constraints <small>(optional, no sensitive details)</small>
            </label>
            <input
              id="c-notes"
              type="text"
              value={typeof draft.constraints.notes === "string" ? draft.constraints.notes : ""}
              onChange={(e) => {
                const c = { ...draft.constraints };
                if (e.target.value === "") delete c.notes;
                else c.notes = e.target.value;
                setDraft({ constraints: c });
              }}
            />
          </div>
        </div>
      </section>

      <section aria-label="Options" className="grid grid-2">
        {draft.options.map((o, i) => (
          <fieldset key={o.option_id} className="card" style={{ margin: 0 }}>
            <legend className="sr-only">Option {i + 1}</legend>
            <div className="card__header">
              <h3>Option {i + 1}</h3>
              {draft.options.length > 2 && (
                <button type="button" className="btn btn--sm btn--ghost" onClick={() => removeOption(i)} aria-label={`Remove option ${i + 1}`}>
                  Remove
                </button>
              )}
            </div>
            <div className="stack" style={{ gap: 10 }}>
              <div className="field">
                <label htmlFor={`o-${i}-label`}>Name</label>
                <input id={`o-${i}-label`} type="text" value={o.label} onChange={(e) => setOption(i, { label: e.target.value })} required />
              </div>
              <div className="field">
                <label htmlFor={`o-${i}-kind`}>Kind</label>
                <select id={`o-${i}-kind`} value={o.kind ?? "other"} onChange={(e) => setOption(i, { kind: e.target.value as DecisionOption["kind"] })}>
                  {KINDS.map((k) => (
                    <option key={k.value} value={k.value}>
                      {k.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor={`o-${i}-desc`}>
                  Notes <small>(optional)</small>
                </label>
                <input id={`o-${i}-desc`} type="text" value={o.description ?? ""} onChange={(e) => setOption(i, { description: e.target.value })} />
              </div>
              <details>
                <summary>Optional numbers for calculations</summary>
                <p className="muted" style={{ fontSize: "0.84rem" }}>
                  Only used for transparent arithmetic such as cash runway. Leave blank if you prefer.
                </p>
                <div className="grid grid-2">
                  {INPUTS.map((inp) => (
                    <div className="field" key={inp.key}>
                      <label htmlFor={`o-${i}-${inp.key}`}>
                        {inp.label} <small>({inp.hint})</small>
                      </label>
                      <input
                        id={`o-${i}-${inp.key}`}
                        type="number"
                        min={0}
                        value={o.inputs?.[inp.key] ?? ""}
                        onChange={(e) => setInput(i, inp.key, e.target.value)}
                      />
                    </div>
                  ))}
                </div>
              </details>
            </div>
          </fieldset>
        ))}
      </section>

      <div className="row">
        <button type="button" className="btn" onClick={addOption} disabled={draft.options.length >= 4}>
          Add option ({draft.options.length}/4)
        </button>
        <span className="spacer" />
        <button type="submit" className="btn btn--future" disabled={!valid || busy}>
          {busy ? "Building scenarios…" : "Explore possible futures"}
        </button>
      </div>
      {!valid && <small>Write a question and name at least two options.</small>}
      {err && <ErrorState error={err} title="Scenarios could not be generated" />}
    </form>
  );
}

function valueLabels(values: Array<{ key: string; label: string }>): Record<string, string> {
  return Object.fromEntries(values.map((v) => [v.key, v.label]));
}

export function PossibleFutures() {
  const { state } = useFuture();
  const res = state.response;
  if (!res) return null;
  const crisis = res.safety?.category === "crisis";
  const labels = valueLabels(state.values);
  const optionLabel = (id: string) => state.draft.options.find((o) => o.option_id === id)?.label ?? id;
  return (
    <div className="stack">
      <SafetyNotice notice={res.safety} />
      {crisis ? (
        <p className="muted">Scenarios are not shown for this question. Please reach out to the support above.</p>
      ) : (
        <>
          <section className="card card--flat">
            <h2>Possible futures for: “{state.draft.question}”</h2>
            <p>{res.value_alignment_note}</p>
            <p className="muted" style={{ fontSize: "0.86rem" }}>
              {res.simulation_separation_note ?? "No result from the synthetic agent society is used as evidence about your life."}
            </p>
          </section>
          <div className="scenario-grid">
            {res.scenarios.map((s) => (
              <div key={s.scenario_id} className="stack" style={{ gap: 6 }}>
                <div className="eyebrow">{optionLabel(s.option_id)}</div>
                <ScenarioCard scenario={s} valueLabels={labels} />
              </div>
            ))}
          </div>
          <div className="row">
            <Link to="/future/evidence" className="btn btn--future">
              Inspect evidence &amp; uncertainty
            </Link>
            <Link to="/future/north-star" className="btn">
              Change my values
            </Link>
            <Link to="/future/decision" className="btn">
              Edit the decision
            </Link>
          </div>
        </>
      )}
    </div>
  );
}

function CalculationRow({ c }: { c: Calculation }) {
  return (
    <tr>
      <td>
        <strong>{c.name}</strong>
        {c.note && <div className="muted" style={{ fontSize: "0.8rem" }}>{c.note}</div>}
      </td>
      <td className="mono" style={{ fontSize: "0.82rem" }}>
        {c.formula}
      </td>
      <td style={{ fontSize: "0.84rem" }}>
        {Object.entries(c.inputs).map(([k, v]) => (
          <div key={k}>
            {titleCase(k)} = {fmtNum(v)}
          </div>
        ))}
      </td>
      <td className="num">
        {c.result === null ? <span className="ev ev--insufficient_evidence">not computable</span> : `${fmtNum(c.result)} ${c.unit}`}
      </td>
      <td>
        <a href={`#evidence-${c.evidence_id}`} className="mono" style={{ fontSize: "0.75rem" }}>
          {c.evidence_id}
        </a>
      </td>
    </tr>
  );
}

export function EvidenceUncertainty() {
  const { state } = useFuture();
  const res = state.response;
  if (!res) return null;
  if (res.safety?.category === "crisis") return <SafetyNotice notice={res.safety} />;
  const optionLabel = (id: string) => state.draft.options.find((o) => o.option_id === id)?.label ?? id;
  return (
    <div className="stack">
      <div className="split">
        <div className="stack">
          {res.scenarios.map((s) => (
            <section key={s.scenario_id} className="card" aria-labelledby={`eu-${s.scenario_id}`}>
              <div className="row">
                <IllustrativeBadge compact />
                <h2 id={`eu-${s.scenario_id}`} style={{ margin: 0, fontSize: "1.15rem" }}>
                  {optionLabel(s.option_id)} — {s.title}
                </h2>
              </div>
              <h3 style={{ marginTop: 14 }}>Assumptions this scenario is conditioned on</h3>
              {s.assumptions && s.assumptions.length ? (
                <ul style={{ listStyle: "none", padding: 0, display: "grid", gap: 6 }}>
                  {s.assumptions.map((a) => (
                    <li key={a.evidence_id} className="row" style={{ alignItems: "baseline", flexWrap: "nowrap" }}>
                      <EvidenceChip kind={evidenceKindOf(a.source_type, a.validation_status)} />
                      <span>{a.claim}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No explicit assumptions listed.</p>
              )}
              <h3>Calculations from your inputs</h3>
              {s.calculations && s.calculations.length ? (
                <div className="table-wrap" tabIndex={0}>
                  <table>
                    <thead>
                      <tr>
                        <th>Calculation</th>
                        <th>Formula</th>
                        <th>Inputs</th>
                        <th className="num">Result</th>
                        <th>Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s.calculations.map((c) => (
                        <CalculationRow key={c.name} c={c} />
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted">No calculations — add optional numbers in the Decision Studio to see transparent arithmetic.</p>
              )}
              <div className="grid grid-2" style={{ marginTop: 12 }}>
                <div>
                  <h3>Key uncertainties</h3>
                  {s.uncertainties && s.uncertainties.length ? (
                    <ul>
                      {s.uncertainties.map((u) => (
                        <li key={u}>
                          <GuardedText text={u} as="span" />
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">None listed.</p>
                  )}
                </div>
                <div>
                  <h3>
                    <span className="ev ev--insufficient_evidence">Insufficient evidence to predict</span>
                  </h3>
                  {s.cannot_predict && s.cannot_predict.length ? (
                    <ul>
                      {s.cannot_predict.map((u) => (
                        <li key={u}>{u}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">None listed.</p>
                  )}
                </div>
              </div>
            </section>
          ))}
        </div>
        <div className="stack">
          <EvidenceLegend />
          <section className="notice notice--info">
            <p>
              <strong>No fake percentages.</strong> Where a number would require a calibrated model of your life that does not exist, we show “insufficient evidence”
              instead of a probability.
            </p>
          </section>
        </div>
      </div>
      <section className="card" aria-labelledby="ledger-title">
        <h2 id="ledger-title">Evidence ledger</h2>
        <p className="muted">Every quantitative statement points to one of these records.</p>
        <EvidenceTable records={res.evidence} />
      </section>
      <div className="row">
        <Link to="/future/plan" className="btn btn--future">
          Plan a reversible experiment
        </Link>
      </div>
    </div>
  );
}
