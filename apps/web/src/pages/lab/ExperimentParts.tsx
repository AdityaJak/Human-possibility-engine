import type { ReactNode } from "react";
import { api } from "../../api/client";
import type { ArmSummary, EffectEstimate, ExperimentDefinition, ExperimentInfo, ExperimentResult } from "../../types/contracts";
import { IntervalPlot } from "../../components/charts/IntervalPlot";
import type { IntervalRow } from "../../components/charts/IntervalPlot";
import { JobStatusChip, ResultStatusChip } from "../../components/StatusChip";
import { SourceTag } from "../../components/Badges";
import { KeyValue } from "../../components/Stat";
import { policyLabel } from "../../lib/families";
import { fmtCI, fmtFixed, fmtInt, shortId, titleCase } from "../../lib/format";

export function ExperimentPicker({
  experiments,
  selected,
  onSelect,
  label,
}: {
  experiments: ExperimentInfo[];
  selected: string | null;
  onSelect: (id: string) => void;
  label: string;
}) {
  return (
    <div className="field">
      <label htmlFor="exp-select">{label}</label>
      <select id="exp-select" value={selected ?? ""} onChange={(e) => onSelect(e.target.value)}>
        {experiments.map((x) => (
          <option key={x.experiment_id} value={x.experiment_id}>
            {x.definition.title} · {x.result ? x.result.status : x.status} · {x.source === "stored_real_output" ? "stored real output" : "live"}
          </option>
        ))}
      </select>
    </div>
  );
}

export function ExperimentHeader({ info }: { info: ExperimentInfo }) {
  const d = info.definition;
  const r = info.result;
  const negative = r && r.status !== "passed";
  const tone = !r ? "" : r.status === "failed" || r.status === "error" ? "negative-result" : r.status === "inconclusive" ? "inconclusive-result" : "";
  return (
    <section className={`card ${tone}`} aria-labelledby="exp-title">
      <div className="row" style={{ marginBottom: 8 }}>
        {r ? <ResultStatusChip status={r.status} /> : <JobStatusChip status={info.status} />}
        <SourceTag source={info.source} />
        <span className="chip chip--neutral">{d.seeds.length} seeds</span>
        <span className="chip chip--neutral">{d.arms.length} arms</span>
        {d.preregistration_hash && (
          <span className="chip chip--neutral mono" title="Stable hash of the definition, frozen before any run">
            prereg {shortId(d.preregistration_hash, 12)}
          </span>
        )}
      </div>
      <h2 id="exp-title">{d.title}</h2>
      {negative && (
        <p className="result-pill" style={{ color: r.status === "inconclusive" ? "var(--warn)" : "var(--bad)" }}>
          {r.status === "failed" ? "Negative result — the preregistered test did not pass." : r.status === "inconclusive" ? "Inconclusive — uncertainty too large to decide." : "Analysis error."}
        </p>
      )}
      {r && <p>{r.status_reason}</p>}
      <KeyValue
        items={[
          ["Hypothesis", d.hypothesis],
          ["Falsified if", d.falsification_criterion],
          ["Primary metric", `${d.analysis.primary_metric} — ${d.analysis.primary_metric_definition}`],
          ["Comparison", `${d.analysis.comparison} (${d.analysis.test ?? "paired_bootstrap"}, α = ${d.analysis.alpha ?? 0.05})`],
          ["Minimum effect of interest", d.analysis.minimum_effect_of_interest !== undefined && d.analysis.minimum_effect_of_interest !== null ? fmtFixed(d.analysis.minimum_effect_of_interest, 3) : "not specified"],
          ["Control arm", d.control_arm],
          ["Seeds", d.seeds.join(", ")],
          ["Population", `${d.base_config.n_agents ?? "?"} agents · ${d.base_config.years ?? "?"} years`],
        ]}
      />
    </section>
  );
}

export function ArmsTable({ arms, def }: { arms: ArmSummary[]; def: ExperimentDefinition }) {
  const metrics = [...new Set(arms.flatMap((a) => Object.keys(a.metrics ?? {})))].sort();
  if (!metrics.length) return <p className="muted">No arm summaries reported.</p>;
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">Arm summaries</caption>
        <thead>
          <tr>
            <th>Metric</th>
            {arms.map((a) => {
              const ad = def.arms.find((x) => x.name === a.arm);
              return (
                <th key={a.arm}>
                  {a.arm}
                  {ad && <div style={{ textTransform: "none", fontWeight: 500 }}>{policyLabel(ad.policy_class)}</div>}
                  <div style={{ textTransform: "none", fontWeight: 500 }}>n = {a.n_runs} runs</div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m}>
              <td className="mono">{m}</td>
              {arms.map((a) => {
                const v = a.metrics?.[m];
                return (
                  <td key={a.arm} className="num">
                    {v ? (
                      <>
                        {fmtFixed(v.mean, 3)}
                        <div className="muted" style={{ fontSize: "0.78rem" }}>
                          {fmtCI(v.ci_low, v.ci_high, 3)} {v.sd !== undefined ? `sd ${fmtFixed(v.sd, 3)}` : ""}
                        </div>
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EffectsView({ effects, title = "Preregistered contrasts (from the analysis service)" }: { effects: EffectEstimate[]; title?: string }) {
  if (!effects.length) return <p className="muted">No effect estimates reported.</p>;
  const rows: IntervalRow[] = effects.map((e, i) => ({
    id: `${e.metric}-${e.contrast}-${i}`,
    label: `${e.contrast} · ${e.metric}`,
    estimate: e.estimate,
    lo: e.ci_low,
    hi: e.ci_high,
    color: e.ci_low > 0 ? "var(--ok)" : e.ci_high < 0 ? "var(--bad)" : "var(--warn)",
  }));
  return (
    <div className="stack">
      <IntervalPlot rows={rows} title={title} xLabel="estimate (interval excludes 0 → coloured green/red)" zeroLine={0} />
      <div className="table-wrap">
        <table>
          <caption className="sr-only">{title}</caption>
          <thead>
            <tr>
              <th>Contrast</th>
              <th>Metric</th>
              <th className="num">Estimate</th>
              <th>Interval</th>
              <th className="num">p</th>
              <th className="num">Std. effect</th>
              <th className="num">Units</th>
              <th>Method</th>
              <th>Min. effect met?</th>
            </tr>
          </thead>
          <tbody>
            {effects.map((e, i) => (
              <tr key={i}>
                <td>{e.contrast}</td>
                <td className="mono">{e.metric}</td>
                <td className="num">{fmtFixed(e.estimate, 4)}</td>
                <td>
                  {fmtCI(e.ci_low, e.ci_high, 4)} <span className="muted">({Math.round((e.ci_level ?? 0.95) * 100)}%)</span>
                </td>
                <td className="num">{e.p_value === null || e.p_value === undefined ? "—" : e.p_value < 0.001 ? "<0.001" : fmtFixed(e.p_value, 3)}</td>
                <td className="num">{fmtFixed(e.standardized_effect, 2)}</td>
                <td className="num">
                  {fmtInt(e.n_units)} {e.unit_of_analysis ?? "seed"}s
                </td>
                <td>{e.method}</td>
                <td>{e.passed_minimum_effect === null || e.passed_minimum_effect === undefined ? "—" : e.passed_minimum_effect ? <span className="chip chip--ok">yes</span> : <span className="chip chip--bad">no</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function cell(v: unknown): ReactNode {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return fmtFixed(v, 4);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v;
  return <code style={{ fontSize: "0.75rem" }}>{JSON.stringify(v)}</code>;
}

/** Render a list of free-form records (ablations, sensitivity analyses) as a table. */
export function RecordsTable({ rows, caption }: { rows: Array<Record<string, unknown>>; caption: string }) {
  if (!rows.length) return <p className="muted">None reported.</p>;
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{titleCase(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c} className={typeof r[c] === "number" ? "num" : undefined}>
                  {cell(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function LimitationsPanel({ result, def }: { result: ExperimentResult | null | undefined; def: ExperimentDefinition }) {
  return (
    <section className="card" aria-labelledby="validity-title" style={{ borderLeft: "5px solid var(--warn)" }}>
      <h2 id="validity-title">Assumptions and validity</h2>
      <p className="notice">
        {result?.synthetic_disclaimer ??
          "Simulated result from a synthetic world. Effects may simply reflect assumptions built into the simulator and are not estimates of real-world causal effects."}
      </p>
      <div className="grid grid-2" style={{ marginTop: 12 }}>
        <div>
          <h3>Validity notes (preregistered)</h3>
          {def.validity_notes && def.validity_notes.length ? (
            <ul>
              {def.validity_notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">None recorded.</p>
          )}
        </div>
        <div>
          <h3>Limitations (reported with results)</h3>
          {result?.limitations && result.limitations.length ? (
            <ul>
              {result.limitations.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">None recorded.</p>
          )}
        </div>
      </div>
      <h3>What this does not show</h3>
      <ul>
        <li>It is not evidence about real people. The simulator's own ground truth is not independent human validation.</li>
        <li>It is not a real-world causal effect estimate; the mechanisms and parameters are modelling assumptions.</li>
        <li>Agent counts are not a success metric.</li>
      </ul>
      {result && (
        <p className="muted" style={{ fontSize: "0.84rem" }}>
          Regenerated from: {result.generated_from ?? "raw event logs"} · {result.run_ids.length} runs · created {result.created_at ?? "—"}
        </p>
      )}
    </section>
  );
}

export function RunDownloads({ runIds }: { runIds: string[] }) {
  if (!runIds.length) return null;
  return (
    <details>
      <summary>Download raw run logs ({runIds.length})</summary>
      <ul className="mono" style={{ fontSize: "0.8rem", columns: 2 }}>
        {runIds.map((r) => (
          <li key={r}>
            <a href={api.downloadUrl(r)}>{r}</a>
          </li>
        ))}
      </ul>
    </details>
  );
}
