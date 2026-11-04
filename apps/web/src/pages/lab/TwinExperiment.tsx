import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import type { ExperimentInfo, Intervention } from "../../types/contracts";
import { Async, EmptyState, ErrorState, Loading } from "../../components/AsyncState";
import { LabBanner } from "../../components/Badges";
import { IntervalPlot } from "../../components/charts/IntervalPlot";
import type { IntervalRow } from "../../components/charts/IntervalPlot";
import { KeyValue } from "../../components/Stat";
import { policyLabel } from "../../lib/families";
import { titleCase } from "../../lib/format";
import { metricsIn, perSeedValues, summarize } from "../../lib/stats";
import { ArmsTable, EffectsView, ExperimentHeader, ExperimentPicker, LimitationsPanel, RecordsTable, RunDownloads } from "./ExperimentParts";
import { armColor, isIntervention } from "../../lib/experiments";

export default function TwinExperiment() {
  const [params, setParams] = useSearchParams();
  const list = useApi((s) => api.listExperiments(s), []);
  const candidates = useMemo(() => (list.data ?? []).filter((x) => isIntervention(x.definition)), [list.data]);
  const id = params.get("id");
  useEffect(() => {
    if (!id && candidates[0]) {
      const pick = candidates.find((c) => c.definition.arms.some((a) => a.interventions?.some((i) => i.kind === "tutoring_access"))) ?? candidates[0];
      setParams({ id: pick.experiment_id }, { replace: true });
    }
  }, [id, candidates, setParams]);

  return (
    <div className="stack">
      <LabBanner />
      <header className="page-head">
        <div className="eyebrow">Research Lab · Twin-society experiment</div>
        <h1>One population, cloned. One intervention, randomized.</h1>
        <p>
          Control and treatment societies start from the same initial population and share random streams (common random numbers). Only the intervention differs. Each seed is an
          independent replication; outcomes are measured on held-out tasks.
        </p>
      </header>
      {list.error && <ErrorState error={list.error} onRetry={list.reload} />}
      {list.loading && !list.data && <Loading label="Loading experiments…" />}
      {list.data && candidates.length === 0 && (
        <EmptyState title="No intervention experiments have been run yet">
          Run <code>make experiments</code>. Results are shown only after they have actually been executed.
        </EmptyState>
      )}
      {candidates.length > 0 && (
        <section className="card">
          <ExperimentPicker experiments={candidates} selected={id} onSelect={(v) => setParams({ id: v })} label="Intervention experiment" />
        </section>
      )}
      {id && <Body key={id} id={id} />}
    </div>
  );
}

function Body({ id }: { id: string }) {
  const info = useApi((s) => api.getExperimentResults(id, s), [id]);
  return (
    <Async {...info} loadingLabel="Loading experiment…">
      {(d) => <TwinBody info={d} />}
    </Async>
  );
}

function InterventionList({ items }: { items: Intervention[] }) {
  if (!items.length) return <span className="muted">none (control conditions)</span>;
  return (
    <ul style={{ margin: 0, paddingLeft: 18 }}>
      {items.map((iv) => (
        <li key={iv.name}>
          <strong>{titleCase(iv.kind)}</strong> “{iv.name}” — ages {iv.target?.age_min ?? 0}–{iv.target?.age_max ?? "any"}, {Math.round((iv.target?.fraction ?? 1) * 100)}% of eligible
          agents (seeded assignment), years {iv.start_year ?? 0}–{iv.end_year ?? "end"}
          {iv.params && Object.keys(iv.params).length > 0 && (
            <>
              {" "}
              · <code>{JSON.stringify(iv.params)}</code>
            </>
          )}
        </li>
      ))}
    </ul>
  );
}

function TwinBody({ info }: { info: ExperimentInfo }) {
  const def = info.definition;
  const r = info.result;
  const outcomes = useMemo(() => info.outcomes_sample ?? [], [info.outcomes_sample]);
  const primary = def.analysis.primary_metric;
  const metrics = metricsIn(outcomes);
  const metric = metrics.includes(primary) ? primary : metrics[0];

  const { armRows, diffRows, paired } = useMemo(() => {
    if (!metric) return { armRows: [] as IntervalRow[], diffRows: [] as IntervalRow[], paired: 0 };
    const per = perSeedValues(outcomes, metric);
    const arms = [...per.keys()].sort((a, b) => (a === def.control_arm ? -1 : b === def.control_arm ? 1 : a.localeCompare(b)));
    const armRows: IntervalRow[] = arms.map((arm, i) => {
      const vals = per.get(arm) ?? [];
      const s = summarize(vals.map((v) => v.value));
      return { id: arm, label: `${arm} (n=${s.n})`, estimate: s.mean, lo: s.lo, hi: s.hi, dots: vals.map((v) => v.value), color: armColor(def, arm, i) };
    });
    const control = new Map((per.get(def.control_arm) ?? []).map((v) => [v.seed, v.value]));
    let paired = 0;
    const diffRows: IntervalRow[] = arms
      .filter((a) => a !== def.control_arm)
      .map((arm, i) => {
        const diffs = (per.get(arm) ?? []).filter((v) => control.has(v.seed)).map((v) => v.value - (control.get(v.seed) as number));
        paired = Math.max(paired, diffs.length);
        const s = summarize(diffs);
        return { id: arm, label: `${arm} − ${def.control_arm}`, estimate: s.mean, lo: s.lo, hi: s.hi, dots: diffs, color: armColor(def, arm, i + 1) };
      });
    return { armRows, diffRows, paired };
  }, [outcomes, metric, def]);

  const mechanism = (r?.ablations ?? []).concat(r?.sensitivity ?? []).filter((x) => JSON.stringify(x).toLowerCase().includes("mechanism"));

  return (
    <div className="stack">
      <ExperimentHeader info={info} />

      <section className="card" aria-labelledby="design-title">
        <h2 id="design-title">Design</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Arm</th>
                <th>Agent class</th>
                <th>Interventions</th>
                <th>World overrides</th>
              </tr>
            </thead>
            <tbody>
              {def.arms.map((a) => (
                <tr key={a.name}>
                  <td>
                    <strong>{a.name}</strong> {a.name === def.control_arm && <span className="chip chip--neutral">control</span>}
                  </td>
                  <td>{policyLabel(a.policy_class)}</td>
                  <td>
                    <InterventionList items={a.interventions ?? []} />
                  </td>
                  <td>{a.world_overrides && Object.keys(a.world_overrides).length ? <code>{JSON.stringify(a.world_overrides)}</code> : <span className="muted">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 12 }}>
          <KeyValue
            items={[
              ["Matched initial state", `population seed ${def.base_config.population_seed ?? "= run seed"} shared by all arms of each replication`],
              ["Replications", `${def.seeds.length} seeds (${def.seeds.join(", ")})`],
              ["Population", `${def.base_config.n_agents ?? "?"} initial agents, ${def.base_config.years ?? "?"} simulated years`],
            ]}
          />
        </div>
      </section>

      <section className="card" aria-labelledby="outcomes-title">
        <h2 id="outcomes-title">Outcomes per seed{metric ? ` · ${metric}` : ""}</h2>
        {!metric ? (
          <p className="muted">No per-seed outcome observations were returned.</p>
        ) : (
          <div className="grid grid-2">
            <div>
              <h3>By arm</h3>
              <IntervalPlot rows={armRows} title={`${metric} by arm, one dot per seed`} xLabel={metric} zeroLine={null} />
            </div>
            <div>
              <h3>Paired difference (treatment − control, same seed)</h3>
              {diffRows.length ? (
                <IntervalPlot rows={diffRows} title={`Paired difference in ${metric}`} xLabel={`difference in ${metric}`} zeroLine={0} />
              ) : (
                <p className="muted">No treatment arm with matching seeds.</p>
              )}
              <p className="muted" style={{ fontSize: "0.82rem" }}>
                {paired} paired seeds. Dots are per-seed values; intervals here are ±1.96 SE computed in the browser (descriptive). The authoritative estimate is the
                preregistered analysis below.
              </p>
            </div>
          </div>
        )}
      </section>

      {r && (
        <section className="card" aria-labelledby="eff-title">
          <h2 id="eff-title">Effect estimates</h2>
          <EffectsView effects={r.effects} />
          <h3 style={{ marginTop: 16 }}>Arm summaries</h3>
          <ArmsTable arms={r.arms} def={def} />
        </section>
      )}

      {r && (
        <div className="grid grid-2">
          <section className="card" aria-labelledby="mech-title">
            <h2 id="mech-title">Mechanism check</h2>
            <p className="muted" style={{ fontSize: "0.88rem" }}>
              If the intervention only changes learning inputs, a non-learning agent class should show no skill effect. A non-zero effect there would mean the outcome is
              hard-coded rather than produced by learning.
            </p>
            {mechanism.length ? <RecordsTable rows={mechanism} caption="Mechanism checks" /> : <p className="muted">No mechanism-check entries reported for this experiment.</p>}
          </section>
          <section className="card" aria-labelledby="sens-title">
            <h2 id="sens-title">Sensitivity checks</h2>
            <RecordsTable rows={(r.sensitivity ?? []).filter((x) => !mechanism.includes(x))} caption="Sensitivity" />
          </section>
        </div>
      )}

      {r && r.ablations && r.ablations.filter((x) => !mechanism.includes(x)).length > 0 && (
        <section className="card">
          <h2>Ablations</h2>
          <RecordsTable rows={r.ablations.filter((x) => !mechanism.includes(x))} caption="Ablations" />
        </section>
      )}

      <LimitationsPanel result={r} def={def} />
      {r && (
        <section className="card card--flat">
          <h2 style={{ fontSize: "1.05rem" }}>Reproduce</h2>
          <p>
            Every run is reproducible from its seed and code version. Preregistration hash <code>{def.preregistration_hash ?? r.preregistration_hash}</code>. Raw logs:
          </p>
          <RunDownloads runIds={r.run_ids} />
          {r.figures && r.figures.length > 0 && (
            <p className="muted" style={{ fontSize: "0.84rem" }}>
              Figures regenerated from raw logs: {r.figures.join(", ")} (in <code>reports/experiments/{def.experiment_id}/</code>).
            </p>
          )}
        </section>
      )}
    </div>
  );
}
