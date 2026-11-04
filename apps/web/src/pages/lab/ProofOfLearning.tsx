import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import type { ExperimentInfo, OutcomeObservation } from "../../types/contracts";
import { Async, EmptyState, ErrorState, Loading } from "../../components/AsyncState";
import { LabBanner } from "../../components/Badges";
import { LineChart } from "../../components/charts/LineChart";
import type { LineSeries } from "../../components/charts/LineChart";
import { IntervalPlot } from "../../components/charts/IntervalPlot";
import { FAMILY_META, chanceFor, policyLabel } from "../../lib/families";
import { curvesFromOutcomes, metricsIn, perSeedValues, summarize } from "../../lib/stats";
import { ArmsTable, EffectsView, ExperimentHeader, ExperimentPicker, LimitationsPanel, RecordsTable, RunDownloads } from "./ExperimentParts";
import { armColor, isBaselineComparison } from "../../lib/experiments";

function chanceForMetric(metric: string): number | undefined {
  if (!/acc/i.test(metric)) return undefined;
  for (const f of Object.keys(FAMILY_META)) if (metric.includes(f)) return chanceFor(f);
  return undefined;
}

export default function ProofOfLearning() {
  const [params, setParams] = useSearchParams();
  const list = useApi((s) => api.listExperiments(s), []);
  const candidates = useMemo(() => (list.data ?? []).filter((x) => isBaselineComparison(x.definition)), [list.data]);
  const id = params.get("id");
  useEffect(() => {
    if (!id && candidates[0]) setParams({ id: candidates[0].experiment_id }, { replace: true });
  }, [id, candidates, setParams]);

  return (
    <div className="stack">
      <LabBanner />
      <header className="page-head">
        <div className="eyebrow">Research Lab · Proof of learning</div>
        <h1>Does an agent that develops from experience learn better?</h1>
        <p>
          Four agent classes see the same task stream with the same feedback and budget. Skill is measured only on held-out symbols with learning switched off. This page shows
          whatever the preregistered analysis found — including when the developmental learner does not win.
        </p>
        <p className="muted" style={{ fontSize: "0.9rem" }}>
          Individual skill-acquisition curves and the parameter changes behind them are on each agent's page — pick an agent in <Link to="/lab/society">Live society</Link>.
        </p>
      </header>
      {list.error && <ErrorState error={list.error} onRetry={list.reload} />}
      {list.loading && !list.data && <Loading label="Loading experiments…" />}
      {list.data && candidates.length === 0 && (
        <EmptyState title="No baseline-comparison experiments have been run yet">
          Run <code>make experiments</code> to execute the preregistered learning-lab experiments. Results appear here only after they actually run.
        </EmptyState>
      )}
      {candidates.length > 0 && (
        <section className="card">
          <ExperimentPicker experiments={candidates} selected={id} onSelect={(v) => setParams({ id: v })} label="Baseline-comparison experiment" />
        </section>
      )}
      {id && <ExperimentResults key={id} id={id} />}
    </div>
  );
}

function ExperimentResults({ id }: { id: string }) {
  const info = useApi((s) => api.getExperimentResults(id, s), [id]);
  return (
    <Async {...info} loadingLabel="Loading results…">
      {(d) => <ResultsBody info={d} />}
    </Async>
  );
}

function ResultsBody({ info }: { info: ExperimentInfo }) {
  const def = info.definition;
  const r = info.result;
  const outcomes = info.outcomes_sample ?? [];
  const curveOutcomes = outcomes.filter((o) => o.split !== "transfer" && (o.age_years !== null && o.age_years !== undefined || (o.tick !== null && o.tick !== undefined)));
  const curveMetrics = metricsIn(curveOutcomes);
  const [metric, setMetric] = useState<string>(() => curveMetrics.find((m) => m === def.analysis.primary_metric) ?? curveMetrics[0] ?? "");
  const transfer = outcomes.filter((o) => o.split === "transfer");

  return (
    <div className="stack">
      <ExperimentHeader info={info} />
      <section className="card" aria-labelledby="curves-title">
        <div className="card__header">
          <h2 id="curves-title">Learning curves on held-out tasks</h2>
          {curveMetrics.length > 1 && (
            <div className="field" style={{ minWidth: 240 }}>
              <label htmlFor="curve-metric" className="sr-only">
                Metric
              </label>
              <select id="curve-metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
                {curveMetrics.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        {metric ? <Curves outcomes={curveOutcomes} metric={metric} info={info} /> : <p className="muted">The analysis returned no time-indexed observations to draw curves from.</p>}
      </section>

      {r && (
        <section className="card" aria-labelledby="arms-title">
          <h2 id="arms-title">Arm summaries</h2>
          <ArmsTable arms={r.arms} def={def} />
        </section>
      )}

      {r && (
        <section className="card" aria-labelledby="effects-title">
          <h2 id="effects-title">Baseline comparison</h2>
          <EffectsView effects={r.effects} />
        </section>
      )}

      <section className="card" aria-labelledby="transfer-title">
        <h2 id="transfer-title">Transfer to an unseen task family</h2>
        <p className="muted">
          The <em>barter</em> family is never practised during development. In the orthogonal control world its rule uses a latent space unrelated to the trained families, so a
          shared representation should not help there.
        </p>
        <TransferView outcomes={transfer} info={info} />
        <WorldVariants info={info} />
      </section>

      {r && (
        <div className="grid grid-2">
          <section className="card" aria-labelledby="abl-title">
            <h2 id="abl-title">Ablations</h2>
            <p className="muted" style={{ fontSize: "0.86rem" }}>
              Removing memory, environmental feedback, or policy learning individually.
            </p>
            <RecordsTable rows={r.ablations ?? []} caption="Ablations" />
          </section>
          <section className="card" aria-labelledby="sens-title">
            <h2 id="sens-title">Sensitivity</h2>
            <RecordsTable rows={r.sensitivity ?? []} caption="Sensitivity analyses" />
          </section>
        </div>
      )}
      <LimitationsPanel result={r} def={def} />
      {r && <RunDownloads runIds={r.run_ids} />}
    </div>
  );
}

function Curves({ outcomes, metric, info }: { outcomes: OutcomeObservation[]; metric: string; info: ExperimentInfo }) {
  const useAge = outcomes.some((o) => o.metric === metric && o.age_years !== null && o.age_years !== undefined);
  const curves = useMemo(() => curvesFromOutcomes(outcomes, metric, useAge ? "age_years" : "tick"), [outcomes, metric, useAge]);
  const arms = [...curves.keys()].sort();
  if (!arms.length) return <p className="muted">No observations for {metric}.</p>;
  const series: LineSeries[] = arms.map((arm, i) => {
    const ad = info.definition.arms.find((a) => a.name === arm);
    return {
      id: arm,
      label: ad ? `${arm} (${policyLabel(ad.policy_class)})` : arm,
      color: armColor(info.definition, arm, i),
      points: (curves.get(arm) ?? []).map((p) => ({ x: p.x, y: p.y, lo: p.lo, hi: p.hi })),
    };
  });
  const chance = chanceForMetric(metric);
  const nSeeds = Math.max(...[...curves.values()].flatMap((pts) => pts.map((p) => p.n ?? 0)));
  return (
    <>
      <LineChart
        title={`${metric} by ${useAge ? "simulated age" : "tick"}`}
        series={series}
        xLabel={useAge ? "simulated age (years)" : "tick"}
        yLabel={metric}
        refLines={chance !== undefined ? [{ y: chance, label: `chance ${chance.toFixed(2)}` }] : []}
        height={300}
      />
      <p className="muted" style={{ fontSize: "0.82rem" }}>
        Lines: mean across seeds (seed is the unit of analysis, up to {nSeeds} per point). Shaded band: ±1.96 standard errors across seeds, computed in the browser from the raw
        observations returned by the API — a descriptive summary. The preregistered inference is the bootstrap analysis in the table below.
      </p>
    </>
  );
}

function TransferView({ outcomes, info }: { outcomes: OutcomeObservation[]; info: ExperimentInfo }) {
  const metrics = metricsIn(outcomes);
  if (!metrics.length) return <p className="muted">This experiment reported no transfer-split observations.</p>;
  return (
    <div className="grid grid-2">
      {metrics.map((m) => {
        const per = perSeedValues(outcomes, m);
        const rows = [...per.entries()]
          .sort((a, b) => a[0].localeCompare(b[0]))
          .map(([arm, vals], i) => {
            const s = summarize(vals.map((v) => v.value));
            return { id: arm, label: arm, estimate: s.mean, lo: s.lo, hi: s.hi, dots: vals.map((v) => v.value), color: armColor(info.definition, arm, i) };
          });
        const chance = chanceForMetric(m);
        return (
          <div key={m}>
            <h3 className="mono" style={{ fontSize: "0.92rem" }}>
              {m}
            </h3>
            <IntervalPlot rows={rows} title={`Transfer: ${m}`} xLabel={m} zeroLine={chance ?? null} />
          </div>
        );
      })}
    </div>
  );
}

function WorldVariants({ info }: { info: ExperimentInfo }) {
  const variants = info.definition.arms.filter((a) => Object.keys(a.world_overrides ?? {}).length > 0 || Object.keys(a.policy_params ?? {}).length > 0);
  if (!variants.length) return null;
  return (
    <details style={{ marginTop: 12 }}>
      <summary>Arm configuration differences ({variants.length})</summary>
      <ul>
        {variants.map((a) => (
          <li key={a.name}>
            <strong>{a.name}</strong> ({policyLabel(a.policy_class)}): <code>{JSON.stringify({ ...a.policy_params, ...a.world_overrides })}</code>
          </li>
        ))}
      </ul>
    </details>
  );
}
