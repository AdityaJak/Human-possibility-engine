import { useMemo, useState } from "react";
import { api } from "../../api/client";
import type { Phase3Experiment, Phase3Family, Phase3Model } from "../../api/client";
import { useApi } from "../../api/hooks";
import { Async, EmptyState } from "../../components/AsyncState";
import { IntervalPlot } from "../../components/charts/IntervalPlot";
import type { IntervalRow } from "../../components/charts/IntervalPlot";
import { LineChart } from "../../components/charts/LineChart";
import type { LineSeries } from "../../components/charts/LineChart";
import { fmtInt, fmtNum } from "../../lib/format";

/** Gates as fixed in docs/phase3/PHASE3_RESEARCH_THESIS.md §6 (amended §11–§12). Status comes only from the verdict file. */
const GATES: Array<{ id: string; name: string; rule: string }> = [
  { id: "A", name: "Learned capability", rule: "Real-data training beats outcome-permuted training; own-history adaptation beats a length-matched donor's" },
  { id: "B", name: "Individual adaptation", rule: "Beats a separately trained no-adaptation twin (≥ 0.01) and the best conventional adapter (≥ 0.005)" },
  { id: "C", name: "Human behavioural prediction", rule: "≥ 0.02 nats/choice better than the strongest tune-selected comparator in ≥ 2 counted families incl. the primary" },
  { id: "D", name: "Cross-domain transfer", rule: "Leave-one-family-out at k = 16: beats specialist, pretrained GRU and hierarchical Bayesian model by ≥ 0.02" },
  { id: "E", name: "Continual learning", rule: "Synthetic switching suite: savings beat fine-tuning, EWC and replay without slower adaptation" },
  { id: "F", name: "Affective / social contribution", rule: "Beats a matched-complexity control on held-out people where affect or social variables are observed" },
  { id: "G", name: "Causal decision support", rule: "Improves real outcomes in an ethically reviewed prospective randomised study" },
];

const ROLE_COLOR: Record<string, string> = {
  candidate: "var(--future)",
  comparator: "var(--lab)",
  reference: "#8a8f98",
};

function HumanDataBanner({ text }: { text: string }) {
  return (
    <div className="zone-banner zone-banner--lab" role="note" aria-label="Phase III data disclosure">
      <span className="badge badge--lab">Real human choice data — research, not predictions about you</span>
      <p>{text}</p>
    </div>
  );
}

function roleOf(name: string, m: Phase3Model): string {
  return name.includes("[") ? "reference" : m.role;
}

function ModelPlot({ fam, family }: { fam: Phase3Family; family: string }) {
  const rows: IntervalRow[] = Object.entries(fam.models)
    .filter(([, m]) => !m.excluded)
    .sort((a, b) => a[1].report_nll - b[1].report_nll)
    .map(([name, m]) => ({
      id: name,
      label: name,
      estimate: m.report_nll,
      lo: m.ci[0],
      hi: m.ci[1],
      color: ROLE_COLOR[roleOf(name, m)] ?? "#8a8f98",
    }));
  if (rows.length === 0) return null;
  return <IntervalPlot rows={rows} title={`${family}: mean NLL per choice (lower is better), 95% CI`} xLabel="nats per choice" zeroLine={null} />;
}

function QuarterPlot({ fam, family }: { fam: Phase3Family; family: string }) {
  const palette = ["var(--future)", "var(--lab)", "#f2b45a", "#8a8f98", "#c06c84", "#4e9a6f"];
  const top = Object.entries(fam.models)
    .filter(([n, m]) => !m.excluded && !n.includes("[") && m.quarters.every((q) => Number.isFinite(q)))
    .sort((a, b) => a[1].report_nll - b[1].report_nll)
    .slice(0, 6);
  const series: LineSeries[] = top.map(([name, m], i) => ({
    id: name,
    label: name,
    color: palette[i % palette.length] as string,
    dashed: m.role !== "candidate",
    points: m.quarters.map((y, q) => ({ x: q + 1, y })),
  }));
  if (series.length === 0) return null;
  return <LineChart series={series} title={`${family}: NLL by quarter of each participant's trials (learning trajectory)`} xLabel="quarter of trials" yLabel="NLL" height={220} />;
}

function ModelTable({ fam }: { fam: Phase3Family }) {
  return (
    <div className="table-wrap" tabIndex={0}>
      <table>
        <caption className="sr-only">Models</caption>
        <thead>
          <tr>
            <th>Model</th>
            <th>Role</th>
            <th className="num">NLL [95% CI]</th>
            <th className="num">n</th>
            <th className="num">Top-1</th>
            <th className="num">ECE</th>
            <th className="num">Params</th>
            <th className="num">Fit CPU s</th>
            <th>Checks</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(fam.models)
            .sort((a, b) => a[1].report_nll - b[1].report_nll)
            .map(([name, m]) => (
              <tr key={name}>
                <td className="mono">{name}</td>
                <td>{roleOf(name, m)}</td>
                <td className="num">
                  {fmtNum(m.report_nll, 4)} [{fmtNum(m.ci[0], 4)}, {fmtNum(m.ci[1], 4)}]
                </td>
                <td className="num">{fmtInt(m.n)}</td>
                <td className="num">{fmtNum(m.top1, 3)}</td>
                <td className="num">{fmtNum(m.ece, 3)}</td>
                <td className="num">{m.params === null ? "—" : fmtInt(m.params)}</td>
                <td className="num">{fmtNum(m.fit_cpu_s, 0)}</td>
                <td>
                  {m.excluded || m.causality_violations.length > 0 ? (
                    <span className="chip chip--bad" title={m.causality_violations.join("; ")}>
                      excluded: causality
                    </span>
                  ) : (
                    <span className="chip chip--ok">causal</span>
                  )}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

function Contrasts({ fam }: { fam: Phase3Family }) {
  if (fam.contrasts.length === 0) return null;
  return (
    <div className="table-wrap" tabIndex={0}>
      <table>
        <caption className="sr-only">Paired contrasts</caption>
        <thead>
          <tr>
            <th>Contrast (negative favours the first)</th>
            <th>Kind</th>
            <th className="num">Estimate</th>
            <th className="num">95% CI</th>
            <th className="num">n</th>
          </tr>
        </thead>
        <tbody>
          {fam.contrasts.map((c) => (
            <tr key={`${c.first}-${c.second}`}>
              <td className="mono">
                {c.first} − {c.second}
              </td>
              <td>{c.kind.replace(/_/g, " ")}</td>
              <td className="num">{fmtNum(c.estimate, 4)}</td>
              <td className="num">
                [{fmtNum(c.ci_low, 4)}, {fmtNum(c.ci_high, 4)}]
              </td>
              <td className="num">{fmtInt(c.n)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExperimentView({ exp }: { exp: Phase3Experiment }) {
  const confirmatory = exp.report_partition === "test";
  return (
    <section className="card stack" aria-label={exp.experiment_id}>
      <div>
        <div className="eyebrow">
          {exp.lane} · {confirmatory ? "confirmatory (held-out test participants)" : "exploratory (tuning participants; selection-optimistic)"}
        </div>
        <h2>{exp.title || exp.experiment_id}</h2>
        <p className="muted mono">{exp.path}</p>
        {exp.hypothesis && <p>{exp.hypothesis}</p>}
      </div>
      {Object.entries(exp.families).map(([family, fam]) => (
        <div key={family} className="stack">
          <h3>
            {family} <span className="chip chip--neutral">{fam.role.replace(/_/g, " ")}</span>
            {fam.best_comparator && <span className="muted"> · strongest comparator on tune: {fam.best_comparator}</span>}
          </h3>
          <ModelPlot fam={fam} family={family} />
          <QuarterPlot fam={fam} family={family} />
          <ModelTable fam={fam} />
          <Contrasts fam={fam} />
        </div>
      ))}
    </section>
  );
}

function RegistryTable({ title, rows, cols }: { title: string; rows: Array<Record<string, string>>; cols: Array<[string, string]> }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {rows.length === 0 ? (
        <p className="muted">None recorded.</p>
      ) : (
        <div className="table-wrap" tabIndex={0}>
          <table>
            <caption className="sr-only">{title}</caption>
            <thead>
              <tr>
                {cols.map(([, label]) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {cols.map(([k]) => (
                    <td key={k}>{(r[k] ?? "").replace(/\*\*/g, "").replace(/`/g, "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function Phase3() {
  const d = useApi((s) => api.phase3(s), []);
  const [selected, setSelected] = useState<string>("");
  return (
    <div className="stack">
      <header className="page-head">
        <div className="eyebrow">Research Lab · Phase III</div>
        <h1>Population-trained models and individual adaptation</h1>
        <p>
          Does a model that learns across people <em>and</em> adapts to each person predict held-out human choices better than strong
          population-only and individual-only models? Every number below is read from committed result files; failures and abandoned attempts
          are shown with the same prominence.
        </p>
      </header>
      <Async {...d} loadingLabel="Loading Phase III results…">
        {(data) => (
          <Phase3Body data={data} selected={selected} setSelected={setSelected} />
        )}
      </Async>
    </div>
  );
}

function Phase3Body({
  data,
  selected,
  setSelected,
}: {
  data: NonNullable<Awaited<ReturnType<typeof api.phase3>>>;
  selected: string;
  setSelected: (s: string) => void;
}) {
  const exps = data.experiments;
  const current = useMemo(() => exps.find((e) => e.experiment_id === selected) ?? exps[exps.length - 1], [exps, selected]);
  const verdictGates = data.verdict?.gates ?? {};
  return (
    <div className="stack">
      <HumanDataBanner text={data.disclaimer} />
      <section className="card">
        <h2>Verdict and gates</h2>
        <p>
          {data.verdict ? (
            <>
              <strong>{data.verdict.decision}</strong> — {data.verdict.summary}
            </>
          ) : (
            "No verdict has been issued yet. Gates are evaluated only on held-out participants after a frozen, critic-approved preregistration."
          )}
        </p>
        <div className="table-wrap" tabIndex={0}>
          <table>
            <caption className="sr-only">Phase III gates</caption>
            <thead>
              <tr>
                <th>Gate</th>
                <th>Question</th>
                <th>Pass rule (fixed before testing)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {GATES.map((g) => {
                const st = verdictGates[g.id];
                return (
                  <tr key={g.id}>
                    <td className="mono">{g.id}</td>
                    <td>{g.name}</td>
                    <td>{g.rule}</td>
                    <td>
                      <span className={`chip chip--${st?.status === "passed" ? "ok" : st?.status === "failed" ? "bad" : "neutral"}`}>
                        {st?.status ?? "not evaluated"}
                      </span>
                      {st?.note && <div className="muted">{st.note}</div>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
      {exps.length === 0 ? (
        <EmptyState title="No Phase III results yet">Results appear here only after an experiment has written its summary.</EmptyState>
      ) : (
        <>
          <label className="field">
            <span>Experiment</span>
            <select value={current?.experiment_id ?? ""} onChange={(e) => setSelected(e.target.value)}>
              {exps.map((e) => (
                <option key={e.experiment_id} value={e.experiment_id}>
                  {e.experiment_id} — {e.report_partition === "test" ? "confirmatory" : "exploratory"}
                </option>
              ))}
            </select>
          </label>
          {current && <ExperimentView exp={current} />}
        </>
      )}
      <RegistryTable
        title="Failures and abandoned attempts"
        rows={data.registry.abandoned}
        cols={[
          ["date", "Date"],
          ["lane", "Lane"],
          ["what", "What"],
          ["why", "Why abandoned"],
        ]}
      />
      <RegistryTable
        title="Exploratory ledger"
        rows={data.registry.exploratory}
        cols={[
          ["id", "ID"],
          ["lane", "Lane"],
          ["question", "Question"],
          ["partition", "Partition"],
          ["result", "Result"],
        ]}
      />
      <RegistryTable
        title="Confirmatory ledger"
        rows={data.registry.confirmatory}
        cols={[
          ["id", "ID"],
          ["gate", "Gate"],
          ["result", "Result"],
          ["verdict", "Verdict"],
        ]}
      />
      <RegistryTable
        title="Amendments and disclosed deviations"
        rows={data.registry.amendments}
        cols={[
          ["date", "Date"],
          ["change", "Change"],
          ["seen_confirmatory", "Seen confirmatory data?"],
        ]}
      />
    </div>
  );
}
