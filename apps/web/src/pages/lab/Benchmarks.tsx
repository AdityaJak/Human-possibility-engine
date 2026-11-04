import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import { Async, EmptyState } from "../../components/AsyncState";
import { LabBanner } from "../../components/Badges";
import { policyLabel } from "../../lib/families";
import { fmtDuration, fmtInt, fmtNum } from "../../lib/format";
import type { BenchmarkRecord } from "../../types/contracts";

const TARGETS = [100, 1000, 10000];

function StatusCell({ s }: { s: BenchmarkRecord["status"] }) {
  const tone = s === "completed" ? "ok" : s === "timeout" ? "warn" : "bad";
  return <span className={`chip chip--${tone}`}>{s}</span>;
}

function ScaleTargets({ records }: { records: BenchmarkRecord[] }) {
  return (
    <div className="grid grid-3">
      {TARGETS.map((n) => {
        const done = records.filter((r) => r.n_agents >= n && r.status === "completed").sort((a, b) => a.wall_time_s - b.wall_time_s)[0];
        return (
          <div key={n} className="stat">
            <div className="stat__label">{fmtInt(n)}-agent target</div>
            <div className="stat__value">{done ? "Measured" : "Not yet measured"}</div>
            <div className="stat__hint">
              {done
                ? `${fmtNum(done.years, 0)} years in ${fmtDuration(done.wall_time_s)} on ${done.hardware}`
                : "Shown only after a real benchmark finishes on disclosed hardware."}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Benchmarks() {
  const b = useApi((s) => api.benchmarks(s), []);
  return (
    <div className="stack">
      <LabBanner>Performance benchmarks measure engineering capacity only. Agent count is not a measure of scientific success.</LabBanner>
      <header className="page-head">
        <div className="eyebrow">Research Lab · Benchmarks</div>
        <h1>Measured capacity and cost</h1>
        <p>Only completed, recorded benchmark runs are listed, with the hardware they ran on. Routine simulation steps make no paid model calls.</p>
      </header>
      <Async
        {...b}
        loadingLabel="Loading benchmark records…"
        isEmpty={(d) => d.length === 0}
        empty={
          <>
            <ScaleTargets records={[]} />
            <EmptyState title="Not yet measured">No benchmark has been recorded. Nothing about scale is claimed until one is.</EmptyState>
          </>
        }
      >
        {(d) => (
          <div className="stack">
            <ScaleTargets records={d} />
            <section className="card">
              <div className="table-wrap">
                <table>
                  <caption className="sr-only">Benchmark records</caption>
                  <thead>
                    <tr>
                      <th>Benchmark</th>
                      <th className="num">Agents</th>
                      <th className="num">Years</th>
                      <th className="num">dt (days)</th>
                      <th>Agent class</th>
                      <th className="num">Wall time</th>
                      <th className="num">Agent-years / s</th>
                      <th className="num">Ticks / s</th>
                      <th className="num">Peak RSS</th>
                      <th className="num">Events</th>
                      <th>Hardware</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...d]
                      .sort((a, b2) => a.n_agents - b2.n_agents)
                      .map((r) => (
                        <tr key={r.benchmark_id}>
                          <td className="mono">
                            {r.benchmark_id}
                            {r.notes && <div className="muted" style={{ fontFamily: "var(--font-sans)", fontSize: "0.8rem" }}>{r.notes}</div>}
                          </td>
                          <td className="num">{fmtInt(r.n_agents)}</td>
                          <td className="num">{fmtNum(r.years, 1)}</td>
                          <td className="num">{r.dt_days}</td>
                          <td>{policyLabel(r.policy_class)}</td>
                          <td className="num">{fmtDuration(r.wall_time_s)}</td>
                          <td className="num">{fmtNum(r.agent_years_per_s, 1)}</td>
                          <td className="num">{fmtNum(r.ticks_per_s, 2)}</td>
                          <td className="num">{r.peak_rss_mb !== null ? `${fmtNum(r.peak_rss_mb, 0)} MB` : "—"}</td>
                          <td className="num">{fmtInt(r.events)}</td>
                          <td>{r.hardware}</td>
                          <td>
                            <StatusCell s={r.status} />
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
      </Async>
    </div>
  );
}
