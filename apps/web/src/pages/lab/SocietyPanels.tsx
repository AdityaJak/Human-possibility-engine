import { Link } from "react-router-dom";
import type { AgentPage, ExperienceEvent, WorldTick } from "../../types/contracts";
import { BarChart, Sparkline } from "../../components/charts/BarChart";
import { Stat } from "../../components/Stat";
import { familyLabel } from "../../lib/families";
import { fmtFixed, fmtInt, fmtNum, fmtShare } from "../../lib/format";
import { describeEvent } from "../../lib/events";

function sortAgeKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => (parseFloat(a) || 0) - (parseFloat(b) || 0));
}

export function SocietyStats({ tick, history }: { tick: WorldTick | undefined; history: WorldTick[] }) {
  if (!tick) return <p className="muted">No tick data yet.</p>;
  const ageKeys = sortAgeKeys(Object.keys(tick.age_histogram ?? {}));
  const fams = Object.keys(tick.mean_measured_accuracy ?? {}).sort();
  return (
    <div className="stack">
      <div className="grid grid-4">
        <Stat label="Population" value={fmtInt(tick.population)} hint={`+${tick.births ?? 0} births · −${tick.deaths ?? 0} deaths this tick`} />
        <Stat label="Virtual time" value={`Year ${Math.floor(tick.year)}`} hint={`tick ${tick.tick} · day ${fmtInt(tick.day)}`} />
        <Stat label="Employment" value={fmtShare(tick.employment_rate)} hint="share of working-age agents employed" />
        <Stat label="School enrolment" value={fmtShare(tick.school_enrolment_rate)} hint="share of school-age agents enrolled" />
        <Stat label="Resource Gini" value={fmtFixed(tick.resource_gini, 3)} hint={<Sparkline values={history.map((h) => h.resource_gini ?? NaN)} label="Gini" />} />
        <Stat label="Mean resources" value={fmtNum(tick.mean_resources)} hint={<Sparkline values={history.map((h) => h.mean_resources ?? NaN)} label="resources" />} />
        <Stat label="Events emitted" value={fmtInt(tick.events_emitted)} hint="this tick" />
        <Stat label="Tick compute" value={tick.wall_time_ms !== undefined ? `${fmtNum(tick.wall_time_ms, 1)} ms` : "—"} hint="measured wall time" />
      </div>
      {fams.length > 0 && (
        <div className="card card--flat">
          <h3>Mean measured held-out accuracy (assessed agents)</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Task family</th>
                  <th className="num">Accuracy</th>
                  <th>Trend</th>
                </tr>
              </thead>
              <tbody>
                {fams.map((f) => (
                  <tr key={f}>
                    <td>{familyLabel(f)}</td>
                    <td className="num">{fmtFixed(tick.mean_measured_accuracy?.[f], 3)}</td>
                    <td>
                      <Sparkline values={history.map((h) => h.mean_measured_accuracy?.[f] ?? NaN)} label={familyLabel(f)} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted" style={{ fontSize: "0.82rem", marginTop: 8 }}>
            Measured by held-out assessment with learning disabled. Population averages mix ages, so they move with births and deaths as well as learning.
          </p>
        </div>
      )}
      {ageKeys.length > 0 && (
        <div className="card card--flat">
          <h3>Age structure</h3>
          <BarChart title="Age structure of living agents" data={ageKeys.map((k) => ({ label: k, value: tick.age_histogram?.[k] ?? 0, color: "var(--lab)" }))} />
        </div>
      )}
    </div>
  );
}

export function EventsFeed({ events, runId }: { events: ExperienceEvent[]; runId: string }) {
  if (!events.length) return <p className="muted">No significant events in this window.</p>;
  return (
    <ul className="timeline" aria-live="polite" aria-label="Significant events" style={{ maxHeight: 300 }}>
      {events.map((e) => (
        <li key={e.seq}>
          <span className="age">t{e.tick}</span>
          <span className="dot" aria-hidden="true" />
          <span>
            {describeEvent(e)}{" "}
            {e.agent_id && (
              <Link to={`/lab/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(e.agent_id)}`} className="mono" style={{ fontSize: "0.78rem" }}>
                {e.agent_id}
              </Link>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function AgentList({
  page,
  runId,
  selectedId,
  onSelect,
}: {
  page: AgentPage | undefined;
  runId: string;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (!page) return null;
  if (!page.items.length) return <p className="muted">No matching agents.</p>;
  return (
    <div className="table-wrap" style={{ maxHeight: 320, overflow: "auto" }}>
      <table>
        <caption className="sr-only">Agents</caption>
        <thead>
          <tr>
            <th>Agent</th>
            <th className="num">Age</th>
            <th>Activity</th>
            <th className="num">Held-out acc.</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {page.items.map((a) => (
            <tr key={a.agent_id} aria-selected={a.agent_id === selectedId} style={a.agent_id === selectedId ? { background: "var(--lab-soft)" } : undefined}>
              <td>
                <button type="button" className="btn btn--ghost btn--sm mono" onClick={() => onSelect(a.agent_id)} aria-label={`Select ${a.agent_id} on the map`}>
                  {a.agent_id}
                </button>
                {!a.alive && <span className="chip chip--neutral">ended</span>}
              </td>
              <td className="num">{fmtNum(a.age_years, 1)}</td>
              <td>{a.activity ?? "—"}</td>
              <td className="num">{fmtFixed(a.mean_measured_accuracy, 2)}</td>
              <td>
                <Link to={`/lab/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(a.agent_id)}`}>Inspect</Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {page.total > page.items.length && (
        <p className="muted" style={{ padding: 8, margin: 0, fontSize: "0.82rem" }}>
          Showing {page.items.length} of {page.total}. Refine the search to narrow down.
        </p>
      )}
    </div>
  );
}
