import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import type { AgentDetail, CapabilityState, ExperienceEvent } from "../../types/contracts";
import { Async } from "../../components/AsyncState";
import { LabBanner } from "../../components/Badges";
import { LineChart } from "../../components/charts/LineChart";
import type { LineSeries } from "../../components/charts/LineChart";
import { Bar, JsonView, KeyValue, Stat } from "../../components/Stat";
import { EVENT_GROUPS, describeEvent, groupOf } from "../../lib/events";
import { FAMILY_META, chanceFor, familyLabel, policyLabel, seriesColor } from "../../lib/families";
import { fmtCI, fmtFixed, fmtInt, fmtNum, titleCase } from "../../lib/format";
import { flattenNumeric } from "../../lib/stats";

export default function AgentInspector() {
  const { runId = "", agentId = "" } = useParams();
  const detail = useApi((s) => api.getAgent(runId, agentId, s), [runId, agentId]);
  return (
    <div className="stack">
      <LabBanner />
      <nav aria-label="Breadcrumb" className="muted" style={{ fontSize: "0.9rem" }}>
        <Link to={`/lab/society?run=${encodeURIComponent(runId)}`}>Live society</Link> / <span className="mono">{agentId}</span>
      </nav>
      <Async {...detail} loadingLabel="Loading agent history…">
        {(d) => <Inspector detail={d} runId={runId} />}
      </Async>
    </div>
  );
}

function ageAtTick(detail: AgentDetail, tick: number): number | undefined {
  // Use skill curve / parameter history points (which carry both) to map tick → age
  const pts = [...detail.skill_curve.map((p) => [p.tick, p.age_years] as const), ...detail.parameter_history.map((p) => [p.tick, p.age_years] as const)];
  if (!pts.length) return undefined;
  pts.sort((a, b) => a[0] - b[0]);
  const first = pts[0];
  if (!first) return undefined;
  // ticks are uniform in time: estimate years per tick from the extremes
  const last = pts[pts.length - 1] ?? first;
  const ypt = last[0] !== first[0] ? (last[1] - first[1]) / (last[0] - first[0]) : 30 / 365;
  return first[1] + (tick - first[0]) * ypt;
}

function Inspector({ detail, runId }: { detail: AgentDetail; runId: string }) {
  const s = detail.state;
  const caps = Object.values(s.capabilities ?? {}).sort((a, b) => a.family.localeCompare(b.family));
  return (
    <div className="stack">
      <header className="page-head">
        <div className="eyebrow">Agent inspector</div>
        <h1 style={{ fontFamily: "var(--font-mono)" }}>
          {s.agent_id}
        </h1>
        <div className="row">
          <span className="chip chip--lab">{policyLabel(s.policy.policy_class)}</span>
          <span className="chip chip--neutral">policy v{s.policy.version}</span>
          <span className={`chip ${s.alive === false ? "chip--neutral" : "chip--ok"}`}>{s.alive === false ? "Simulated life ended" : "Alive in simulation"}</span>
          <span className="chip chip--neutral">{titleCase(s.stage)}</span>
        </div>
      </header>

      <div className="grid grid-4">
        <Stat label="Simulated age" value={fmtNum(s.age_years, 1)} hint={`cohort year ${s.birth_cohort_year}`} />
        <Stat label="Household" value={<span className="mono">{s.household_id}</span>} hint={`${s.parents?.length ?? 0} parents · ${s.children?.length ?? 0} children`} />
        <Stat label="Education" value={s.education_years !== undefined ? `${fmtNum(s.education_years, 1)} y` : "—"} hint={s.tutored ? "received tutoring" : s.enrolled_school ? "enrolled in school" : "not enrolled"} />
        <Stat label="Resources" value={fmtNum(s.resources)} hint={`household ${fmtNum(s.household_resources)}`} />
      </div>

      <section className="card" aria-labelledby="skills-title">
        <div className="card__header">
          <h2 id="skills-title">Acquired skills — measured on held-out tasks</h2>
          <small>Novel symbols never seen in practice · learning disabled during assessment</small>
        </div>
        <SkillCurves detail={detail} />
        {caps.length > 0 && <CapabilityTable caps={caps} />}
      </section>

      <section className="card" aria-labelledby="params-title">
        <div className="card__header">
          <h2 id="params-title">Policy parameter changes</h2>
          <small>Observable learned-parameter history — the mechanism behind the skill curve</small>
        </div>
        <ParameterHistory detail={detail} />
      </section>

      <div className="grid grid-2">
        <section className="card" aria-labelledby="timeline-title">
          <h2 id="timeline-title">Sourced life timeline</h2>
          <Timeline detail={detail} />
        </section>
        <div className="stack">
          <section className="card" aria-labelledby="decision-title">
            <h2 id="decision-title">Latest decision explanation</h2>
            {detail.latest_decision_explanation ? (
              <DecisionExplanation value={detail.latest_decision_explanation} />
            ) : (
              <p className="muted">No sampled decision explanation recorded for this agent yet.</p>
            )}
          </section>
          <section className="card" aria-labelledby="rel-title">
            <h2 id="rel-title">Relationships ({detail.relationships.length})</h2>
            {detail.relationships.length === 0 ? (
              <p className="muted">No relationships recorded.</p>
            ) : (
              <div className="table-wrap" tabIndex={0} style={{ maxHeight: 280, overflow: "auto" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Relation</th>
                      <th>Trust</th>
                      <th className="num">Interactions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...detail.relationships]
                      .sort((a, b) => (b.trust ?? 0) - (a.trust ?? 0))
                      .map((e) => {
                        const other = e.src === s.agent_id ? e.dst : e.src;
                        return (
                          <tr key={`${e.src}-${e.dst}-${e.kind}`}>
                            <td>
                              <Link className="mono" to={`/lab/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(other)}`}>
                                {other}
                              </Link>
                            </td>
                            <td>{e.kind}</td>
                            <td style={{ minWidth: 90 }}>
                              <Bar value={e.trust ?? 0} label={`trust ${fmtFixed(e.trust, 2)}`} />
                            </td>
                            <td className="num">{fmtInt(e.interactions)}</td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          <section className="card" aria-labelledby="mem-title">
            <h2 id="mem-title">Remembered episodes</h2>
            {detail.recent_memories && detail.recent_memories.length > 0 ? (
              <MemoryTable rows={detail.recent_memories} />
            ) : (
              <p className="muted">This policy class exposes no episodic memories{s.memory_items ? ` (memory size ${s.memory_items})` : ""}.</p>
            )}
          </section>
          <section className="card card--flat" aria-labelledby="persona-title">
            <h2 id="persona-title" style={{ fontSize: "1rem" }}>
              Generated biography
            </h2>
            <p className="muted" style={{ fontSize: "0.84rem" }}>
              Descriptive text generated from abstract model traits. It is not evidence of skill and is not used by learning agents.
            </p>
            <p>{s.persona || <span className="muted">No biography text.</span>}</p>
            {s.traits && Object.keys(s.traits).length > 0 && (
              <KeyValue items={Object.entries(s.traits).map(([k, v]) => [titleCase(k), fmtFixed(v, 2)])} />
            )}
            {s.affect && (
              <>
                <h3 style={{ fontSize: "0.92rem", marginTop: 12 }}>Affect proxies (modelled signals, not emotions)</h3>
                <KeyValue items={Object.entries(s.affect).map(([k, v]) => [titleCase(k), fmtFixed(v as number, 2)])} />
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

function SkillCurves({ detail }: { detail: AgentDetail }) {
  const byFam = useMemo(() => {
    const m = new Map<string, AgentDetail["skill_curve"]>();
    for (const p of detail.skill_curve) {
      const arr = m.get(p.family) ?? [];
      arr.push(p);
      m.set(p.family, arr);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [detail.skill_curve]);
  if (!byFam.length) return <p className="muted">No held-out assessments recorded for this agent yet (assessments run once per simulated year).</p>;
  return (
    <div className="grid grid-2">
      {byFam.map(([fam, pts], i) => {
        const series: LineSeries[] = [
          {
            id: fam,
            label: `${familyLabel(fam)} held-out accuracy`,
            color: seriesColor(i),
            points: [...pts].sort((a, b) => a.age_years - b.age_years).map((p) => ({ x: p.age_years, y: p.accuracy, lo: p.ci_low, hi: p.ci_high })),
          },
        ];
        const chance = chanceFor(fam);
        const first = pts[0];
        const last = pts[pts.length - 1];
        return (
          <div key={fam} className="card card--flat">
            <h3>
              {familyLabel(fam)} {FAMILY_META[fam]?.transfer && <span className="chip chip--warn">never practised</span>}
            </h3>
            <LineChart
              title={`${familyLabel(fam)}: held-out accuracy by simulated age`}
              series={series}
              xLabel="simulated age (years)"
              yLabel="held-out accuracy"
              yDomain={[0, 1]}
              refLines={chance !== undefined ? [{ y: chance, label: `chance ${fmtFixed(chance, 2)}` }] : []}
              height={210}
            />
            {first && last && (
              <p className="muted" style={{ fontSize: "0.84rem", margin: "6px 0 0" }}>
                {fmtFixed(first.accuracy)} at age {fmtNum(first.age_years, 1)} → {fmtFixed(last.accuracy)} at age {fmtNum(last.age_years, 1)} · {last.n_items} items per
                assessment · 95% Wilson interval shaded
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CapabilityTable({ caps }: { caps: CapabilityState[] }) {
  return (
    <div className="table-wrap" tabIndex={0} style={{ marginTop: 14 }}>
      <table>
        <caption className="sr-only">Capabilities: measured versus self-estimated</caption>
        <thead>
          <tr>
            <th>Family</th>
            <th className="num">Measured accuracy</th>
            <th>95% interval</th>
            <th className="num">Self-estimate</th>
            <th className="num">Gap (self − measured)</th>
            <th className="num">Practice episodes</th>
          </tr>
        </thead>
        <tbody>
          {caps.map((c) => {
            const gap = c.self_estimate_mean !== null && c.self_estimate_mean !== undefined && c.measured_accuracy !== null && c.measured_accuracy !== undefined ? c.self_estimate_mean - c.measured_accuracy : undefined;
            return (
              <tr key={c.family}>
                <td>{familyLabel(c.family)}</td>
                <td className="num">{fmtFixed(c.measured_accuracy, 3)}</td>
                <td>{fmtCI(c.measured_ci_low, c.measured_ci_high)}</td>
                <td className="num">
                  {fmtFixed(c.self_estimate_mean, 3)}
                  {c.self_estimate_sd !== null && c.self_estimate_sd !== undefined ? ` ± ${fmtFixed(c.self_estimate_sd, 2)}` : ""}
                </td>
                <td className="num">{gap === undefined ? "—" : `${gap > 0 ? "+" : ""}${fmtFixed(gap, 3)}`}</td>
                <td className="num">{fmtInt(c.practice_episodes)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: "0.82rem", padding: "8px 10px", margin: 0 }}>
        Self-estimate is the policy's internal belief. It is shown for calibration only and is never counted as skill.
      </p>
    </div>
  );
}

function ParameterHistory({ detail }: { detail: AgentDetail }) {
  const flat = useMemo(
    () => detail.parameter_history.map((p) => ({ tick: p.tick, age: p.age_years, values: flattenNumeric(p.summary) })).sort((a, b) => a.age - b.age),
    [detail.parameter_history],
  );
  const keys = useMemo(() => {
    const all = new Set<string>();
    for (const f of flat) for (const k of Object.keys(f.values)) all.add(k);
    // keep keys that actually vary over time (these show learning); then stable ones
    const varying = [...all].filter((k) => new Set(flat.map((f) => f.values[k])).size > 1);
    return { varying: varying.sort(), all: [...all].sort() };
  }, [flat]);
  const [chosen, setChosen] = useState<string[] | null>(null);
  const selected = chosen ?? keys.varying.filter((k) => !k.startsWith("compute.")).slice(0, 4);

  if (!flat.length) return <p className="muted">No parameter snapshots recorded for this agent.</p>;
  const last = detail.parameter_history[detail.parameter_history.length - 1];
  const series: LineSeries[] = selected.map((k, i) => ({
    id: k,
    label: k,
    color: seriesColor(i),
    points: flat.filter((f) => f.values[k] !== undefined).map((f) => ({ x: f.age, y: f.values[k] as number })),
  }));
  return (
    <div className="stack">
      {keys.varying.length === 0 ? (
        <p className="notice">
          No parameter in this policy's summary changed over the recorded history. For a non-learning baseline this is the expected result; for a learner it would indicate that
          no learning occurred.
        </p>
      ) : (
        <>
          <fieldset style={{ border: "none", padding: 0, margin: 0 }}>
            <legend className="muted" style={{ fontSize: "0.86rem" }}>
              Parameters that changed over the agent's life ({keys.varying.length}). Select up to 6.
            </legend>
            <div className="row" style={{ gap: 6 }}>
              {keys.varying.slice(0, 30).map((k) => {
                const on = selected.includes(k);
                return (
                  <button
                    key={k}
                    type="button"
                    className="btn btn--sm mono"
                    aria-pressed={on}
                    onClick={() => setChosen(on ? selected.filter((x) => x !== k) : [...selected, k].slice(-6))}
                    style={on ? { borderColor: "var(--lab)", background: "var(--lab-soft)" } : undefined}
                  >
                    {k}
                  </button>
                );
              })}
            </div>
          </fieldset>
          <div className="grid grid-2">
            {series.map((s) => (
              <div key={s.id}>
                <h3 className="mono" style={{ fontSize: "0.85rem", margin: "0 0 4px" }}>
                  {s.label}
                </h3>
                <LineChart title={`${s.label} by simulated age`} series={[s]} xLabel="simulated age (years)" showBands={false} height={190} />
              </div>
            ))}
          </div>
        </>
      )}
      {last && (
        <details>
          <summary>Latest full parameter summary (tick {last.tick})</summary>
          <JsonView value={last.summary} label="Latest parameter summary" />
        </details>
      )}
    </div>
  );
}

function Timeline({ detail }: { detail: AgentDetail }) {
  const [groups, setGroups] = useState<Set<string>>(new Set(Object.keys(EVENT_GROUPS).filter((g) => g !== "decisions" && g !== "social")));
  const events = useMemo(() => [...detail.timeline].sort((a, b) => a.seq - b.seq), [detail.timeline]);
  const shown = events.filter((e) => groups.has(groupOf(e.type)));
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row" role="group" aria-label="Filter timeline">
        {Object.entries(EVENT_GROUPS).map(([k, g]) => {
          const on = groups.has(k);
          const n = events.filter((e) => groupOf(e.type) === k).length;
          return (
            <button
              key={k}
              type="button"
              className="btn btn--sm"
              aria-pressed={on}
              onClick={() => {
                const next = new Set(groups);
                if (on) next.delete(k);
                else next.add(k);
                setGroups(next);
              }}
            >
              <i style={{ width: 8, height: 8, borderRadius: 4, background: g.color, display: "inline-block" }} aria-hidden="true" /> {g.label} ({n})
            </button>
          );
        })}
      </div>
      {shown.length === 0 ? (
        <p className="muted">No events in the selected categories.</p>
      ) : (
        <ol className="timeline" tabIndex={0} aria-label="Life timeline">
          {shown.map((e) => (
            <TimelineItem key={e.seq} e={e} age={ageAtTick(detail, e.tick)} />
          ))}
        </ol>
      )}
    </div>
  );
}

function TimelineItem({ e, age }: { e: ExperienceEvent; age: number | undefined }) {
  const g = EVENT_GROUPS[groupOf(e.type)];
  return (
    <li>
      <span className="age">{age !== undefined ? `age ${fmtNum(Math.max(0, age), 1)}` : `t${e.tick}`}</span>
      <span className="dot" style={{ background: g?.color }} aria-hidden="true" />
      <span>
        {describeEvent(e)}
        <div className="prov">
          seq {e.seq} · tick {e.tick} · source: {e.provenance.source} · mechanism <span className="mono">{e.provenance.mechanism}</span>
        </div>
      </span>
    </li>
  );
}

function DecisionExplanation({ value }: { value: Record<string, unknown> }) {
  const flat = flattenNumeric(value);
  const numericEntries = Object.entries(flat);
  const text = Object.entries(value).filter(([, v]) => typeof v === "string") as Array<[string, string]>;
  return (
    <div className="stack" style={{ gap: 8 }}>
      {text.length > 0 && <KeyValue items={text.map(([k, v]) => [titleCase(k), v])} />}
      {numericEntries.length > 0 && numericEntries.length <= 16 && (
        <div className="table-wrap" tabIndex={0}>
          <table>
            <tbody>
              {numericEntries.map(([k, v]) => (
                <tr key={k}>
                  <td className="mono">{k}</td>
                  <td className="num">{fmtFixed(v, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Raw explanation</summary>
        <JsonView value={value} label="Decision explanation" />
      </details>
    </div>
  );
}

function MemoryTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))].slice(0, 6);
  return (
    <div className="table-wrap" tabIndex={0} style={{ maxHeight: 240, overflow: "auto" }}>
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{titleCase(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 40).map((r, i) => (
            <tr key={i}>
              {cols.map((c) => {
                const v = r[c];
                return <td key={c}>{typeof v === "number" ? fmtFixed(v, 2) : typeof v === "string" ? v : JSON.stringify(v)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
