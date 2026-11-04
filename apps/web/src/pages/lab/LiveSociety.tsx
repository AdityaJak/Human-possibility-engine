import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../../api/client";
import { isStaticMode } from "../../api/config";
import { useApi } from "../../api/hooks";
import type { AgentFrame, AgentPage, ExperienceEvent, FramesResponse, PolicyClass, RunInfo, WorldMap, WorldTick } from "../../types/contracts";
import { LabBanner, SourceTag } from "../../components/Badges";
import { EmptyState, ErrorState, Loading } from "../../components/AsyncState";
import { JobStatusChip } from "../../components/StatusChip";
import { SocietyMap } from "../../components/SocietyMap";
import type { ColorBy } from "../../components/SocietyMap";
import { AgentList, EventsFeed, SocietyStats } from "./SocietyPanels";
import { SIGNIFICANT_EVENT_TYPES } from "../../lib/events";
import { POLICY_LABEL, policyLabel } from "../../lib/families";
import { fmtNum, shortId } from "../../lib/format";

const SPEEDS = [
  { ticks: 1, label: "1 tick" },
  { ticks: 3, label: "3 ticks" },
  { ticks: 12, label: "≈1 year" },
];

function useAgentSearch(runId: string | null, query: string, version: number) {
  return useApi<AgentPage | undefined>(
    (s) => (runId ? api.listAgents(runId, { q: query || undefined, limit: 40, alive: undefined }, s) : Promise.resolve(undefined)),
    [runId, query, version],
    !!runId,
  );
}

function NewRunForm({ onCreated, defaultSeed }: { onCreated: (r: RunInfo) => void; defaultSeed: number }) {
  const [seed, setSeed] = useState(defaultSeed);
  const [nAgents, setNAgents] = useState(100);
  const [years, setYears] = useState(80);
  const [policy, setPolicy] = useState<PolicyClass>("developmental");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const run = await api.createRun({ mode: "interactive", config: { seed, n_agents: nAgents, years, policy_class: policy, label: "dashboard interactive run" } });
      onCreated(run);
    } catch (ex) {
      setErr(ex instanceof Error ? ex : new Error(String(ex)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="stack" aria-label="Start a live interactive simulation">
      <div className="grid grid-2">
        <div className="field">
          <label htmlFor="nr-seed">Seed</label>
          <input id="nr-seed" type="number" value={seed} min={0} onChange={(e) => setSeed(Number(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="nr-agents">Initial agents</label>
          <input id="nr-agents" type="number" value={nAgents} min={10} max={2000} onChange={(e) => setNAgents(Number(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="nr-years">Years</label>
          <input id="nr-years" type="number" value={years} min={1} max={100} onChange={(e) => setYears(Number(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="nr-policy">Agent class</label>
          <select id="nr-policy" value={policy} onChange={(e) => setPolicy(e.target.value as PolicyClass)}>
            {Object.entries(POLICY_LABEL)
              .filter(([k]) => k !== "random")
              .map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
          </select>
        </div>
      </div>
      <div className="row">
        <button type="submit" className="btn btn--lab" disabled={busy}>
          {busy ? "Starting…" : "Start live simulation"}
        </button>
        <small>Runs on the local CPU. No language-model calls.</small>
      </div>
      {err && <ErrorState error={err} title="Could not start the run" />}
    </form>
  );
}

export default function LiveSociety() {
  const [params, setParams] = useSearchParams();
  const runs = useApi((s) => api.listRuns(s), []);
  const runId = params.get("run");
  const run = runs.data?.find((r) => r.run_id === runId) ?? null;

  // default selection: newest interactive run, otherwise first stored real output
  useEffect(() => {
    if (runId || !runs.data || !runs.data.length) return;
    const pick = runs.data.find((r) => r.source === "stored_real_output") ?? runs.data[0];
    if (pick) setParams({ run: pick.run_id }, { replace: true });
  }, [runId, runs.data, setParams]);

  const onCreated = useCallback(
    (r: RunInfo) => {
      setParams({ run: r.run_id });
      runs.reload();
    },
    [runs, setParams],
  );

  return (
    <div className="stack">
      <LabBanner />
      <header className="page-head">
        <div className="eyebrow">Research Lab · Live society</div>
        <h1>A developmental society you can inspect</h1>
        <p>
          Watch agents move between homes, school, workplaces and the plaza. Pick any agent to see its sourced life timeline and the skills it measurably acquired on held-out
          tasks.
        </p>
      </header>
      <section className="card" aria-labelledby="runs-title">
        <div className="card__header">
          <h2 id="runs-title">Runs</h2>
          <button type="button" className="btn btn--sm" onClick={runs.reload}>
            Refresh
          </button>
        </div>
        {runs.error && <ErrorState error={runs.error} onRetry={runs.reload} />}
        {runs.loading && !runs.data && <Loading label="Loading runs…" />}
        {runs.data && (
          <div className="split">
            <div>
              {runs.data.length === 0 ? (
                <EmptyState title="No runs yet">Start a live simulation, or generate the stored demo run with <code>make demo-data</code>.</EmptyState>
              ) : (
                <div className="field">
                  <label htmlFor="run-select">Select a run</label>
                  <select id="run-select" value={runId ?? ""} onChange={(e) => setParams({ run: e.target.value })}>
                    {runs.data.map((r) => (
                      <option key={r.run_id} value={r.run_id}>
                        {shortId(r.run_id, 18)} · {r.mode} · {r.source === "stored_real_output" ? "stored real output" : "live"} · {policyLabel(r.manifest?.policy_class ?? "")} · seed{" "}
                        {r.manifest?.config.seed ?? "?"}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {run && <RunMeta run={run} />}
            </div>
            {isStaticMode() ? (
              <p className="muted" data-testid="static-live-notice" style={{ margin: 0 }}>
                Live interactive runs need the API, which this static deployment does not include. You are replaying a stored real output; run{" "}
                <code>make demo</code> locally or deploy the API (docs/DEPLOYMENT.md) to start new simulations.
              </p>
            ) : (
              <details>
                <summary>
                  <strong>Start a new live run</strong> (switch seed)
                </summary>
                <div style={{ marginTop: 10 }}>
                  <NewRunForm onCreated={onCreated} defaultSeed={(run?.manifest?.config.seed ?? 0) + 1} />
                </div>
              </details>
            )}
          </div>
        )}
      </section>
      {run && <RunView key={run.run_id} run={run} onRunChanged={runs.reload} />}
    </div>
  );
}

function RunMeta({ run }: { run: RunInfo }) {
  const m = run.manifest;
  return (
    <div className="row" style={{ marginTop: 10 }}>
      <JobStatusChip status={run.status} />
      <SourceTag source={run.source} />
      {m && (
        <span className="muted" style={{ fontSize: "0.86rem" }}>
          {policyLabel(m.policy_class)} {m.policy_version ? `v${m.policy_version}` : ""} · {m.config.n_agents ?? "?"} initial agents · {m.config.years ?? "?"} years · dt{" "}
          {m.config.dt_days ?? 30} d · config <span className="mono">{shortId(m.config_hash, 10)}</span>
          {m.code_version && (
            <>
              {" "}
              · code <span className="mono">{shortId(m.code_version, 10)}</span>
            </>
          )}
        </span>
      )}
      <a className="btn btn--sm" href={api.downloadUrl(run.run_id)}>
        Download raw log
      </a>
    </div>
  );
}

function RunView({ run, onRunChanged }: { run: RunInfo; onRunChanged: () => void }) {
  return run.mode === "interactive" ? <InteractiveView run={run} onRunChanged={onRunChanged} /> : <ReplayView run={run} />;
}

// ----------------------------------------------------------------------------------------
// Shared right-hand panel: agent search + selected agent + events
// ----------------------------------------------------------------------------------------

function SidePanel({
  runId,
  selectedId,
  onSelect,
  events,
  version,
}: {
  runId: string;
  selectedId: string | null;
  onSelect: (id: string) => void;
  events: ExperienceEvent[];
  version: number;
}) {
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 250);
    return () => clearTimeout(t);
  }, [q]);
  const agents = useAgentSearch(runId, debounced, version);
  return (
    <div className="stack">
      <section className="card" aria-labelledby="agents-title">
        <h2 id="agents-title" style={{ fontSize: "1.1rem" }}>
          Find an agent
        </h2>
        <div className="field" style={{ marginBottom: 10 }}>
          <label htmlFor="agent-q" className="sr-only">
            Search agents
          </label>
          <input id="agent-q" type="search" placeholder="Search by id or household…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {agents.error ? <ErrorState error={agents.error} onRetry={agents.reload} /> : <AgentList page={agents.data} runId={runId} selectedId={selectedId} onSelect={onSelect} />}
        {selectedId && (
          <p style={{ marginTop: 10, marginBottom: 0 }}>
            <Link className="btn btn--lab btn--sm" to={`/lab/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(selectedId)}`}>
              Inspect {selectedId}: timeline, learned skills, decisions →
            </Link>
          </p>
        )}
      </section>
      <section className="card" aria-labelledby="events-title">
        <h2 id="events-title" style={{ fontSize: "1.1rem" }}>
          Significant events
        </h2>
        <EventsFeed events={events} runId={runId} />
      </section>
    </div>
  );
}

function MapCard({
  map,
  frames,
  selectedId,
  onSelect,
  children,
  title,
}: {
  map: WorldMap | undefined;
  frames: AgentFrame[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  children: React.ReactNode;
  title: string;
}) {
  const [colorBy, setColorBy] = useState<ColorBy>("activity");
  return (
    <section className="card" aria-labelledby="map-title">
      <div className="card__header">
        <h2 id="map-title" style={{ fontSize: "1.1rem" }}>
          {title}
        </h2>
        <div className="row" role="group" aria-label="Colour agents by">
          <button type="button" className="btn btn--sm" aria-pressed={colorBy === "activity"} onClick={() => setColorBy("activity")}>
            Activity
          </button>
          <button type="button" className="btn btn--sm" aria-pressed={colorBy === "age"} onClick={() => setColorBy("age")}>
            Age
          </button>
        </div>
      </div>
      {map ? <SocietyMap map={map} frames={frames} colorBy={colorBy} selectedId={selectedId} onSelect={onSelect} /> : <Loading label="Loading map…" />}
      <div style={{ marginTop: 12 }}>{children}</div>
    </section>
  );
}

// ----------------------------------------------------------------------------------------
// Interactive (live) runs: client-driven stepping through POST /runs/{id}/step
// ----------------------------------------------------------------------------------------

function InteractiveView({ run, onRunChanged }: { run: RunInfo; onRunChanged: () => void }) {
  const [map, setMap] = useState<WorldMap | undefined>(undefined);
  const [frames, setFrames] = useState<AgentFrame[]>([]);
  const [tick, setTick] = useState<WorldTick | undefined>(run.latest_tick ?? undefined);
  const [history, setHistory] = useState<WorldTick[]>([]);
  const [events, setEvents] = useState<ExperienceEvent[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState<Error | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const playingRef = useRef(false);
  const busyRef = useRef(false);

  const loadInitial = useCallback(async () => {
    try {
      const fr: FramesResponse = await api.getFrames(run.run_id, { every: 1 });
      setMap(fr.map);
      const last = fr.frames[fr.frames.length - 1];
      if (last) setFrames(last);
      const ts = await api.getTimeseries(run.run_id).catch(() => undefined);
      if (ts && ts.ticks.length) {
        setHistory(ts.ticks.slice(-240));
        setTick(ts.ticks[ts.ticks.length - 1]);
      }
      const ev = await api.listEvents(run.run_id, { types: SIGNIFICANT_EVENT_TYPES, limit: 30 }).catch(() => undefined);
      if (ev) setEvents([...ev.items].reverse());
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, [run.run_id]);

  useEffect(() => {
    void loadInitial();
  }, [loadInitial]);

  const stepOnce = useCallback(
    async (n: number) => {
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        const prevTick = tick?.tick ?? -1;
        const t = await api.stepRun(run.run_id, { ticks: n, include_frames: true });
        setTick(t);
        setHistory((h) => [...h.slice(-239), t]);
        if (t.frames) setFrames(t.frames);
        setError(null);
        const ev = await api
          .listEvents(run.run_id, { types: SIGNIFICANT_EVENT_TYPES, tick_from: prevTick + 1, tick_to: t.tick, limit: 50 })
          .catch(() => undefined);
        if (ev && ev.items.length)
          setEvents((old) => {
            const seen = new Set(old.map((x) => x.seq));
            const fresh = [...ev.items].reverse().filter((x) => !seen.has(x.seq));
            return [...fresh, ...old].slice(0, 60);
          });
        if (t.tick % 12 === 0) setVersion((v) => v + 1);
      } catch (e) {
        setError(e instanceof Error ? e : new Error(String(e)));
        playingRef.current = false;
        setPlaying(false);
      } finally {
        busyRef.current = false;
      }
    },
    [run.run_id, tick],
  );

  useEffect(() => {
    playingRef.current = playing;
    if (!playing) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      if (cancelled || !playingRef.current) return;
      await stepOnce(speed);
      if (!cancelled && playingRef.current) timer = setTimeout(loop, 220);
    };
    void loop();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed]);

  async function control(action: "pause" | "resume" | "reset") {
    try {
      await api.controlRun(run.run_id, { action });
      if (action === "reset") {
        setPlaying(false);
        setHistory([]);
        setEvents([]);
        setTick(undefined);
        await loadInitial();
        onRunChanged();
      }
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }

  function togglePlay() {
    const next = !playing;
    setPlaying(next);
    void api.controlRun(run.run_id, { action: next ? "resume" : "pause" }).catch(() => undefined);
  }

  const finished = run.status === "completed" || (tick !== undefined && run.manifest?.config.years !== undefined && tick.year >= run.manifest.config.years);

  return (
    <div className="split">
      <div className="stack">
        <MapCard map={map} frames={frames} selectedId={selected} onSelect={setSelected} title={`Live map · ${tick ? `Year ${fmtNum(tick.year, 1)}` : "not started"}`}>
          <div className="row" role="toolbar" aria-label="Simulation controls">
            <button type="button" className="btn btn--lab" onClick={togglePlay} disabled={finished} aria-pressed={playing}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" className="btn" onClick={() => void stepOnce(1)} disabled={playing || finished}>
              Step 1 tick
            </button>
            <button type="button" className="btn" onClick={() => void stepOnce(12)} disabled={playing || finished}>
              Step ≈1 year
            </button>
            <button type="button" className="btn btn--danger" onClick={() => void control("reset")}>
              Reset
            </button>
            <label htmlFor="speed" className="sr-only">
              Ticks per step
            </label>
            <select id="speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: "auto" }}>
              {SPEEDS.map((s) => (
                <option key={s.ticks} value={s.ticks}>
                  {s.label} per step
                </option>
              ))}
            </select>
            {finished && <span className="chip chip--ok">Run reached its configured horizon</span>}
          </div>
          {error && (
            <div style={{ marginTop: 10 }}>
              <ErrorState error={error} title="Simulation step failed" />
            </div>
          )}
        </MapCard>
        <section className="card" aria-labelledby="stats-title">
          <h2 id="stats-title" style={{ fontSize: "1.1rem" }}>
            Society indicators
          </h2>
          <SocietyStats tick={tick} history={history} />
        </section>
      </div>
      <SidePanel runId={run.run_id} selectedId={selected} onSelect={setSelected} events={events} version={version} />
    </div>
  );
}

// ----------------------------------------------------------------------------------------
// Stored / background runs: replay from /frames + /timeseries
// ----------------------------------------------------------------------------------------

function ReplayView({ run }: { run: RunInfo }) {
  const [every, setEvery] = useState(6);
  const frames = useApi((s) => api.getFrames(run.run_id, { every }, s), [run.run_id, every]);
  const series = useApi((s) => api.getTimeseries(run.run_id, s), [run.run_id]);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const n = frames.data?.ticks.length ?? 0;
  const currentTick = frames.data?.ticks[idx];

  useEffect(() => {
    setIdx(0);
  }, [frames.data]);

  useEffect(() => {
    if (!playing || n === 0) return;
    const t = setInterval(() => setIdx((i) => (i + 1 >= n ? (setPlaying(false), i) : i + 1)), 260);
    return () => clearInterval(t);
  }, [playing, n]);

  const tickData = useMemo(() => {
    const ticks = series.data?.ticks ?? [];
    if (!ticks.length || currentTick === undefined) return { tick: undefined, history: [] as WorldTick[] };
    let j = 0;
    for (let i = 0; i < ticks.length; i++) if ((ticks[i]?.tick ?? 0) <= currentTick) j = i;
    return { tick: ticks[j], history: ticks.slice(Math.max(0, j - 240), j + 1) };
  }, [series.data, currentTick]);

  const windowFrom = Math.max(0, (currentTick ?? 0) - every * 12);
  const events = useApi(
    (s) =>
      currentTick === undefined
        ? Promise.resolve(undefined)
        : api.listEvents(run.run_id, { types: SIGNIFICANT_EVENT_TYPES, tick_from: windowFrom, tick_to: currentTick, limit: 40 }, s),
    [run.run_id, currentTick === undefined ? -1 : Math.floor(currentTick / (every * 2))],
    currentTick !== undefined,
  );

  if (frames.error) return <ErrorState error={frames.error} onRetry={frames.reload} title="Could not load replay frames" />;

  return (
    <div className="split">
      <div className="stack">
        <MapCard
          map={frames.data?.map}
          frames={frames.data?.frames[idx] ?? []}
          selectedId={selected}
          onSelect={setSelected}
          title={`Replay · ${tickData.tick ? `Year ${fmtNum(tickData.tick.year, 1)}` : currentTick !== undefined ? `tick ${currentTick}` : "loading"}`}
        >
          {frames.loading && !frames.data ? (
            <Loading label="Loading recorded frames…" />
          ) : n === 0 ? (
            <EmptyState title="This run has no recorded map frames" />
          ) : (
            <div className="stack" style={{ gap: 10 }}>
              <div className="row" role="toolbar" aria-label="Replay controls">
                <button type="button" className="btn btn--lab" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
                  {playing ? "Pause" : idx >= n - 1 ? "Replay ended" : "Play replay"}
                </button>
                <button type="button" className="btn" onClick={() => setIdx((i) => Math.min(n - 1, i + 1))} disabled={playing}>
                  Step
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setPlaying(false);
                    setIdx(0);
                  }}
                >
                  Reset
                </button>
                <label htmlFor="every">Frame spacing</label>
                <select id="every" value={every} onChange={(e) => setEvery(Number(e.target.value))} style={{ width: "auto" }}>
                  {[1, 3, 6, 12].map((v) => (
                    <option key={v} value={v}>
                      every {v} tick{v > 1 ? "s" : ""}
                    </option>
                  ))}
                </select>
              </div>
              <label htmlFor="scrub" className="sr-only">
                Replay position
              </label>
              <input
                id="scrub"
                type="range"
                min={0}
                max={Math.max(0, n - 1)}
                value={idx}
                onChange={(e) => {
                  setPlaying(false);
                  setIdx(Number(e.target.value));
                }}
                style={{ accentColor: "var(--lab)" }}
                aria-valuetext={currentTick !== undefined ? `tick ${currentTick}` : undefined}
              />
              <small>
                Deterministic replay of a recorded run: the same seed and code reproduce the same event stream. Tick {currentTick ?? "—"} of {frames.data?.ticks[n - 1] ?? "—"}.
              </small>
            </div>
          )}
        </MapCard>
        <section className="card" aria-labelledby="stats-title">
          <h2 id="stats-title" style={{ fontSize: "1.1rem" }}>
            Society indicators
          </h2>
          {series.error ? <ErrorState error={series.error} onRetry={series.reload} /> : <SocietyStats tick={tickData.tick} history={tickData.history} />}
        </section>
      </div>
      <SidePanel runId={run.run_id} selectedId={selected} onSelect={setSelected} events={[...(events.data?.items ?? [])].reverse()} version={0} />
    </div>
  );
}
