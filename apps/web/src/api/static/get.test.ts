import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../client";
import { filterAgents, filterEvents, filterFrames, staticDownloadUrl, staticGet } from "./get";
import type { AgentPage, EventPage, FramesResponse } from "../../types/contracts";

function agent(id: string, household: string, alive = true, stage = "adult") {
  return { agent_id: id, household_id: household, age_years: 30, stage, alive, activity: null, job_id: null, education_years: 10, resources: 1, mean_measured_accuracy: null, n_friends: 0 };
}

function event(seq: number, type: string, tick: number, agent_id: string | null = "a1") {
  return { seq, run_id: "r", tick, day: tick * 30, type, agent_id, other_ids: [], payload: {}, provenance: { source: "world", mechanism: "m" }, namespace: "synthetic_simulation" } as EventPage["items"][number];
}

describe("static GET filters (same semantics as the API handlers)", () => {
  it("agents: q matches agent or household id, alive filter, sorted, paged", () => {
    const page: AgentPage = { items: [agent("a2", "h1", false), agent("a1", "h9"), agent("b3", "h1")], total: 3, offset: 0, limit: 3 };
    const r = filterAgents(page, { q: "h1", limit: 40 });
    expect(r.items.map((a) => a.agent_id)).toEqual(["a2", "b3"]);
    expect(r.total).toBe(2);
    expect(filterAgents(page, { alive: true }).items.map((a) => a.agent_id)).toEqual(["a1", "b3"]);
    expect(filterAgents(page, { offset: 1, limit: 1 }).items.map((a) => a.agent_id)).toEqual(["a2"]);
  });

  it("events: types, tick window, limit; total counts matches after filtering", () => {
    const page: EventPage = { items: [event(1, "birth", 0), event(2, "death", 5), event(3, "birth", 9), event(4, "wage", 9)], total: 4, offset: 0, limit: 4 };
    const r = filterEvents(page, { types: ["birth", "death"], tick_from: 1, tick_to: 9, limit: 40 });
    expect(r.items.map((e) => e.seq)).toEqual([2, 3]);
    expect(r.total).toBe(2);
    expect(filterEvents(page, { agent_id: "nobody" }).total).toBe(0);
  });

  it("frames: every is relative to the file's native stride", () => {
    const resp: FramesResponse = { run_id: "r", map: { width: 1, height: 1, places: [] }, ticks: [0, 3, 6, 9, 12, 15], frames: [[], [], [], [], [], []] };
    expect(filterFrames(resp, { every: 6 }).ticks).toEqual([0, 6, 12]);
    expect(filterFrames(resp, { every: 1 }).ticks).toEqual([0, 3, 6, 9, 12, 15]);
  });

  it("frames: window, every, limit", () => {
    const resp: FramesResponse = { run_id: "r", map: { width: 1, height: 1, places: [] }, ticks: [0, 1, 2, 3, 4, 5, 6], frames: [[], [], [], [], [], [], []] };
    expect(filterFrames(resp, { every: 3 }).ticks).toEqual([0, 3, 6]);
    expect(filterFrames(resp, { tick_from: 2, tick_to: 5, every: 2 }).ticks).toEqual([2, 4]);
    expect(filterFrames(resp, { limit: 2 }).ticks).toEqual([0, 1]);
  });
});

describe("staticGet over fetch", () => {
  const realFetch = globalThis.fetch;
  beforeEach(() => {
    vi.stubEnv("BASE_URL", "/site/");
  });
  afterEach(() => {
    globalThis.fetch = realFetch;
  });

  it("loads <base>/api<path>.json and applies the query client-side", async () => {
    const calls: string[] = [];
    globalThis.fetch = vi.fn(async (url: RequestInfo | URL) => {
      calls.push(String(url));
      const body: AgentPage = { items: [agent("a1", "h1"), agent("a2", "h2")], total: 2, offset: 0, limit: 2 };
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;
    const r = (await staticGet("/runs/demo/agents", { query: { q: "a2", limit: 40 } })) as AgentPage;
    expect(calls).toEqual(["/site/api/runs/demo/agents.json"]);
    expect(r.items.map((a) => a.agent_id)).toEqual(["a2"]);
  });

  it("404 becomes ApiError(404); network failure becomes ApiError(0)", async () => {
    globalThis.fetch = vi.fn(async () => new Response("nope", { status: 404 })) as typeof fetch;
    await expect(staticGet("/runs/missing", {})).rejects.toMatchObject({ status: 404 });
    globalThis.fetch = vi.fn(async () => {
      throw new TypeError("offline");
    }) as typeof fetch;
    const err = await staticGet("/health", {}).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
  });

  it("download url points at the pre-rendered gzip log", () => {
    expect(staticDownloadUrl("run x")).toBe("/site/api/runs/run%20x/download.jsonl.gz");
  });
});
