import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../client";
import { futuresTransport, isFuturesPath } from "./futures";
import { engineProgress, setPyodideLoader, type PyodideLike } from "./pyodide";
import * as store from "./storage";

const USER = "u-fut";

/** A fake Pyodide whose `hpe.futures.bridge` mimics the Python bridge's contract. */
function fakePyodide(opts: { fail?: "consent" | "validation" | "boot" } = {}): PyodideLike {
  const loadPackage = vi.fn(async () => undefined);
  const runPythonAsync = vi.fn(async (code: string) => {
    expect(code).toContain("micropip.install");
    expect(code).toContain("deps=False");
    return undefined;
  });
  const bridge = {
    generate: (requestJson: string, decisionId: string | null) => {
      const req = JSON.parse(requestJson) as { consent_given: boolean; options: unknown[] };
      if (opts.fail === "consent" || !req.consent_given) throw new Error("Traceback\nhpe.futures.bridge.ConsentRequired: consent_given must be true");
      if (opts.fail === "validation" || req.options.length < 2) throw new Error("Traceback\npydantic_core._pydantic_core.ValidationError: 1 validation error");
      const id = decisionId ?? "dec-abcdefabcdef";
      return JSON.stringify({
        decision_id: id,
        scenarios: [{ scenario_id: `${id}-a`, decision_id: id, option_id: "a", title: "A", label: "illustrative scenario", narrative: "n" }],
        evidence: [{ evidence_id: `${id}-fact-x`, domain: "user_input", source_type: "user_fact", claim: "c", validation_status: "unvalidated" }],
        safety: null,
        value_alignment_note: "note",
      });
    },
    evidence_catalog: () => JSON.stringify([{ evidence_id: "lib-1" }]),
  };
  return {
    loadPackage,
    runPythonAsync,
    pyimport: (name: string) => {
      expect(name).toBe("hpe.futures.bridge");
      return bridge;
    },
    globals: { get: () => undefined },
  };
}

const request = {
  user_id: USER,
  consent_given: true,
  question: "job or startup?",
  options: [
    { option_id: "a", label: "A", kind: "new_job", inputs: {} },
    { option_id: "b", label: "B", kind: "start_project", inputs: {} },
  ],
  values: [{ key: "income", label: "Income", weight: 0.5 }],
  constraints: {},
  horizon_years: 3,
};

describe("isFuturesPath", () => {
  it("claims only Future Self paths", () => {
    for (const p of ["/scenarios", "/decisions/scenarios", "/decisions/reflect", "/journey/u", "/privacy/export"]) expect(isFuturesPath("POST", p)).toBe(true);
    for (const p of ["/runs", "/evidence", "/experiments", "/research/claims", "/health"]) expect(isFuturesPath("GET", p)).toBe(false);
  });
});

describe("futuresTransport with a fake Pyodide", () => {
  beforeEach(() => {
    store.deleteData(USER);
    setPyodideLoader(async () => fakePyodide());
  });

  it("boots lazily, generates scenarios through the bridge and stores the decision locally", async () => {
    expect(engineProgress().stage).toBe("idle");
    const resp = (await futuresTransport("POST", "/scenarios", { body: request })) as { decision_id: string; scenarios: unknown[] };
    expect(resp.scenarios).toHaveLength(1);
    expect(engineProgress().stage).toBe("ready");
    const j = (await futuresTransport("GET", `/journey/${USER}`, {})) as { decisions: unknown[] };
    expect(j.decisions).toHaveLength(1);
  });

  it("maps consent and validation failures to the API's status codes", async () => {
    await expect(futuresTransport("POST", "/scenarios", { body: { ...request, consent_given: false } })).rejects.toMatchObject({ status: 400 });
    setPyodideLoader(async () => fakePyodide({ fail: "validation" }));
    await expect(futuresTransport("POST", "/scenarios", { body: request })).rejects.toMatchObject({ status: 422 });
  });

  it("reflections need consent and a known decision; export and delete round-trip", async () => {
    const now = new Date().toISOString();
    await expect(futuresTransport("POST", "/decisions/reflect", { body: { user_id: USER, decision_id: "x", created_at: now } })).rejects.toMatchObject({ status: 403 });
    await futuresTransport("POST", "/privacy/consent", { body: { user_id: USER, purpose: "reflection_storage", granted: true, created_at: now } });
    await expect(futuresTransport("POST", "/decisions/reflect", { body: { user_id: USER, decision_id: "x", created_at: now } })).rejects.toMatchObject({ status: 404 });
    const resp = (await futuresTransport("POST", "/scenarios", { body: request })) as { decision_id: string };
    await futuresTransport("POST", "/decisions/reflect", { body: { user_id: USER, decision_id: resp.decision_id, what_happened: "ok", created_at: now } });
    const exported = (await futuresTransport("POST", "/privacy/export", { body: { user_id: USER } })) as { records: Record<string, unknown[]> };
    expect(exported.records.user_reflections).toHaveLength(1);
    const deleted = (await futuresTransport("POST", "/privacy/delete", { body: { user_id: USER } })) as { deleted: Record<string, number> };
    expect(deleted.deleted.user_reflections).toBe(1);
    expect(((await futuresTransport("GET", `/journey/${USER}`, {})) as { decisions: unknown[] }).decisions).toHaveLength(0);
  });

  it("surfaces a boot failure as a 503 ApiError and allows retry", async () => {
    setPyodideLoader(async () => {
      throw new Error("network down");
    });
    await expect(futuresTransport("POST", "/scenarios", { body: request })).rejects.toBeInstanceOf(ApiError);
    expect(engineProgress().stage).toBe("error");
    setPyodideLoader(async () => fakePyodide());
    const resp = (await futuresTransport("POST", "/scenarios", { body: request })) as { scenarios: unknown[] };
    expect(resp.scenarios).toHaveLength(1);
  });

  it("rejects unknown mutations instead of pretending", async () => {
    await expect(futuresTransport("POST", "/privacy/nope", { body: {} })).rejects.toMatchObject({ status: 405 });
  });
});
