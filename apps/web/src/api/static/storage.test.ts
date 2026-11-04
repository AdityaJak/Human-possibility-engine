import { beforeEach, describe, expect, it } from "vitest";
import type { DecisionRequest, ScenarioResponse } from "../../types/contracts";
import * as store from "./storage";

const USER = "u-test";

const request: DecisionRequest = {
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

const response: ScenarioResponse = {
  decision_id: "dec-000000000001",
  scenarios: [
    {
      scenario_id: "dec-000000000001-a",
      decision_id: "dec-000000000001",
      option_id: "a",
      title: "A",
      label: "illustrative scenario",
      narrative: "n",
    } as ScenarioResponse["scenarios"][number],
  ],
  evidence: [
    { evidence_id: "dec-000000000001-fact-x", domain: "user_input", source_type: "user_fact", claim: "c", validation_status: "unvalidated" } as ScenarioResponse["evidence"][number],
    { evidence_id: "lib-1", domain: "d", source_type: "assumption", claim: "c", validation_status: "unvalidated" } as ScenarioResponse["evidence"][number],
  ],
  value_alignment_note: "note",
};

describe("browser-local user store", () => {
  beforeEach(() => {
    store.deleteData(USER);
  });

  it("records consent with latest-wins semantics", () => {
    const now = new Date().toISOString();
    store.consent({ user_id: USER, purpose: "reflection_storage", granted: true, created_at: now });
    expect(store.hasConsent(USER, ["reflection_storage"])).toBe(true);
    store.consent({ user_id: USER, purpose: "reflection_storage", granted: false, created_at: now });
    expect(store.hasConsent(USER, ["reflection_storage"])).toBe(false);
    expect(store.hasConsent(USER)).toBe(false);
  });

  it("keeps decisions, scenarios and only the user's own evidence; records implicit demo consent", () => {
    store.addDecision(request, response);
    const j = store.journey(USER);
    expect(j.decisions).toHaveLength(1);
    expect(j.decisions[0]).toMatchObject({ decision_id: "dec-000000000001" });
    expect(j.decisions[0]?.scenarios).toHaveLength(1);
    expect(store.hasConsent(USER, ["future_self_demo"])).toBe(true);
    const exported = store.exportData(USER);
    expect(Object.keys(exported.records).sort()).toEqual(
      ["audit_log", "user_consent", "user_decisions", "user_evidence", "user_reflections", "user_scenarios"].sort(),
    );
    expect(exported.records.user_evidence).toHaveLength(1); // library records are not user data
  });

  it("refuses reflections without consent (403) or for unknown decisions (404), like the API", () => {
    const now = new Date().toISOString();
    expect(() => store.addReflection({ user_id: USER, decision_id: "dec-000000000001", created_at: now })).toThrowError(
      expect.objectContaining({ status: 403 }),
    );
    store.consent({ user_id: USER, purpose: "reflection_storage", granted: true, created_at: now });
    expect(() => store.addReflection({ user_id: USER, decision_id: "nope", created_at: now })).toThrowError(
      expect.objectContaining({ status: 404 }),
    );
    store.addDecision(request, response);
    store.addReflection({ user_id: USER, decision_id: "dec-000000000001", what_happened: "took A", created_at: now });
    expect(store.journey(USER).reflections).toHaveLength(1);
  });

  it("delete removes everything and reports counts", () => {
    store.addDecision(request, response);
    const r = store.deleteData(USER);
    expect(r.deleted.user_decisions).toBe(1);
    expect(r.deleted.user_scenarios).toBe(1);
    expect(store.journey(USER).decisions).toHaveLength(0);
    expect(window.localStorage.getItem(`${store.STORAGE_PREFIX}${USER}`)).toBeNull();
  });

  it("data never leaves the device: it is keyed per user in localStorage only", () => {
    store.addDecision(request, response);
    expect(window.localStorage.getItem(`${store.STORAGE_PREFIX}${USER}`)).toContain("dec-000000000001");
    expect(store.journey("someone-else").decisions).toHaveLength(0);
  });
});
