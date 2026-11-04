/**
 * Browser-local user-private store for the static deployment (lane pyodide-futures).
 *
 * Mirrors `hpe.storage.UserRepository` + the API's journey/export/delete semantics, but every record
 * lives only in this browser's localStorage under `hpe.user.<user_id>` and never leaves the device.
 * The API's error codes are preserved: reflections need `reflection_storage` consent (403) and an
 * existing decision (404). Storage access is wrapped in try/catch so private windows degrade to an
 * in-memory session store instead of crashing.
 */

import type { ConsentRecord, DecisionRequest, EvidenceRecord, Reflection, Scenario, ScenarioResponse } from "../../types/contracts";

export const STORAGE_PREFIX = "hpe.user.";

/** Same table names the API's SQLite namespace uses, so exports look identical in shape. */
export const USER_PRIVATE_TABLES = ["user_consent", "user_decisions", "user_scenarios", "user_evidence", "user_reflections"] as const;

export interface StoredDecision {
  decision_id: string;
  created_at: string;
  request: DecisionRequest;
  scenarios: Scenario[];
  evidence: EvidenceRecord[];
}

export interface UserStore {
  user_id: string;
  consents: ConsentRecord[];
  decisions: StoredDecision[];
  reflections: Reflection[];
  audit_log: Array<{ action: string; at: string; detail?: Record<string, unknown> }>;
}

export class StoreError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "StoreError";
    this.status = status;
  }
}

const memoryFallback = new Map<string, string>();

function backend(): Pick<Storage, "getItem" | "setItem" | "removeItem"> {
  try {
    if (typeof localStorage !== "undefined") {
      const probe = `${STORAGE_PREFIX}__probe`;
      localStorage.setItem(probe, "1");
      localStorage.removeItem(probe);
      return localStorage;
    }
  } catch {
    /* private mode or blocked storage: fall through */
  }
  return {
    getItem: (k) => memoryFallback.get(k) ?? null,
    setItem: (k, v) => void memoryFallback.set(k, v),
    removeItem: (k) => void memoryFallback.delete(k),
  };
}

function key(userId: string): string {
  return `${STORAGE_PREFIX}${userId}`;
}

function fresh(userId: string): UserStore {
  return { user_id: userId, consents: [], decisions: [], reflections: [], audit_log: [] };
}

export function load(userId: string): UserStore {
  try {
    const raw = backend().getItem(key(userId));
    if (!raw) return fresh(userId);
    const parsed = JSON.parse(raw) as Partial<UserStore>;
    return { ...fresh(userId), ...parsed, user_id: userId };
  } catch {
    return fresh(userId);
  }
}

function save(store: UserStore): void {
  try {
    backend().setItem(key(store.user_id), JSON.stringify(store));
  } catch {
    /* quota or blocked: keep going; the in-memory copy served this call */
  }
}

function audit(store: UserStore, action: string, detail?: Record<string, unknown>): void {
  store.audit_log.push({ action, at: new Date().toISOString(), ...(detail ? { detail } : {}) });
}

// ------------------------------------------------------------------ consent

export function consent(record: ConsentRecord): ConsentRecord {
  const store = load(record.user_id);
  store.consents.push(record);
  audit(store, record.granted ? "consent.grant" : "consent.revoke", { purpose: record.purpose });
  save(store);
  return record;
}

/** Latest record per purpose wins; true if any listed purpose is currently granted (as the API). */
export function hasConsent(userId: string, purposes?: readonly string[]): boolean {
  const latest = new Map<string, boolean>();
  for (const c of load(userId).consents) latest.set(c.purpose, c.granted);
  if (!purposes) return [...latest.values()].some(Boolean);
  return purposes.some((p) => latest.get(p) === true);
}

// ------------------------------------------------------------------ decisions

export function addDecision(request: DecisionRequest, response: ScenarioResponse): void {
  const store = load(request.user_id);
  if (!hasConsent(request.user_id, ["future_self_demo"])) {
    store.consents.push({ user_id: request.user_id, purpose: "future_self_demo", granted: true, created_at: new Date().toISOString() });
    audit(store, "consent.implicit_from_request");
  }
  const userEvidence = response.evidence.filter((e) => e.evidence_id.startsWith(response.decision_id));
  const existing = store.decisions.findIndex((d) => d.decision_id === response.decision_id);
  const rec: StoredDecision = {
    decision_id: response.decision_id,
    created_at: new Date().toISOString(),
    request,
    scenarios: response.scenarios,
    evidence: userEvidence,
  };
  if (existing >= 0) store.decisions[existing] = rec;
  else store.decisions.push(rec);
  audit(store, "scenarios.generate", {
    decision_id: response.decision_id,
    n_scenarios: response.scenarios.length,
    safety: response.safety?.category ?? "none",
  });
  save(store);
}

/** Same shape as the API's `ctx.users.decisions(user_id)`. */
export function decisions(userId: string): Array<{ decision_id: string; created_at: string; request: DecisionRequest; scenarios: Scenario[] }> {
  return load(userId).decisions.map(({ decision_id, created_at, request, scenarios }) => ({ decision_id, created_at, request, scenarios }));
}

// ------------------------------------------------------------------ reflections

export function addReflection(reflection: Reflection): Reflection {
  const store = load(reflection.user_id);
  if (!hasConsent(reflection.user_id, ["future_self_demo", "reflection_storage"])) {
    throw new StoreError("no active consent for this user; grant consent before storing reflections", 403);
  }
  if (!store.decisions.some((d) => d.decision_id === reflection.decision_id)) {
    throw new StoreError(`decision '${reflection.decision_id}' not found for this user`, 404);
  }
  store.reflections.push(reflection);
  audit(store, "reflection.add", { decision_id: reflection.decision_id });
  save(store);
  return reflection;
}

export function journey(userId: string): { user_id: string; decisions: ReturnType<typeof decisions>; reflections: Reflection[] } {
  return { user_id: userId, decisions: decisions(userId), reflections: load(userId).reflections };
}

// ------------------------------------------------------------------ export / delete

/** Same top-level keys as the API's PrivacyExport.records (one list per private table + audit_log). */
export function exportData(userId: string): { user_id: string; exported_at: string; records: Record<string, Array<Record<string, unknown>>> } {
  const store = load(userId);
  const records: Record<string, Array<Record<string, unknown>>> = {
    user_consent: store.consents.map((c, i) => ({ id: i + 1, ...c })),
    user_decisions: store.decisions.map((d) => ({ decision_id: d.decision_id, user_id: userId, request_json: JSON.stringify(d.request), created_at: d.created_at })),
    user_scenarios: store.decisions.flatMap((d) => d.scenarios.map((s) => ({ scenario_id: s.scenario_id, decision_id: d.decision_id, user_id: userId, scenario_json: JSON.stringify(s), created_at: d.created_at }))),
    user_evidence: store.decisions.flatMap((d) => d.evidence.map((e) => ({ evidence_id: e.evidence_id, decision_id: d.decision_id, user_id: userId, record_json: JSON.stringify(e), created_at: d.created_at }))),
    user_reflections: store.reflections.map((r, i) => ({ id: i + 1, user_id: userId, decision_id: r.decision_id, reflection_json: JSON.stringify(r), created_at: r.created_at ?? "" })),
    audit_log: store.audit_log.map((a) => ({ ...a, subject: userId })),
  };
  audit(store, "privacy.export");
  save(store);
  return { user_id: userId, exported_at: new Date().toISOString(), records };
}

export function deleteData(userId: string): { user_id: string; deleted: Record<string, number> } {
  const store = load(userId);
  const deleted: Record<string, number> = {
    user_consent: store.consents.length,
    user_decisions: store.decisions.length,
    user_scenarios: store.decisions.reduce((n, d) => n + d.scenarios.length, 0),
    user_evidence: store.decisions.reduce((n, d) => n + d.evidence.length, 0),
    user_reflections: store.reflections.length,
  };
  try {
    backend().removeItem(key(userId));
  } catch {
    /* nothing to do */
  }
  return { user_id: userId, deleted };
}
