import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, buildQuery } from "./client";
import { apiBaseUrl, isMockMode } from "./config";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("api client", () => {
  it("calls the configured base URL and parses JSON", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "http://example.test:9000/");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { status: "ok", version: "1", contract_version: "0.1.0" }));
    vi.stubGlobal("fetch", fetchMock);
    const h = await api.health();
    expect(h.status).toBe("ok");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://example.test:9000/health");
  });

  it("defaults to the local API", () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    expect(apiBaseUrl()).toBe("http://127.0.0.1:8000");
  });

  it("raises ApiError with FastAPI detail on HTTP errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(404, { detail: "run not found" })));
    await expect(api.getRun("nope")).rejects.toMatchObject({ name: "ApiError", status: 404, message: "run not found" });
  });

  it("summarises FastAPI validation errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(422, { detail: [{ loc: ["body", "x"], msg: "field required", type: "missing" }] })));
    const err = await api.createRun({ config: {} }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).message).toContain("field required");
  });

  it("reports network failures as status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = (await api.claims().catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.isNetwork).toBe(true);
    expect(err.message).toMatch(/cannot reach the api/i);
  });

  it("sends JSON bodies for POST", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { user_id: "u", deleted: {} }));
    vi.stubGlobal("fetch", fetchMock);
    await api.deleteData("u");
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ user_id: "u" });
  });

  it("encodes list query params as repeated keys", () => {
    expect(buildQuery({ types: ["birth", "death"], limit: 5, empty: undefined })).toBe("?limit=5&types=birth&types=death");
    expect(buildQuery({})).toBe("");
  });

  it("is in real-API mode unless VITE_USE_MOCKS is exactly '1'", () => {
    vi.stubEnv("VITE_USE_MOCKS", "");
    expect(isMockMode()).toBe(false);
    vi.stubEnv("VITE_USE_MOCKS", "true");
    expect(isMockMode()).toBe(false);
    vi.stubEnv("VITE_USE_MOCKS", "1");
    expect(isMockMode()).toBe(true);
  });
});
