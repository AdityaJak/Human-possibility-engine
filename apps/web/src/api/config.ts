/** Runtime configuration read from Vite env at call time (so tests can stub it). */

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export function apiBaseUrl(): string {
  const raw = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";
  const url = raw.trim() || DEFAULT_API_BASE_URL;
  return url.replace(/\/+$/, "");
}

/** Mock mode is OFF unless VITE_USE_MOCKS is exactly "1". Real API is the default. */
export function isMockMode(): boolean {
  return import.meta.env.VITE_USE_MOCKS === "1";
}
