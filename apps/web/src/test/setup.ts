import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import { setTransport } from "../api/client";

afterEach(() => {
  cleanup();
  setTransport(null);
  try {
    window.localStorage.clear();
  } catch {
    /* ignore */
  }
});

// jsdom lacks these browser APIs used by the map and layout.
if (typeof window !== "undefined") {
  if (!window.matchMedia) {
    window.matchMedia = ((q: string) => ({
      matches: false,
      media: q,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    })) as unknown as typeof window.matchMedia;
  }
  window.scrollTo = (() => undefined) as typeof window.scrollTo;
  HTMLCanvasElement.prototype.getContext = (() => null) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}
