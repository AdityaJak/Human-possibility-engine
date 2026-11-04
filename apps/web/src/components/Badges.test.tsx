import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LabBanner, MockBanner } from "./Badges";

afterEach(() => vi.unstubAllEnvs());

describe("MockBanner", () => {
  it("is hidden by default (real API mode)", () => {
    vi.stubEnv("VITE_USE_MOCKS", "");
    render(<MockBanner />);
    expect(screen.queryByTestId("mock-banner")).toBeNull();
  });

  it("is hidden for any value other than '1'", () => {
    vi.stubEnv("VITE_USE_MOCKS", "0");
    render(<MockBanner />);
    expect(screen.queryByTestId("mock-banner")).toBeNull();
  });

  it("appears with VITE_USE_MOCKS=1", () => {
    vi.stubEnv("VITE_USE_MOCKS", "1");
    render(<MockBanner />);
    expect(screen.getByTestId("mock-banner")).toHaveTextContent("MOCK DATA — not real results");
  });
});

describe("LabBanner", () => {
  it("states that the lab is a synthetic simulation", () => {
    render(<LabBanner />);
    expect(screen.getByText("Synthetic simulation — not real people")).toBeInTheDocument();
  });
});
