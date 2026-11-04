import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { setTransport } from "../api/client";
import { ApiError } from "../api/client";
import { AppRoutes } from "../App";

function at(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

describe("routes", () => {
  it("landing page renders mission and honest status even when the API is down", async () => {
    setTransport(async () => {
      throw new ApiError("Cannot reach the API", 0, "x");
    });
    at("/");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/better decisions for your future self/i);
    expect(screen.getByText(/what is built — and what remains unproven/i)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/not reachable right now/i)).toBeInTheDocument());
  });

  it("benchmarks page shows 'Not yet measured' when no records exist", async () => {
    setTransport(async (_m, path) => (path === "/benchmarks" ? [] : { status: "ok", version: "t", contract_version: "0.1.0" }));
    at("/lab/benchmarks");
    expect(await screen.findAllByText("Not yet measured")).not.toHaveLength(0);
    expect(screen.getByText("Synthetic simulation — not real people")).toBeInTheDocument();
  });

  it("lab pages surface API errors instead of fake data", async () => {
    setTransport(async (_m, path) => {
      if (path === "/health") return { status: "ok", version: "t", contract_version: "0.1.0" };
      throw new ApiError("Cannot reach the API at http://127.0.0.1:8000. Is the backend running?", 0, path);
    });
    at("/lab/learning");
    expect(await screen.findByText("Cannot reach the API", { selector: "h3" })).toBeInTheDocument();
  });

  it("evidence page renders claims from the API", async () => {
    setTransport(async (_m, path) =>
      path === "/research/claims"
        ? [{ claim_id: "c", claim: "Developmental learner beats memory-only on held-out tasks", status: "inconclusive", evidence_kind: "synthetic_simulation" }]
        : { status: "ok", version: "t", contract_version: "0.1.0" },
    );
    at("/lab/evidence");
    expect(await screen.findByText(/beats memory-only/)).toBeInTheDocument();
  });

  it("investor demo lists the walkthrough steps", async () => {
    setTransport(async () => []);
    at("/demo");
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent(/what we built/i);
    expect(screen.getAllByRole("button", { name: /twin-society experiment/i }).length).toBeGreaterThan(0);
  });
});
