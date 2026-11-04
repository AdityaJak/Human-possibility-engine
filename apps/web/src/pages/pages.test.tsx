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

  it("phase III page shows committed results, gates as not evaluated, and failures", async () => {
    const fixture = {
      disclaimer: "Phase III research results on real human choice data.",
      experiments: [
        {
          experiment_id: "x_demo",
          lane: "director",
          title: "Demo run",
          hypothesis: "h",
          report_partition: "tune",
          path: "reports/phase3/director/x_demo",
          preregistration_hash: "abc",
          families: {
            restless4: {
              role: "counted_primary",
              best_comparator: "pop_gru",
              models: {
                pop_gru: { report_nll: 0.71, ci: [0.69, 0.73], n: 174, top1: 0.7, ece: 0.03, quarters: [0.8, 0.7, 0.7, 0.68], select_nll: 0.71, params: 3000, role: "comparator", lapse: 0, grid_edges: [], fit_cpu_s: 40, causality_violations: [] },
              },
              contrasts: [],
            },
          },
        },
      ],
      registry: { exploratory: [], confirmatory: [], abandoned: [{ date: "2026-11-04", lane: "director", what: "X3-001a", why: "under-converged comparator" }], amendments: [] },
      verdict: null,
    };
    setTransport(async (_m, path) => (path === "/research/phase3" ? fixture : { status: "ok", version: "t", contract_version: "0.1.0" }));
    at("/lab/phase3");
    expect(await screen.findByRole("heading", { name: /population-trained models and individual adaptation/i })).toBeInTheDocument();
    expect(await screen.findByText(/under-converged comparator/)).toBeInTheDocument();
    expect(screen.getAllByText("not evaluated")).toHaveLength(7);
    expect(screen.getByText(/research, not predictions about you/i)).toBeInTheDocument();
  });

  it("lab pages surface API errors instead of fake data", async () => {
    setTransport(async (_m, path) => {
      if (path === "/health") return { status: "ok", version: "t", contract_version: "0.1.0" };
      throw new ApiError("Cannot reach the API at http://127.0.0.1:8000. Is the backend running?", 0, path);
    });
    at("/lab/learning");
    expect(await screen.findByText("Cannot reach the API", { selector: "h2" })).toBeInTheDocument();
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
