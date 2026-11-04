import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { setTransport } from "../../api/client";
import type { RequestOptions } from "../../api/client";
import FutureSelf from "./FutureSelf";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/future/:step" element={<FutureSelf />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("Future Self consent gate", () => {
  it("blocks the North Star step until consent is given", () => {
    renderAt("/future/north-star");
    expect(screen.getByTestId("consent-gate")).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: /your values/i })).toBeNull();
  });

  it("blocks the Decision Studio and scenario steps until consent is given", () => {
    renderAt("/future/decision");
    expect(screen.getByTestId("consent-gate")).toBeInTheDocument();
    expect(screen.queryByLabelText(/what decision are you weighing/i)).toBeNull();
  });

  it("keeps the continue button disabled until both required boxes are checked", () => {
    renderAt("/future/consent");
    const btn = screen.getByRole("button", { name: /i consent/i });
    expect(btn).toBeDisabled();
    const boxes = screen.getAllByRole("checkbox");
    fireEvent.click(boxes[0] as HTMLElement);
    expect(btn).toBeDisabled();
    fireEvent.click(boxes[1] as HTMLElement);
    expect(btn).toBeEnabled();
  });

  it("records consent through the API and then unlocks the flow", async () => {
    const calls: Array<{ path: string; body: unknown }> = [];
    setTransport(async (_m: string, path: string, opts: RequestOptions) => {
      calls.push({ path, body: opts.body });
      return opts.body;
    });
    renderAt("/future/consent");
    const boxes = screen.getAllByRole("checkbox");
    fireEvent.click(boxes[0] as HTMLElement);
    fireEvent.click(boxes[1] as HTMLElement);
    fireEvent.click(screen.getByRole("button", { name: /i consent/i }));
    await waitFor(() => expect(screen.getByRole("list", { name: /your values/i })).toBeInTheDocument());
    expect(calls.filter((c) => c.path === "/privacy/consent")).toHaveLength(2);
    expect(calls[0]?.body).toMatchObject({ purpose: "future_self_demo", granted: true });
  });

  it("stays locked when consent cannot be recorded", async () => {
    setTransport(vi.fn().mockRejectedValue(new Error("server down")));
    renderAt("/future/consent");
    const boxes = screen.getAllByRole("checkbox");
    fireEvent.click(boxes[0] as HTMLElement);
    fireEvent.click(boxes[1] as HTMLElement);
    fireEvent.click(screen.getByRole("button", { name: /i consent/i }));
    await waitFor(() => expect(screen.getByText(/consent could not be recorded/i)).toBeInTheDocument());
    expect(screen.queryByRole("list", { name: /your values/i })).toBeNull();
  });

  it("shows the Future Self disclosure on every step", () => {
    renderAt("/future/welcome");
    expect(screen.getAllByText(/illustrative scenario/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/does not predict your future/i)).toBeInTheDocument();
  });
});
