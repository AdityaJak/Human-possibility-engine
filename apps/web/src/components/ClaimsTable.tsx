import { useMemo, useState } from "react";
import type { ClaimRecord } from "../types/contracts";
import { ClaimStatusChip } from "./StatusChip";
import { EVIDENCE_KIND_LABEL } from "../lib/labels";

const STATUSES: ClaimRecord["status"][] = ["supported", "not_supported", "inconclusive", "not_tested", "blocked"];

export function ClaimsTable({ claims }: { claims: ClaimRecord[] }) {
  const [filter, setFilter] = useState<ClaimRecord["status"] | "all">("all");
  const counts = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of claims) m.set(c.status, (m.get(c.status) ?? 0) + 1);
    return m;
  }, [claims]);
  const shown = filter === "all" ? claims : claims.filter((c) => c.status === filter);

  return (
    <div className="stack">
      <div className="row" role="group" aria-label="Filter claims by status">
        <button type="button" className="btn btn--sm" aria-pressed={filter === "all"} onClick={() => setFilter("all")}>
          All ({claims.length})
        </button>
        {STATUSES.map((s) => (
          <button key={s} type="button" className="btn btn--sm" aria-pressed={filter === s} onClick={() => setFilter(s)} disabled={!counts.get(s)}>
            <ClaimStatusChip status={s} /> {counts.get(s) ?? 0}
          </button>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <caption className="sr-only">Claims and their evidence</caption>
          <thead>
            <tr>
              <th scope="col">Claim</th>
              <th scope="col">Status</th>
              <th scope="col">Evidence</th>
              <th scope="col">Metric / value</th>
              <th scope="col">Runs &amp; config</th>
              <th scope="col">Limitations</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => (
              <tr key={c.claim_id} data-testid="claim-row">
                <td>
                  <strong>{c.claim}</strong>
                  <div className="muted mono" style={{ fontSize: "0.72rem" }}>
                    {c.claim_id}
                  </div>
                </td>
                <td>
                  <ClaimStatusChip status={c.status} />
                </td>
                <td>{EVIDENCE_KIND_LABEL[c.evidence_kind] ?? c.evidence_kind}</td>
                <td>
                  {c.metric ? <span className="mono">{c.metric}</span> : <span className="muted">—</span>}
                  {c.value && <div>{c.value}</div>}
                </td>
                <td>
                  {c.run_ids && c.run_ids.length > 0 ? (
                    <details>
                      <summary>{c.run_ids.length} run{c.run_ids.length === 1 ? "" : "s"}</summary>
                      <ul className="mono" style={{ fontSize: "0.75rem", paddingLeft: 16, margin: 0 }}>
                        {c.run_ids.map((r) => (
                          <li key={r}>{r}</li>
                        ))}
                      </ul>
                    </details>
                  ) : (
                    <span className="muted">no runs</span>
                  )}
                  {c.config_ref && <div className="mono muted" style={{ fontSize: "0.72rem" }}>{c.config_ref}</div>}
                </td>
                <td>
                  {c.limitations && c.limitations.length > 0 ? (
                    <ul style={{ paddingLeft: 16, margin: 0, fontSize: "0.85rem" }}>
                      {c.limitations.map((l) => (
                        <li key={l}>{l}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="muted">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
