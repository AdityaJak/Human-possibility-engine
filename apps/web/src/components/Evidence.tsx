import type { EvidenceRecord } from "../types/contracts";
import { EVIDENCE_META, FUTURE_SELF_LEGEND, VALIDATION_LABEL, evidenceKindOf } from "../lib/evidence";
import type { EvidenceKind } from "../lib/evidence";

export function EvidenceChip({ kind }: { kind: EvidenceKind }) {
  const meta = EVIDENCE_META[kind];
  return (
    <span className={`ev ev--${kind}`} title={meta.description} data-evidence-kind={kind}>
      {meta.label}
    </span>
  );
}

export function EvidenceLegend({ kinds = FUTURE_SELF_LEGEND }: { kinds?: EvidenceKind[] }) {
  return (
    <section aria-labelledby="ev-legend-title" className="card card--flat">
      <h3 id="ev-legend-title">How to read evidence</h3>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 8 }}>
        {kinds.map((k) => (
          <li key={k} className="row" style={{ alignItems: "baseline" }}>
            <EvidenceChip kind={k} />
            <span className="muted" style={{ fontSize: "0.88rem" }}>
              {EVIDENCE_META[k].description}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function EvidenceTable({ records }: { records: EvidenceRecord[] }) {
  if (!records.length) return <p className="muted">No evidence records were returned.</p>;
  return (
    <div className="table-wrap" tabIndex={0}>
      <table>
        <caption className="sr-only">Evidence ledger</caption>
        <thead>
          <tr>
            <th scope="col">Type</th>
            <th scope="col">Claim</th>
            <th scope="col">Source / reference</th>
            <th scope="col">Applicability</th>
            <th scope="col">Uncertainty</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => {
            const kind = evidenceKindOf(r.source_type, r.validation_status);
            return (
              <tr key={r.evidence_id} id={`evidence-${r.evidence_id}`}>
                <td>
                  <EvidenceChip kind={kind} />
                </td>
                <td>{r.claim}</td>
                <td>
                  {r.citation ?? r.data_ref ?? <span className="muted">—</span>}
                  <div className="muted mono" style={{ fontSize: "0.72rem" }}>
                    {r.evidence_id}
                  </div>
                </td>
                <td>{r.applicability || <span className="muted">—</span>}</td>
                <td>{r.uncertainty || <span className="muted">—</span>}</td>
                <td>
                  <span className="muted">{VALIDATION_LABEL[r.validation_status]}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
