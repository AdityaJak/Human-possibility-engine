import { api } from "../../api/client";
import { useApi } from "../../api/hooks";
import { Async, EmptyState } from "../../components/AsyncState";
import { LabBanner } from "../../components/Badges";
import { ClaimsTable } from "../../components/ClaimsTable";

export default function EvidencePage() {
  const claims = useApi((s) => api.claims(s), []);
  return (
    <div className="stack">
      <LabBanner>The evidence ledger lists every public claim with its status, the run ids and metric behind it, and its limitations. Negative results are listed too.</LabBanner>
      <header className="page-head">
        <div className="eyebrow">Research Lab · Evidence</div>
        <h1>Every claim, with its evidence</h1>
        <p>
          A claim is only “supported” when a committed run or experiment produced it. Synthetic-simulation results are not evidence about real people; human-dataset results
          are labelled separately.
        </p>
      </header>
      <section className="card">
        <Async
          {...claims}
          loadingLabel="Loading the evidence ledger…"
          isEmpty={(d) => d.length === 0}
          empty={<EmptyState title="No claims recorded yet">Claims appear here once experiments and benchmarks have actually run.</EmptyState>}
        >
          {(d) => <ClaimsTable claims={d} />}
        </Async>
      </section>
    </div>
  );
}
