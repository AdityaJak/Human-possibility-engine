import type { Scenario } from "../types/contracts";
import { IllustrativeBadge } from "./Badges";
import { redactProbabilityClaims } from "../lib/probabilityGuard";
import { titleCase } from "../lib/format";

const DIR_SYMBOL: Record<string, { sym: string; label: string }> = {
  likely_better: { sym: "↑", label: "may serve this value better" },
  likely_worse: { sym: "↓", label: "may serve this value worse" },
  mixed: { sym: "↕", label: "mixed effects on this value" },
  unclear: { sym: "?", label: "unclear effect on this value" },
};

/** Guarded text: unsupported numeric probabilities are replaced with a visible marker. */
export function GuardedText({ text, as: Tag = "p" }: { text: string; as?: "p" | "span" }) {
  const { text: safe, hits } = redactProbabilityClaims(text);
  return (
    <>
      <Tag>{safe}</Tag>
      {hits.length > 0 && (
        <p className="notice notice--bad" role="alert" data-testid="probability-guard">
          Removed {hits.length} unsupported numeric probabilit{hits.length === 1 ? "y" : "ies"} from this illustrative text. Future Self never shows
          individualized odds.
        </p>
      )}
    </>
  );
}

function hueFor(id: string): number {
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return h;
}

/** Abstract generated artwork — explicitly an illustrative visualization, never a likeness. */
function ScenarioArt({ id, title }: { id: string; title: string }) {
  const h = hueFor(id);
  return (
    <figure className="scenario-card__art" aria-label={`Illustrative visualization for ${title}`}>
      <svg viewBox="0 0 300 96" width="100%" height="96" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id={`g-${id}`} x1="0" x2="1">
            <stop offset="0" stopColor={`hsl(${h} 60% 62%)`} />
            <stop offset="1" stopColor={`hsl(${(h + 60) % 360} 55% 48%)`} />
          </linearGradient>
        </defs>
        <rect width="300" height="96" fill={`url(#g-${id})`} />
        <path d={`M0 80 C 60 ${40 + (h % 30)}, 120 ${70 - (h % 20)}, 180 50 S 260 ${20 + (h % 25)}, 300 30 L300 96 L0 96 Z`} fill="rgb(255 255 255 / 22%)" />
        <path d="M0 90 C 80 70, 160 86, 300 60" stroke="rgb(255 255 255 / 70%)" strokeWidth="2" fill="none" strokeDasharray="4 6" />
      </svg>
      <figcaption>Illustrative visualization</figcaption>
    </figure>
  );
}

export function ScenarioCard({ scenario, valueLabels = {} }: { scenario: Scenario; valueLabels?: Record<string, string> }) {
  return (
    <article className="card scenario-card" aria-labelledby={`sc-${scenario.scenario_id}`} data-testid="scenario-card">
      <div className="row">
        <IllustrativeBadge />
        <span className="spacer" />
        {scenario.narrative_source === "llm_external" && (
          <span className="chip chip--warn" title="Narration produced by an external language model behind an evidence boundary.">
            External LLM narration
          </span>
        )}
      </div>
      <ScenarioArt id={scenario.scenario_id} title={scenario.title} />
      <h3 id={`sc-${scenario.scenario_id}`}>{scenario.title}</h3>
      <GuardedText text={scenario.narrative} />
      {scenario.tradeoffs && scenario.tradeoffs.length > 0 && (
        <section aria-label="Trade-offs against your values">
          <h4 style={{ fontSize: "0.9rem", marginBottom: 6 }}>Trade-offs against your values</h4>
          <ul className="tradeoffs">
            {scenario.tradeoffs.map((t) => {
              const d = DIR_SYMBOL[t.direction] ?? { sym: "?", label: t.direction };
              const rationale = redactProbabilityClaims(t.rationale).text;
              return (
                <li key={t.value_key}>
                  <span className={`dir dir--${t.direction}`} aria-label={d.label} title={d.label}>
                    {d.sym}
                  </span>
                  <span>
                    <strong>{valueLabels[t.value_key] ?? titleCase(t.value_key)}</strong> — {rationale}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {scenario.cannot_predict && scenario.cannot_predict.length > 0 && (
        <p className="muted" style={{ fontSize: "0.86rem", marginTop: 4 }}>
          <span className="ev ev--insufficient_evidence">Insufficient evidence</span> We cannot predict: {scenario.cannot_predict.join("; ")}.
        </p>
      )}
      <p className="muted" style={{ fontSize: "0.82rem", marginTop: "auto" }}>
        {scenario.disclaimer ?? "Illustrative scenario conditioned on the assumptions listed. Not a prediction."}
      </p>
    </article>
  );
}
