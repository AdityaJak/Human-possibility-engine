/**
 * Template-based Future Self "conversation". No language model is involved: replies are
 * assembled deterministically from the scenario's own narrative, trade-offs, uncertainties
 * and reversible experiments. Every reply carries a simulation disclosure and passes the
 * probability guard.
 */

import type { Scenario } from "../types/contracts";
import { redactProbabilityClaims } from "./probabilityGuard";

export const CONVERSATION_DISCLOSURE =
  "Simulated conversation. This is a template role-play built from an illustrative scenario. It is not your real future self, it does not know your future, and it cannot make this decision for you.";

export const SUGGESTED_QUESTIONS = [
  "What should I try first?",
  "What are the biggest risks?",
  "How does this fit what matters to me?",
  "What about money?",
  "Would I be happier?",
];

function firstSentence(text: string): string {
  const m = /^(.+?[.!?])(\s|$)/.exec(text.trim());
  return (m?.[1] ?? text).trim();
}

function list(items: string[], max = 3): string {
  const xs = items.slice(0, max);
  if (xs.length <= 1) return xs.join("");
  return `${xs.slice(0, -1).join("; ")}; and ${xs[xs.length - 1]}`;
}

export function simulatedReply(question: string, scenario: Scenario, optionLabel: string, valueLabels: Record<string, string> = {}): string {
  const q = question.toLowerCase();
  const prefix = `As an illustrative scenario for “${optionLabel}” — not your real future self: `;
  const exps = scenario.reversible_experiments ?? [];
  const unc = scenario.uncertainties ?? [];
  const cannot = scenario.cannot_predict ?? [];
  const trade = scenario.tradeoffs ?? [];
  const calcs = scenario.calculations ?? [];
  let body: string;

  if (/(should i|what do i do|decide|choose|which one|try first|next step|start)/.test(q)) {
    body = exps.length
      ? `I can't choose for you, and nobody knows how this would turn out. A small, reversible way to learn more first: ${exps[0]?.title} — ${exps[0]?.description} (about ${exps[0]?.duration_days} days).`
      : "I can't choose for you, and nobody knows how this would turn out. Consider a small, reversible step that would tell you more before committing.";
  } else if (/(money|income|afford|saving|runway|cost|pay|salary|financ)/.test(q)) {
    const computed = calcs.filter((c) => c.result !== null);
    body = computed.length
      ? `Using only the numbers you entered: ${list(computed.map((c) => `${c.name} ≈ ${Number(c.result).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${c.unit} (${c.formula})`))}. These are arithmetic on your inputs, not forecasts. For a high-stakes financial decision, a qualified professional can help.`
      : "You didn't give numbers for this path, so I can't calculate anything. Adding income, expenses and savings in the Decision Studio would let the tool show a transparent runway calculation — still not a forecast.";
  } else if (/(risk|worr|afraid|fail|regret|scared|wrong|downside)/.test(q)) {
    const parts = [...unc, ...cannot.map((c) => `we cannot predict ${c}`)];
    body = parts.length
      ? `The honest answer is uncertainty. What this scenario flags: ${list(parts, 4)}.`
      : "This scenario lists no specific uncertainties, which does not mean there are none.";
  } else if (/(happ|fulfil|satisf|feel|enjoy|stress)/.test(q)) {
    body =
      "No one — including this tool — can predict how you would feel. People are often inaccurate when forecasting their own future feelings. What you can do is test the experience in a small way and reflect on it afterwards.";
  } else if (/(value|matter|important|priorit|fit)/.test(q)) {
    body = trade.length
      ? `Against the values you set: ${list(
          trade.map((t) => `${valueLabels[t.value_key] ?? t.value_key} — ${t.direction.replace("_", " ")} (${t.rationale})`),
          4,
        )}.`
      : "This scenario has no value trade-offs listed yet. Adjust your North Star values and regenerate.";
  } else {
    body = `${firstSentence(scenario.narrative)} You could ask me about risks, money, how it fits your values, or what to try first.`;
  }
  return redactProbabilityClaims(prefix + body).text;
}
