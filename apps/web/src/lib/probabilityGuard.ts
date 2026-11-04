/**
 * Guard against unsupported numeric probabilities in illustrative / narrative text.
 *
 * Spec §19 / §36D: the Future Self experience must never show individualized success odds
 * such as "82% chance of success". Calculations with explicit formulas and units are shown in
 * a separate, structured panel and are NOT passed through this guard.
 */

const PATTERNS: RegExp[] = [
  /\b\d{1,3}(?:[.,]\d+)?\s?(?:%|percent|per\s?cent)(?=\W|$)/gi,
  /\b\d+\s?(?:in|out of)\s?\d+\s+(?:chance|odds|probability|likelihood)\b/gi,
  /\b(?:\d+\s?(?:in|out of)\s?\d+)\s+(?:people|users|founders|startups)\s+(?:succeed|fail|regret)\b/gi,
  /\b(?:chance|probability|likelihood|odds)\s+(?:of|that)\s+[^.]{0,40}?\b(?:is|are|=)\s*(?:about\s+|around\s+|roughly\s+)?\d+(?:[.,]\d+)?\b/gi,
  /\b\d+(?:[.,]\d+)?\s?:\s?\d+\s+odds\b/gi,
  /\bp\s?=\s?0?\.\d+/gi,
];

export function findProbabilityClaims(text: string): string[] {
  if (!text) return [];
  const hits: string[] = [];
  for (const re of PATTERNS) {
    re.lastIndex = 0;
    for (const m of text.matchAll(re)) {
      hits.push(m[0].trim());
    }
  }
  return Array.from(new Set(hits));
}

export const REDACTION = "[unsupported probability removed]";

/** Replace any probability-like phrase with a visible redaction marker. */
export function redactProbabilityClaims(text: string): { text: string; hits: string[] } {
  const hits = findProbabilityClaims(text);
  if (!hits.length) return { text, hits };
  let out = text;
  for (const re of PATTERNS) {
    re.lastIndex = 0;
    out = out.replace(re, REDACTION);
  }
  return { text: out, hits };
}
