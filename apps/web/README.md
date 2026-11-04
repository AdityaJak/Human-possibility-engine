# HPE web app (`apps/web`)

React 18 + TypeScript (strict) + Vite + react-router. Charts are hand-built SVG; the society map is a `<canvas>`. No runtime dependencies beyond React and react-router.

```bash
npm ci
npm run dev        # http://127.0.0.1:5173, talks to the API at VITE_API_BASE_URL (default http://127.0.0.1:8000)
npm run lint && npm run typecheck && npm test && npm run build
```

## Data sources — real API by default

All scientific numbers come from the backend API (`docs/INTERFACES.md` §4) via the typed client in `src/api/client.ts`. Types are generated from the Python contracts (`src/types/contracts.ts`, do not edit). When the API is unreachable, pages show an error state; nothing falls back to fake data.

`VITE_USE_MOCKS=1 npm run dev` enables a development-only mock transport (`src/api/mocks/`) with procedurally generated placeholder data. Every mock label contains "MOCK" and a persistent red "MOCK DATA — not real results" banner is shown. The mock module is dead-code-eliminated from production builds made without that variable.

## Routes

| Route | Zone | Purpose |
|---|---|---|
| `/` | — | Mission, built vs unproven, live claim counts |
| `/lab/society?run=<id>` | Research Lab | Map, indicators, events, agent search; interactive runs (play/pause/step/reset/seed) or replay of stored runs |
| `/lab/runs/:runId/agents/:agentId` | Research Lab | Sourced timeline, held-out skill curves with CIs vs chance, parameter-change history, self-estimate vs measured, relationships, memories, decision explanation |
| `/lab/learning?id=<experiment>` | Research Lab | Baseline comparison: learning curves, arm summaries, effects, transfer, ablations, sensitivity |
| `/lab/experiment?id=<experiment>` | Research Lab | Twin-society intervention: design, per-seed dots, paired differences, effects, mechanism check, assumptions & validity, raw log downloads |
| `/lab/evidence` | Research Lab | Claims ledger (`/research/claims`) |
| `/lab/benchmarks` | Research Lab | Measured benchmark records only; "Not yet measured" otherwise |
| `/future/:step` | Future Self | welcome → consent → north-star → decision → futures → evidence → plan → journey → conversation → privacy |
| `/demo` | — | Guided investor walkthrough (narration + deep links, no research numbers) |

## Honesty guards (tested)

- Every Research Lab page shows "Synthetic simulation — not real people"; every scenario card shows "Illustrative scenario".
- `src/lib/probabilityGuard.ts` redacts percentage/odds language from narrative and trade-off text and shows a visible warning.
- Future Self steps after consent are blocked by `ConsentGate` until consent is recorded by `POST /privacy/consent`.
- The Future Self conversation is template-based (no LLM) and carries a permanent simulation disclosure.
- Evidence types (user fact / calculation / external study / assumption / narrative / insufficient evidence) have distinct styles and a legend; simulated outcomes are never offered as Future Self evidence.

## API shape assumptions beyond the frozen contracts

- `GET /journey/{user_id}` → `{ decisions: object[], reflections: Reflection[] }` (decisions may carry `decision_id`, `question`, `created_at`).
- `GET /research/metrics` → free-form summary object (not rendered by any page yet).
- List query params are sent as repeated keys (`?types=birth&types=death`).
- `GET /runs/{id}/frames?every=N` subsamples ticks; it is also called for interactive runs to obtain the `WorldMap`.
- `GET /runs/{id}/download` is linked directly as a file download.
- Learning curves are drawn from `ExperimentInfo.outcomes_sample` grouped by `(arm, metric, age_years | tick)`; transfer panels use observations with `split: "transfer"`. Mechanism checks are rows in `result.ablations`/`result.sensitivity` whose content mentions "mechanism".
