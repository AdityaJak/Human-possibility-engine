# Deployment

> Synthetic research prototype; not validated for predicting human futures.
> Nothing in this document is deployed yet. Every path below needs the founder's own hosting account;
> the repository contains verified configuration and exact steps, not a running service.

Three independent paths. A is fully automatic once GitHub Pages is enabled; B and C need an account
each and take about ten minutes.

| Path | What it hosts | Needs | Live simulation? | Future Self? |
|---|---|---|---|---|
| **A. GitHub Pages (static)** | Web app + JSON pre-rendered from the committed stored real outputs | GitHub Pages enabled for the repo | No (stored runs replay) | Yes — the repository's own Python engine runs in the browser (Pyodide) |
| **B. Render (API)** | FastAPI backend in **read-only** mode | Render account (Starter plan for a persistent disk; Free works without one) | No (405 by design) | Yes, server-side under rate limits |
| **C. Vercel (web)** | Web app talking to B | Vercel account | — | — |

Live simulation jobs and experiment runs are **intentionally not exposed** on any public path: they
are CPU-heavy, unauthenticated, and would let anyone write to research state. Reviewers who want live
runs clone the repository and run `make demo` (one command, no API keys).

## A. GitHub Pages — static site (automatic)

The workflow `.github/workflows/pages.yml` (lane *static-site*) builds the web app in static mode on
every push to `claude/human-possibility-engine-yjujoc`: it pre-renders every read endpoint from
`reports/` with the real API code (`scripts/export_static_api.py`), builds the Python wheel the browser
loads for Future Self, and publishes `apps/web/dist`.

Founder action (once), only if the workflow's "Setup Pages" step fails with a permissions error:

1. GitHub → repository → **Settings → Pages**.
2. **Build and deployment → Source: GitHub Actions**. Save.
3. Re-run the failed `pages` workflow (Actions → pages → Re-run all jobs).

URL: `https://adityajak.github.io/Human-possibility-engine/` (the workflow prints it; it is live only
after the first successful deploy).

What it serves: the Living Society replay, Agent Inspector, Proof of Learning, Twin-society
experiment, Evidence and Benchmarks pages from stored real outputs; the complete Future Self flow with
scenarios generated in the browser; user-private records stay in the visitor's browser storage. Buttons
that would start a live run show a notice instead.

## B. Render — the API, read-only (`render.yaml`)

1. <https://dashboard.render.com> → **New → Blueprint**.
2. Connect GitHub and pick **AdityaJak/Human-possibility-engine**; branch
   **claude/human-possibility-engine-yjujoc**. Render reads `render.yaml` at the repo root.
3. Review the single service `hpe-api` (Docker, `infra/docker/api.Dockerfile`, health check `/health`,
   1 GB disk at `/data`, plan **starter**). Click **Apply**. First build takes a few minutes (the image
   bakes `reports/` so the research lab has data immediately).
4. After the service is live, open it → **Environment** → set **`HPE_CORS_ORIGINS`** to the exact
   origin(s) of your web app, comma-separated and without trailing slashes, e.g.
   `https://<project>.vercel.app,https://adityajak.github.io`. Save (Render redeploys).
5. Note the service URL, `https://hpe-api-<hash>.onrender.com`.

Free plan instead: in `render.yaml` set `plan: free`, delete the `disk:` block, and set
`HPE_DATA_DIR=/app/data`. Free instances spin down after ~15 minutes idle (first request then waits
for a cold start) and cannot keep a disk, so pseudonymous demo records vanish on redeploy.

Unverified against a live Render account (no credentials in this environment): disk mount ownership
for the non-root container user. If `/health` reports healthy but Future Self POSTs return 500 with a
SQLite "unable to open database" error, set `HPE_DATA_DIR=/app/data` (ephemeral) or run the container
as root for the disk — and tell us.

Equivalents: **Fly.io** — `fly launch --dockerfile infra/docker/api.Dockerfile`, then
`fly volumes create hpe_data --size 1`, mount at `/data`, and `fly secrets set HPE_READ_ONLY=1 HPE_DATA_DIR=/data HPE_CORS_ORIGINS=...`.
**Railway** — new project from the GitHub repo, set the Dockerfile path to
`infra/docker/api.Dockerfile`, add a volume at `/data`, and the same variables. All three set `PORT`;
the image honours it.

## C. Vercel — the web app (`apps/web/vercel.json`)

1. <https://vercel.com/new> → **Import Git Repository** → AdityaJak/Human-possibility-engine.
2. **Root Directory: `apps/web`** (Edit → select). Framework preset: Vite (auto-detected from
   `vercel.json`). Build command `npm run build`, output `dist` (already in `vercel.json`).
3. **Environment Variables**: `VITE_API_BASE_URL` = the Render URL from B (no trailing slash).
   Leave `VITE_API_MODE` unset (live API mode) and `VITE_USE_MOCKS` unset.
4. **Production Branch** (Settings → Git): `claude/human-possibility-engine-yjujoc`.
5. Deploy. Then go back to Render (B, step 4) and add the Vercel origin to `HPE_CORS_ORIGINS`.

Custom domains: add them in Vercel (web) and Render (API) and update `HPE_CORS_ORIGINS` and
`VITE_API_BASE_URL` to match — CORS is matched on exact origin.

## Security posture of a public deployment

* `HPE_READ_ONLY=1`: `POST /runs`, `/runs/{id}/step`, `/runs/{id}/control`, `/worlds*`,
  `/experiments`, `/experiments/{id}/run` return **405** with an explanatory detail; nothing can
  start a simulation or alter research artifacts. `GET /deployment` shows the posture.
* Future Self POSTs stay available, behind per-client token buckets (`HPE_RATE_LIMIT_PER_MINUTE=20`,
  `HPE_RATE_LIMIT_BURST=5` in the blueprint), with input size limits and the no-probability validator.
  The container runs uvicorn with `--proxy-headers --forwarded-allow-ips='*'` so the limiter sees the
  real client address behind the platform proxy.
* No authentication exists: the Future Self data is **pseudonymous demo data only**; the UI says so
  and offers export/delete. Do not direct real users with sensitive situations to this deployment.
* No secrets are required anywhere; `HPE_LLM_PROVIDER=none` disables the optional narration adapter.
* `reports/` in the image contains only synthetic outputs and committed, licence-checked derived
  metrics — never raw external datasets or personal data (`docs/DATA_SOURCES.md`).

## Cost notes

Render Starter ≈ US$7/month plus ≈ US$0.25/GB-month for the disk at the time of writing (check
Render's pricing page); Free is US$0 with cold starts. Vercel Hobby and GitHub Pages are free for a
public repository. The API makes no paid calls of any kind.

## Post-deploy verification

```bash
API=https://hpe-api-<hash>.onrender.com          # from Render
WEB=https://<project>.vercel.app                 # from Vercel (or https://adityajak.github.io/Human-possibility-engine)

curl -s $API/health                              # {"status":"ok",...,"offline_demo_runs":1}
curl -s $API/deployment | python3 -m json.tool   # "read_only": true
curl -s -o /dev/null -w "%{http_code}\n" -X POST $API/runs -H 'content-type: application/json' \
     -d '{"config":{"seed":1,"n_agents":5,"years":1},"mode":"interactive"}'     # 405
curl -s $API/runs | python3 -c 'import sys,json; print([r["run_id"] for r in json.load(sys.stdin)])'
curl -s $API/experiments | python3 -c 'import sys,json; print([(e["experiment_id"], e["status"]) for e in json.load(sys.stdin)])'
curl -s $API/research/claims | python3 -c 'import sys,json; print(len(json.load(sys.stdin)), "claims")'
# CORS pairing: the preflight must echo your web origin
curl -s -i -X OPTIONS $API/scenarios -H "Origin: $WEB" -H 'Access-Control-Request-Method: POST' | grep -i access-control-allow-origin
# Future Self end to end (consent required; no probabilities in the output)
curl -s -X POST $API/scenarios -H 'content-type: application/json' -d '{
  "user_id":"verify","consent_given":true,"question":"Take the higher-paying job, stay, or start a startup?",
  "options":[{"option_id":"a","label":"Higher-paying job","kind":"new_job","inputs":{"monthly_income":6000}},
             {"option_id":"b","label":"Stay","kind":"current_path","inputs":{"monthly_income":5000}}],
  "values":[{"key":"income","label":"Income","weight":0.7}]}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d["scenarios"]), "scenarios;", d["scenarios"][0]["label"])'
curl -s -X POST $API/privacy/delete -H 'content-type: application/json' -d '{"user_id":"verify"}'
```

Then open `$WEB` in a browser and walk `/demo`; the footer's API status chip must read **ok**.
