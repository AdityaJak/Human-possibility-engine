# Security policy

## Reporting
Please report vulnerabilities privately to the repository owner via GitHub's private vulnerability reporting (Security → Report a vulnerability). Do not open public issues for security problems.

## Current posture (prototype)
* Local-first demo. No authentication: **do not store real personal or sensitive data.** Future Self users are pseudonymous demo users in a local SQLite file under `data/` (gitignored).
* No secrets in the repository. Optional API keys are read from environment variables only (`.env.example`); the optional LLM adapter is disabled by default.
* API protections: request validation (Pydantic), quotas on run size, token-bucket rate limits on expensive endpoints, CORS allow-list.
* Privacy controls: consent records, export, delete, audit log for the user-private namespace.
* Not yet implemented (required before any real-user data): authentication/authorisation, encryption at rest, retention enforcement, incident-response runbook, external security review.

## Public deployment posture
* Deploy the API only with `HPE_READ_ONLY=1` (see `docs/DEPLOYMENT.md`): simulation and experiment
  starters answer 405, so an unauthenticated internet cannot consume CPU or alter research artifacts.
  `GET /deployment` reports the posture.
* Future Self stays available behind per-client token buckets (`HPE_RATE_LIMIT_PER_MINUTE`,
  `HPE_RATE_LIMIT_BURST`), input size limits and the no-probability validator; uvicorn runs with
  `--proxy-headers --forwarded-allow-ips='*'` so limits key on the real client address behind the
  platform proxy. There is still no authentication: public Future Self data is pseudonymous demo data.
* CORS is an exact-origin allow-list (`HPE_CORS_ORIGINS`); credentials are never allowed.
* The container runs as a non-root user; the image contains only synthetic outputs and licence-checked
  derived metrics, never raw external datasets, personal data or secrets.
