# Security policy

## Reporting
Please report vulnerabilities privately to the repository owner via GitHub's private vulnerability reporting (Security → Report a vulnerability). Do not open public issues for security problems.

## Current posture (prototype)
* Local-first demo. No authentication: **do not store real personal or sensitive data.** Future Self users are pseudonymous demo users in a local SQLite file under `data/` (gitignored).
* No secrets in the repository. Optional API keys are read from environment variables only (`.env.example`); the optional LLM adapter is disabled by default.
* API protections: request validation (Pydantic), quotas on run size, token-bucket rate limits on expensive endpoints, CORS allow-list.
* Privacy controls: consent records, export, delete, audit log for the user-private namespace.
* Not yet implemented (required before any real-user data): authentication/authorisation, encryption at rest, retention enforcement, incident-response runbook, external security review.
