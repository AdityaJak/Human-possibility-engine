# Contributing

1. Read [`AGENTS.md`](AGENTS.md) (lane ownership, frozen contracts) and [`docs/CLAIMS_POLICY.md`](docs/CLAIMS_POLICY.md).
2. Setup: `make setup` (Python via uv + web via npm). Everything runs without API keys.
3. Before opening a PR: `make lint typecheck test contracts-check`.
4. Contract changes (`packages/hpe/contracts/`) need an ADR in `docs/ADRS/` and a `CONTRACT_VERSION` bump, then `make contracts`.
5. Determinism: randomness only from `hpe.core.rng` streams; add a replay test for new stochastic mechanisms.
6. Never commit secrets, real-user data, or external datasets. Results committed under `reports/` must be produced by code in this repository, with a manifest.
7. Report negative results. Do not change a preregistered primary metric after seeing data.
