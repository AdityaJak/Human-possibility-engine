# ADR-0007: License and repository visibility
Status: accepted for visibility (2026-11-04); license still open — awaiting founder decision

**Facts (verified 2026-11-04).** The GitHub repository `AdityaJak/Human-possibility-engine` already existed with visibility **public**. The spec asks for a PRIVATE repository until the founder authorises public release. This session has no tool to change repository visibility.

**Decision.** The founder was asked explicitly and on 2026-11-04 authorised pushing to the existing public repository. No license is assigned (all rights reserved by default) until the founder chooses one. No secrets, real-user data or unlicensed datasets are committed (`.gitignore` excludes `data/`, `.env`, `datasets/external/`).
