# AgentFlow Agent Guidance

## Development Workflow

- GitHub Issues are the active backlog. `BACKLOG.md` is historical context only.
- Work one issue per branch and open a pull request when verification passes.
- Do not deploy directly from a development run. Deployment happens after a PR is reviewed and merged.
- Request Codex cloud review on orchestrator-created pull requests with `@codex review`.
- Keep the local open-source proxy separate from the private managed optimizer server.

## Review Guidelines

- Prioritize correctness, local-first privacy, and provider compatibility over savings.
- Treat accidental request-body logging, credential exposure, or LAN exposure of the provider proxy as P1 or higher.
- Check that local proxy changes do not add billing, tenant, account, or hosted-server behavior.
- Verify that routing, crunching, and caching decisions remain auditable through machine-readable metadata.
- For dashboard/API changes, look for stale served assets, broken JSON shape compatibility, and misleading cost/savings math.
