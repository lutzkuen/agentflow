# AgentFlow

Local visibility and cost control for coding-agent LLM traffic.

AgentFlow runs on your machine between tools such as Claude Code, Claude CLI, Codex, and LLM provider APIs. It records local usage metadata, shows token and spend behavior in a dashboard, and applies conservative optimizations such as model routing, prompt crunching, and safe local caching.

AgentFlow is local-first:

- provider calls still happen from your machine with your credentials
- prompt and response bodies are not stored by default
- the dashboard is read-only
- routing, crunching, and cache decisions are visible and auditable
- when AgentFlow is unsure, it forwards the request unchanged and records why

## What it helps you answer

- How many tokens did my agents use today?
- Which sessions, apps, or providers drove spend?
- Which calls were routed, crunched, cached, skipped, or errored?
- How much did local optimization save?
- Which policies are active right now?
- Did any local policy file change and need reload?

## Current support

| Surface | Status |
| --- | --- |
| Anthropic `/v1/messages` | Primary supported proxy path |
| OpenAI `/v1/responses` and `/v1/chat/completions` | Supported API-compatible proxy path |
| Codex app-server | Experimental telemetry path; token/cost accounting and optimization are being expanded |
| Dashboard | Read-only local/LAN observability |
| Local policy files | Routing, crunching, cache, review/apply/rollback tools |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run tests with:

```bash
python -m unittest discover -s tests
```

## Quick start: Claude / Anthropic

Start the local Anthropic-compatible proxy:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
agentflow-proxy --provider anthropic --host 127.0.0.1 --port 4000
```

Point Claude-compatible clients at the local proxy. Exact environment variables can differ by client version, but the intended setup is:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_API_KEY"
claude
```

The proxy accepts incoming Anthropic auth headers and forwards them upstream. Do not expose this proxy port outside localhost unless you add your own network/auth boundary.

Open the dashboard:

```text
http://127.0.0.1:4000/agentflow/dashboard
```

## Quick start: OpenAI-compatible clients

Run the OpenAI-compatible proxy on a separate port:

```bash
export OPENAI_API_KEY="sk-..."
agentflow-proxy --provider openai --openai-auth-mode proxy --host 127.0.0.1 --port 4003
```

Example Codex API-compatible usage:

```bash
codex exec --config 'openai_base_url="http://127.0.0.1:4003/v1"' "Reply with ok"
```

Provider modes are intentionally separate. An Anthropic-mode process serves `/v1/messages`; an OpenAI-mode process serves `/v1/responses` and `/v1/chat/completions`.

## Codex app-server telemetry

For Codex OAuth/subscription quota, the API-compatible `openai_base_url` path may not be the right fit. AgentFlow also includes an experimental Codex app-server relay:

```bash
codex app-server --listen ws://127.0.0.1:4014
agentflow-codex-app-proxy --host 127.0.0.1 --port 4013 --upstream ws://127.0.0.1:4014
printf 'Reply with exactly: ok\n' | agentflow-codex-app-client --url ws://127.0.0.1:4013 --cd "$PWD"
```

This path currently focuses on telemetry. It records redacted JSON-RPC method names and size-derived metadata without storing raw prompts by default. Codex token/cost accounting, routing, crunching, and cache support are being expanded incrementally.

To inspect whether recent live Codex turns were routable, crunchable, or cache-eligible, run:

```bash
agentflow-codex-diagnose --db ~/.agentflow/agentflow.sqlite3 --pretty
```

The same read-only report is available from a running dashboard or proxy at:

```text
/agentflow/stats/codex-effectiveness?limit=500
```

The report uses stored metadata, sizes, and decision JSON only. It does not include raw prompts, params, responses, transcripts, or tool payloads.

To measure replay-safe cache opportunity and blockers across recent Anthropic, OpenAI, and Codex metadata, run:

```bash
agentflow-cache-replayability-report --db ~/.agentflow/agentflow.sqlite3 --pretty
```

The report groups local metadata by replay fingerprint and includes stream/tool flags, workflow phase, cacheability bucket, safe invalidation evidence, current cache decision, blocker counts, and projected repeated-call cost. It does not print raw prompts, responses, tool payloads, request IDs, cache keys, file paths, transcripts, or raw session IDs.

To dry-run proposed cache replay pattern rules against recent local traffic without writing cache rows, replaying responses, or calling providers, run:

```bash
agentflow-cache-replay-dry-run proposed-cache-policy.json --db ~/.agentflow/agentflow.sqlite3 --pretty
```

The dry-run reports projected exact and streaming hits, canary holdouts, blocked rows, invalidation-required rows, unsupported source surfaces, stale-risk blockers, rule IDs, candidate IDs, and estimated repeated-call savings from metadata-derived replay fingerprints only.

To measure Anthropic phase-routing opportunity and blockers before changing routing behavior, run:

```bash
agentflow-phase-routing-report --db ~/.agentflow/agentflow.sqlite3 --pretty
```

The report groups recent local metadata by provisional workflow phase and downgrade pair, with projected savings, current routed counts, blocker counts, and risk exclusions. It does not print raw prompts, responses, session IDs, request IDs, file paths, or error bodies.

To dry-run a proposed phase-aware routing policy or managed policy bundle against recent local traffic without writing policy files or changing provider routing, run:

```bash
agentflow-phase-routing-report --db ~/.agentflow/agentflow.sqlite3 --dry-run-policy proposed-routing.yaml --pretty
```

The dry-run reports matched counts, projected candidate counts and savings, exclusions such as thinking, high error rate, stale evidence, unsupported shadow/streaming evidence, missing baseline support, insufficient samples, candidate rule IDs, and metadata-only privacy flags.

To export family-agnostic optimization candidates for local evaluation, run:

```bash
agentflow-optimization-eval-plan --db ~/.agentflow/agentflow.sqlite3 --pretty > eval-plan.json
```

To score an eval plan with local fixture evidence or record explicit blockers without changing live traffic or policy files, run:

```bash
agentflow-optimization-shadow-eval eval-plan.json --db ~/.agentflow/agentflow.sqlite3 --pretty
```

The shadow-eval command writes metadata-only result records. Provider execution is off by default and requires both `--execute` and a positive `--budget-usd`; rows without replayable local inputs are recorded as blocked instead of attempting a call.

## Dashboard

The dashboard shows local usage and optimization behavior, including:

- tokens, spend, savings, and hard-floor estimates
- recent calls and Codex turns
- usage by app, engineer, or session when labels are available
- routing, crunching, cache, and limiter decisions
- error breakdowns
- active local policy state and reload status

The proxy serves the dashboard at `/agentflow/dashboard`. For a separate read-only dashboard process, run:

```bash
agentflow-dashboard --host 0.0.0.0 --port 4002
```

The standalone dashboard reads the same local database. It does not expose provider proxy routes.

## Safety and privacy defaults

- Proxy ports should stay on `127.0.0.1` unless you add your own access control.
- Prompt and response bodies are not stored by default.
- Body logging is only for local debugging:

  ```bash
  export AGENTFLOW_LOG_BODIES=1
  ```

- The LAN dashboard is read-only.
- Tool-call caching is off by default because tool calls often depend on live filesystem state.
- Exact local cache decisions include explicit hit/miss/skip reasons.
- Policy changes are local files and can be reviewed before apply.

## Local policies

AgentFlow can use local YAML-backed policy files for routing, crunching, cache, routing experiments, and safe Codex app-server summary/cache actions. The dashboard shows which policies are loaded and whether any file needs reload.

Common policy commands:

| Command | Purpose |
| --- | --- |
| `agentflow-policy-export --pretty` | Export the current effective local policy bundle |
| `agentflow-policy-validate bundle.json` | Validate a policy bundle before use |
| `agentflow-policy-diff before.json after.json --pretty` | Compare two policy bundles |
| `agentflow-policy-review proposed.json --pretty` | Review changes and warnings before apply |
| `agentflow-policy-fetch-review --url http://127.0.0.1:4100/v1/policy-bundle-recommendation --allow-unauthenticated --pretty` | Fetch an opt-in managed recommendation and review it without applying |
| `agentflow-cache-replayability-report --db ~/.agentflow/agentflow.sqlite3 --pretty` | Measure replay-safe cache opportunity and blockers from local metadata |
| `agentflow-cache-replay-dry-run proposed-cache-policy.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Dry-run cache replay pattern rules against recent local metadata without mutating cache entries |
| `agentflow-optimization-eval-plan --db ~/.agentflow/agentflow.sqlite3 --pretty` | Export metadata-only optimization candidates for local evaluation |
| `agentflow-optimization-shadow-eval eval-plan.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Record metadata-only local shadow-eval pass/fail/blocked/unknown results without provider calls by default |
| `agentflow-optimization-eval-next --db ~/.agentflow/agentflow.sqlite3 --limit 10 --pretty` | Select a bounded highest-value eval queue batch and record local eval evidence without provider calls by default |
| `agentflow-optimization-promotion-report eval-plan.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Score local eval, canary, and holdout evidence into widen/hold/rollback/needs_eval promotion verdicts |
| `agentflow-optimization-promotion-actions promotion-report.json --pretty` | Convert passing promotion verdicts into privacy-safe local rollout-action bundles with explicit omissions |
| `agentflow-optimization-rollout-actions-review --url http://127.0.0.1:4100/v1/optimization-rollout-actions --allow-unauthenticated --pretty` | Review managed eval-gated optimization rollout actions before any local apply step |
| `agentflow-optimization-promotion-canaries-apply promotion-actions.json --config-dir ~/.agentflow --dry-run --pretty` | Preview promotion canary routing edits before writing local YAML files |
| `agentflow-old-context-summary-dry-run proposed.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Dry-run old-context summarization settings against recent local traffic without calling the summary model |
| `agentflow-old-context-summary-impact dry-run.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Compare post-apply old-context summarization metadata against a prior dry-run projection |
| `agentflow-policy-apply proposed.json --dry-run --pretty` | Preview local file writes |
| `agentflow-policy-apply proposed.json --config-dir ~/.agentflow` | Apply reviewed local policy files |
| `agentflow-policy-rollback --config-dir ~/.agentflow --dry-run --pretty` | Preview rollback to the newest backup |
| `agentflow-policy-reload` | Reload changed policies in a running local proxy |
| `agentflow-managed-feedback-status --pretty` | Inspect the local managed feedback queue without printing payloads |
| `agentflow-managed-feedback-flush --limit 5 --dry-run --pretty` | Preview a bounded retry batch for due managed feedback |
| `agentflow-managed-feedback-flush --limit 5 --pretty` | Flush due managed feedback when managed recommendations are enabled |
| `agentflow-managed-pattern-rollups --limit 500 --pretty` | Export metadata-only managed pattern canary cohort outcome rollups for review |
| `agentflow-managed-rollout-actions-review --url http://127.0.0.1:4100/v1/pattern-rollout-actions --allow-unauthenticated --pretty` | Review managed pattern rollout actions against local crunch/cache rule files |
| `agentflow-managed-rollout-actions-dry-run actions.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Estimate rollout action impact against recent local traffic metadata without writing policy files |
| `agentflow-managed-rollout-actions-impact dry-run.json --db ~/.agentflow/agentflow.sqlite3 --pretty` | Compare post-apply metadata against rollout-action dry-run projections |
| `agentflow-managed-rollout-actions-apply actions.json --config-dir ~/.agentflow --dry-run --pretty` | Preview approved rollout action edits before writing local YAML files |

Policy operations can append compact local audit events under `~/.agentflow/policy_events.jsonl`. Set `AGENTFLOW_POLICY_EVENTS=0` to disable that audit log.
Managed fetch/review is disabled unless a recommendation URL is supplied, and authenticated
managed servers should use `AGENTFLOW_MANAGED_API_KEY` or `--api-key-env` rather than putting
secrets in command history. The command validates and reviews the bundle only; use
`agentflow-policy-apply --dry-run` separately before writing local YAML files.
Managed rollout actions are also recommendation-only: review/apply commands reject unknown
local rules, raw prompt-like payloads, managed-enforced actions, and unsafe rule sources before
writing. Apply only updates matching local `pattern_rules` metadata in `crunch_rules.yaml` or
`cache_rules.yaml`, creates backups, and still requires explicit policy reload afterward.
If `AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET` or
`AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS` is configured, managed policy bundles must
include matching HMAC provenance before `agentflow-policy-apply` writes local YAML files.
Managed optimization rollout action review fails closed for missing provenance, expired
bundles, incompatible local executors, stale/missing local eval evidence, managed-enforced
actions, or raw prompt-like payloads. Managed rollout action bundles use the same
verification secrets when configured.
Unsigned local-default/local-manual bundles remain valid for offline use.
Managed outcome and rollout-action lifecycle feedback retries are disabled unless
`AGENTFLOW_RECOMMENDATION_ENABLED=1`. Status and flush output is metadata-only; queued
feedback payload JSON is not printed.
Pattern rollup output is aggregate-only and omits raw prompts, messages, responses,
transcripts, tool payloads, cache keys, file paths, request IDs, and local session IDs.

The runtime managed recommendation bridge is also opt-in. With
`AGENTFLOW_RECOMMENDATION_ENABLED=0`, AgentFlow runs in local-only mode and does not contact
the managed optimizer. To enable the bridge, set:

```bash
export AGENTFLOW_RECOMMENDATION_ENABLED=1
export AGENTFLOW_RECOMMENDATION_SERVER_URL=http://127.0.0.1:4100
export AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS=1.5
export AGENTFLOW_RECOMMENDATION_FAILURE_MODE=fallback-local
export AGENTFLOW_MANAGED_API_KEY="..."  # when the managed server requires auth
```

`fallback-local` is the only supported runtime failure mode today. If the server URL is unset,
unreachable, times out, returns non-2xx, returns invalid JSON/schema, recommends an unsupported
target model, or includes a replacement prompt, AgentFlow keeps the local routing/crunch/cache
decision and records the fallback reason in local managed recommendation metadata. Managed API
keys are sent as bearer tokens only when configured and are not included in dashboard or CLI
metadata.

Before eligible provider calls and Codex app turns, the bridge posts an
`agentflow.optimization_unit_features.v1` metadata-only feature unit to the managed
recommendation endpoint. The payload includes derived request size, category, local policy
decisions, candidate target model, privacy summary, and hashed grouping identifiers. It does
not include prompts, messages, params, provider request bodies, responses, transcripts, tool
payloads, API keys, or raw local session IDs in the default profile.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENTFLOW_DB` | `~/.agentflow/agentflow.sqlite3` | Local SQLite cache/log database |
| `AGENTFLOW_DATABASE_URL` | unset | Use Postgres instead of SQLite |
| `AGENTFLOW_CACHE` | `1` | Enable exact local cache where safe |
| `AGENTFLOW_CACHE_TOOL_CALLS` | `0` | Allow caching tool-call requests; keep off unless you understand the risk |
| `AGENTFLOW_ROUTING` | `1` | Enable local model routing |
| `AGENTFLOW_LOG_BODIES` | `0` | Store raw request/response bodies for local debugging |
| `AGENTFLOW_RECOMMENDATION_ENABLED` | `0` | Opt into runtime managed optimizer recommendations; disabled means local-only with no managed network call |
| `AGENTFLOW_RECOMMENDATION_SERVER_URL` | `http://127.0.0.1:4100` | Managed optimizer base URL; runtime recommendations post feature units to `/v1/recommendation` |
| `AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS` | `1.5` | Bounded timeout for managed recommendation and feedback calls |
| `AGENTFLOW_RECOMMENDATION_FAILURE_MODE` | `fallback-local` | Keep local policy authoritative on managed bridge failure |
| `AGENTFLOW_MANAGED_API_KEY` | unset | Optional bearer token for managed optimizer requests; value is not printed in status metadata |
| `AGENTFLOW_DASHBOARD_HOST` | `0.0.0.0` | Host for `agentflow-dashboard` |
| `AGENTFLOW_DASHBOARD_PORT` | `4002` | Port for `agentflow-dashboard` |
| `AGENTFLOW_CODEX_APP_MODEL` | current bundled default | Model used for Codex app-server token/cost estimates |
| `AGENTFLOW_CODEX_APP_RULES` | `~/.agentflow/codex_app_rules.yaml` | Optional local Codex app-server rule file for safe summary model hints and exact summary cache |
| `AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT` | `0` | Opt into local canary routing for safe Codex app summary turns |
| `AGENTFLOW_CODEX_APP_SUMMARY_MODEL_HINT_TARGET` | `gpt-5-codex` | Target model for the summary-turn model hint canary |
| `AGENTFLOW_CODEX_APP_CACHE` | `0` | Opt into exact Codex app summary-turn replay for safe, action-free JSON-RPC turns |
| `AGENTFLOW_CODEX_APP_SESSION_COST_ALERT_USD` | `AGENTFLOW_SESSION_COST_ALERT_USD` or `5.0` | Log a local warning when one Codex app thread/session crosses a daily estimated spend threshold |

When `AGENTFLOW_DATABASE_URL` is set, AgentFlow uses Postgres with a small connection pool. SQLite remains the default because the local proxy must work offline.

## Limitations

- Cost and token numbers are estimates unless the upstream provider returns exact usage.
- Streaming requests are forwarded and logged, but local exact-cache replay is intentionally conservative.
- Codex app-server support is experimental and currently more limited than direct provider proxying.
- Aggressive routing or caching can break agent behavior. Start conservative and inspect the dashboard.
- AgentFlow is not an authentication gateway. Keep provider proxy ports local.
