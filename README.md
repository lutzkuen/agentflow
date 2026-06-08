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

AgentFlow can use local YAML-backed policy files for routing, crunching, cache, and routing experiments. The dashboard shows which policies are loaded and whether any file needs reload.

Common policy commands:

| Command | Purpose |
| --- | --- |
| `agentflow-policy-export --pretty` | Export the current effective local policy bundle |
| `agentflow-policy-validate bundle.json` | Validate a policy bundle before use |
| `agentflow-policy-diff before.json after.json --pretty` | Compare two policy bundles |
| `agentflow-policy-review proposed.json --pretty` | Review changes and warnings before apply |
| `agentflow-policy-apply proposed.json --dry-run --pretty` | Preview local file writes |
| `agentflow-policy-apply proposed.json --config-dir ~/.agentflow` | Apply reviewed local policy files |
| `agentflow-policy-rollback --config-dir ~/.agentflow --dry-run --pretty` | Preview rollback to the newest backup |
| `agentflow-policy-reload` | Reload changed policies in a running local proxy |

Policy operations can append compact local audit events under `~/.agentflow/policy_events.jsonl`. Set `AGENTFLOW_POLICY_EVENTS=0` to disable that audit log.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENTFLOW_DB` | `~/.agentflow/agentflow.sqlite3` | Local SQLite cache/log database |
| `AGENTFLOW_DATABASE_URL` | unset | Use Postgres instead of SQLite |
| `AGENTFLOW_CACHE` | `1` | Enable exact local cache where safe |
| `AGENTFLOW_CACHE_TOOL_CALLS` | `0` | Allow caching tool-call requests; keep off unless you understand the risk |
| `AGENTFLOW_ROUTING` | `1` | Enable local model routing |
| `AGENTFLOW_LOG_BODIES` | `0` | Store raw request/response bodies for local debugging |
| `AGENTFLOW_DASHBOARD_HOST` | `0.0.0.0` | Host for `agentflow-dashboard` |
| `AGENTFLOW_DASHBOARD_PORT` | `4002` | Port for `agentflow-dashboard` |
| `AGENTFLOW_CODEX_APP_MODEL` | current bundled default | Model used for Codex app-server token/cost estimates |

When `AGENTFLOW_DATABASE_URL` is set, AgentFlow uses Postgres with a small connection pool. SQLite remains the default because the local proxy must work offline.

## Limitations

- Cost and token numbers are estimates unless the upstream provider returns exact usage.
- Streaming requests are forwarded and logged, but local exact-cache replay is intentionally conservative.
- Codex app-server support is experimental and currently more limited than direct provider proxying.
- Aggressive routing or caching can break agent behavior. Start conservative and inspect the dashboard.
- AgentFlow is not an authentication gateway. Keep provider proxy ports local.
