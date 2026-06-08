# AgentFlow Proxy v0.1

A local provider-specific proxy for Claude Code / Claude CLI and Codex experiments.

It runs on `127.0.0.1`, accepts whatever auth header the client sends, forwards that auth upstream to the selected provider, and adds:

- `/v1/messages` Anthropic-compatible proxy
- `/v1/responses` and `/v1/chat/completions` OpenAI-compatible proxy
- streaming pass-through
- conservative prompt crunching
- provider-local model routing
- exact SQLite caching for non-stream, non-tool requests
- SQLite call logging
- `/agentflow/stats` endpoint

This is a first prototype, not production software.

## Target architecture

The current product is the **local AgentFlow module**: a Python package that runs on the
user's machine as provider-specific middleware, stores logs/cache locally, exposes a read-only dashboard,
and will provide local manual rules for model selection, crunching, and exact-match hash cache
matching.

The future product is a separate **managed optimizer server** for paying users. It should be
opt-in and tenant-aware, and should improve routing, crunching, and cache policy from broader
aggregate learning. It is not a dependency for the local middleware.

The intended product split is:

- free local package: low-level middleware, dashboard, local logs/cache, manual YAML rules,
  and conservative deterministic savings;
- premium managed optimizer: learned policy bundles, broader cache/policy intelligence,
  quality evaluation, and the strongest cost-saving recommendations.

See `ARCHITECTURE.md` before making architectural changes.

## Install

```bash
cd agentflow
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or without editable install:

```bash
pip install -r requirements.txt
```

## Test

```bash
"${AGENTFLOW_TARGET_PYTHON:-python3}" -m unittest discover -s tests
```

## Run

Claude / Anthropic mode:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
agentflow-proxy --provider anthropic --host 127.0.0.1 --port 4000
```

Codex / OpenAI mode:

```bash
export OPENAI_API_KEY="sk-..."
agentflow-proxy --provider openai --host 127.0.0.1 --port 4003
```

Health check:

```bash
curl http://127.0.0.1:4000/health
```

Stats:

```bash
curl http://127.0.0.1:4000/agentflow/stats | python -m json.tool
```

Reload local policy files after editing YAML rules:

```bash
agentflow-policy-reload | python -m json.tool
```

The command posts to the proxy's loopback-only admin endpoint at
`http://127.0.0.1:${AGENTFLOW_PORT:-4000}/agentflow/admin/reload-policies` and prints the
refreshed `agentflow.policy_reload.v1` JSON. It refuses non-loopback URLs by default.

Export the effective local policy bundle without contacting the proxy or any managed service:

```bash
agentflow-policy-export --pretty
```

The command prints `agentflow.policy_bundle.v1` JSON containing routing, crunch, cache, and
routing-experiment policy state, including policy sources, rule paths, and reload status. This
is an offline local export shape for auditability and future optimizer interfaces; it does not
upload data or enable managed-server behavior.

Validate an exported policy bundle offline:

```bash
agentflow-policy-export | agentflow-policy-validate -
```

The validator prints `agentflow.policy_bundle_validation.v1` JSON and exits non-zero for
malformed JSON or bundles that do not match the local offline export contract.

Compare two exported policy bundles offline before reloading or importing policy changes:

```bash
agentflow-policy-diff before.json after.json --pretty
```

The diff command prints `agentflow.policy_bundle_diff.v1` JSON with changed policy sections
and JSON paths under routing, crunch, cache, and routing experiments. It validates both
inputs first and exits non-zero for malformed bundles.

Review and apply a proposed policy bundle offline:

```bash
agentflow-policy-review proposed.json --pretty
agentflow-policy-apply proposed.json --dry-run --pretty
agentflow-policy-apply proposed.json --config-dir ~/.agentflow
```

The apply command writes loader-compatible YAML rule files for routing, crunch, cache, and
routing experiments. It validates the bundle first, refuses risky policy warnings unless
`--allow-risky` is explicit, creates timestamped backups before changed files are overwritten,
and does not contact the proxy or any managed service.

Policy reload/export/validate/diff/review/apply operations append compact local audit events to
`~/.agentflow/policy_events.jsonl` by default. The read-only dashboard exposes recent entries
at `/agentflow/stats/policy-events` and in the Policies tab. Set `AGENTFLOW_POLICY_EVENTS=0`
to disable the audit log, or `AGENTFLOW_POLICY_EVENTS_LOG=/path/to/events.jsonl` to choose a
different local file.

## Point Claude Code / Claude CLI at it

The exact environment variable names can differ by Claude Code version, but the intended setup is:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_API_KEY"
# or, for clients that use x-api-key directly:
export ANTHROPIC_API_KEY="sk-ant-..."
claude
```

The proxy does not require local auth. It simply forwards incoming `authorization`, `x-api-key`, `anthropic-version`, and `anthropic-beta` headers to Anthropic.

## Point Codex at it

Run the OpenAI proxy on a separate port when you want to test OpenAI API-compatible clients
or API-key billing:

```bash
export OPENAI_API_KEY="sk-..."
agentflow-proxy --provider openai --openai-auth-mode proxy --host 127.0.0.1 --port 4003
codex exec --config 'openai_base_url="http://127.0.0.1:4003/v1"' "Reply with ok"
```

Provider modes are intentionally separate. An Anthropic-mode process serves `/v1/messages`;
an OpenAI-mode process serves `/v1/responses` and `/v1/chat/completions`. Cross-provider
routing is not supported.

Use `--openai-auth-mode client` to preserve whatever auth an API-compatible client sends.
Use `--openai-auth-mode proxy` when you intentionally want the proxy to replace client auth
with `OPENAI_API_KEY` or `AGENTFLOW_OPENAI_API_KEY`.

For Codex OAuth/subscription quota, do not set `openai_base_url`: that forces the public
OpenAI API `/v1/responses` path and requires API scopes. Use Codex's default profile/auth,
or the safer experimental Codex app-server protocol:

```bash
codex app-server --listen ws://127.0.0.1:4014
agentflow-codex-app-proxy --host 127.0.0.1 --port 4013 --upstream ws://127.0.0.1:4014
printf 'Reply with exactly: ok\n' | agentflow-codex-app-client --url ws://127.0.0.1:4013 --cd "$PWD"
```

This relay is pass-through telemetry first. It records redacted JSON-RPC method names and sizes
to SQLite without storing raw prompts.

If your client refuses `ANTHROPIC_AUTH_TOKEN`, try keeping `ANTHROPIC_API_KEY` set. The proxy accepts both header styles.

## Direct test

```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4.5",
    "max_tokens": 200,
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

## What crunching does

This prototype is deliberately conservative for agent usage:

1. Normalizes whitespace in large text blocks.
2. Removes exact duplicate large text blocks within the same request.
3. Removes older near-duplicate assistant `thinking` blocks while preserving the latest assistant thinking block.
4. If the request is very large, shortens older non-tool text blocks by keeping the head and tail.
5. Never changes `tool_use` or `tool_result` blocks.

Model-assisted old-context summarization exists but is disabled by default. To enable it,
set `old_context_summarization.enabled: true` in `config/crunch_rules.yaml` or set
`AGENTFLOW_HAIKU_SUMMARIZE_OLD_CONTEXT=1`. When enabled, AgentFlow may make one additional
Anthropic request to summarize old non-tool turns. The resulting summary is inserted as a
tagged top-level `system` context block rather than an ordinary user message; recent turns and
all `tool_use` / `tool_result` protocol messages stay unchanged. Summary creation cost,
summary cache reuse, estimated saved tokens, and net savings are exposed under
`/agentflow/stats/full` and on the dashboard. The summary cache is used when available, but
exact request caching is not required for summarization eligibility.

## What routing does

Set `AGENTFLOW_ROUTING=0` to disable all routing.

Default routing is conservative:

- non-tool Opus requests under threshold may route to Sonnet
- small non-tool Sonnet requests may route to Haiku
- midsize non-tool, non-code Sonnet requests from 8k to 30k text chars may route to Haiku
  when `AGENTFLOW_ROUTE_MIDSIZE=1`
- tiny tool requests from Opus may route to Sonnet
- otherwise it keeps the requested model

You can override target aliases:

```bash
export AGENTFLOW_HAIKU_MODEL="claude-haiku-4.5"
export AGENTFLOW_SONNET_MODEL="claude-sonnet-4.5"
export AGENTFLOW_OPUS_MODEL="claude-opus-4.5"
```

## Routing experiments

A/B routing experiments are disabled by default because they send an additional shadow
request to the originally requested model.

To opt in, copy `agentflow_proxy/routing_experiments.yaml` to
`~/.agentflow/routing_experiments.yaml` or set `AGENTFLOW_ROUTING_EXPERIMENTS` to a local
YAML file. A minimal policy is:

```yaml
enabled: true
sample_rate: 0.05
categories:
  - tool-result
similarity_threshold: 0.86
store_response_bodies: false
```

Experiments only sample non-streaming calls that AgentFlow routed down. The normal client
receives the routed response; AgentFlow sends a shadow request to the original model and
stores status, latency, output hashes, output similarity, and estimated costs in SQLite.
`/agentflow/stats/full` exposes `routing_experiment_summary` with average similarity,
pass rate, and a conservative confidence score by route/category. Response bodies are only
stored when `store_response_bodies: true` or `AGENTFLOW_LOG_BODIES=1` is enabled.

## Managed recommendations

The local proxy can optionally ask a separate `agentflow_server` instance for a feature-only
recommendation before it forwards a provider request. This is disabled by default, provider
calls still happen locally with the user's credentials, and server failures fall back to the
local routing/crunch/cache policy.

```bash
export AGENTFLOW_RECOMMENDATION_ENABLED=1
export AGENTFLOW_RECOMMENDATION_SERVER_URL="http://127.0.0.1:4100"
export AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS=1.5
```

The proxy posts normalized optimization-unit metadata to `/v1/recommendation`; it does not
send raw prompt bodies by default. Returned `target_model` values may update the final local
model when they stay inside the active provider family. Returned `replacement_prompt` values
are recorded only as presence/hash metadata and are not applied by this local bridge yet.
Recommendation status, policy ID, confidence, failure reason, and fallback metadata are stored
inside `routing_json.managed_recommendation`.

## Caching behavior

By default, exact cache is enabled only for non-streaming requests without tool blocks.

Reason: Claude Code tool calls often depend on live filesystem state. Caching tool turns can be unsafe.

Controls:

```bash
export AGENTFLOW_CACHE=1
export AGENTFLOW_CACHE_TOOL_CALLS=0
```

Cache/log DB defaults to:

```text
~/.agentflow/agentflow.sqlite3
```

Override:

```bash
export AGENTFLOW_DB="/path/to/agentflow.sqlite3"
```

SQLite remains the default because the local proxy must work offline. To use Postgres as the
durable backend, install the package dependencies and set:

```bash
export AGENTFLOW_DATABASE_URL="postgresql://agentflow:agentflow@127.0.0.1:5432/agentflow"
```

When `AGENTFLOW_DATABASE_URL` is set, AgentFlow creates the `cache`, `semantic_cache`,
`cache_file_deps`, `calls`, `routing_experiments`, and `codex_app_events` tables in Postgres
and uses a small connection pool. `AGENTFLOW_POSTGRES_POOL_MIN` and
`AGENTFLOW_POSTGRES_POOL_MAX` control the pool size.

Migration from an existing SQLite DB is deferred: start Postgres with an empty database, or
export/import rows manually for local experiments. The proxy interface is backend-neutral, so
adding a first-class migration command can be done without changing request handling.

## Logging

The proxy logs metadata for every call:

- requested model
- routed model
- stream yes/no
- cache hit yes/no
- status code
- latency
- rough input/output token estimates
- rough cost estimate
- crunch/routing metadata

By default it does **not** store prompt/response bodies.

Enable body logging only for local debugging:

```bash
export AGENTFLOW_LOG_BODIES=1
```

## Important limitations

- This is a prototype.
- Streaming is pass-through and not cached.
- Cost estimates are rough and based on approximate chars/token.
- Model prices are embedded and may be stale. Update `MODEL_PRICES` in `server.py`.
- Claude Code compatibility depends on the exact version and headers it sends.
- Aggressive routing can break agent behavior. Start conservative.
- This server should bind to `127.0.0.1` unless you add real auth.

## Roadmap

Good next steps:

1. Add a small terminal dashboard.
2. Add configurable YAML routing rules.
3. Add exact cache for selected tool-safe endpoints.
4. Add Anthropic-style streaming event parsing for better logging.
5. Add evaluation coverage for opt-in model-assisted summarization.
6. Add per-session savings estimates against a baseline model.
7. Add `agentflow doctor` to validate Claude Code config.
