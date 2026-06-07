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
3. If the request is very large, shortens older non-tool text blocks by keeping the head and tail.
4. Never changes `tool_use` or `tool_result` blocks.

It does not yet use model-assisted summarization. That should be added later behind a careful fallback/eval path.

## What routing does

Set `AGENTFLOW_ROUTING=0` to disable all routing.

Default routing is conservative:

- non-tool Opus requests under threshold may route to Sonnet
- small non-tool Sonnet requests may route to Haiku
- tiny tool requests from Opus may route to Sonnet
- otherwise it keeps the requested model

You can override target aliases:

```bash
export AGENTFLOW_HAIKU_MODEL="claude-haiku-4.5"
export AGENTFLOW_SONNET_MODEL="claude-sonnet-4.5"
export AGENTFLOW_OPUS_MODEL="claude-opus-4.5"
```

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
5. Add model-assisted summarization behind opt-in fallback.
6. Add per-session savings estimates against a baseline model.
7. Add `agentflow doctor` to validate Claude Code config.
