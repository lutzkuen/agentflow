# AgentFlow — Codebase Guide

Read this before making any change. It applies to human developers and agent runs equally.

## What this repo is

A local provider-specific proxy that reduces API cost via crunching, routing, and caching.
See `ARCHITECTURE.md` for the target product shape, `NORTH_STAR.md` for goals, and
`BACKLOG.md` for what to work on.

## Architecture contract

AgentFlow has two planned layers:

- the local Python module: localhost provider middleware, read-only LAN dashboard, local SQLite
  logs/cache, and local manual rules for model routing, crunching, and exact-match cache policy;
- the future managed optimizer: a separate opt-in server for paying users that can provide
  better routing/crunching policies and a wider policy/cache knowledge base.

Do not build SaaS concerns into the local proxy. The local module may define clean interfaces
for later policy import/export, but it must remain useful without a managed server. Keep the
free/local package focused on low-level manual controls and conservative deterministic savings;
reserve learned, cross-install, aggressive, or continuously optimized policies for the future
premium managed service.

## Ports

| Instance | Port | DB | Purpose |
|----------|------|----|---------|
| prod | 4000 | `~/.agentflow/agentflow.sqlite3` | Live traffic from Claude Code / Claude CLI |
| dev  | 4001 | `~/.agentflow/dev.sqlite3` | Agent development and testing |
| dashboard | 4002 | `~/.agentflow/agentflow.sqlite3` | Read-only LAN dashboard |
| OpenAI proxy | 4003 | `~/.agentflow/agentflow.sqlite3` | Localhost-only Codex/Codex extension proxy |
| Codex app proxy | 4013 | `~/.agentflow/agentflow.sqlite3` | Experimental Codex OAuth app-server telemetry relay |
| Codex app upstream | 4014 | `~/.agentflow/agentflow.sqlite3` | Local Codex app-server behind the relay |

**Never restart prod mid-development.** The developer agent works against port 4001.
Only the orchestrator promotes to prod after tests pass.
Never expose provider proxy endpoints on the LAN. Only the read-only dashboard may bind
outside localhost.

## Module structure

The codebase will grow. Split into modules when a logical unit exceeds ~200 lines
or needs to be tested in isolation. The target structure:

```
agentflow_proxy/
  server.py        — FastAPI app, route handlers only; imports from other modules
  store.py         — Store class, all SQLite logic
  crunch.py        — crunch_body() and all text-reduction logic
  router.py        — route_model() and all routing logic
  cache.py         — cache key, get/set, TTL, semantic cache when added
  pricing.py       — MODEL_PRICES, estimate_cost()
  config.py        — load file-backed local rules and defaults
  dashboard.py     — /agentflow/dashboard HTML endpoint
```

`server.py` should only contain: app init, middleware, and the `/v1/messages` handler that
calls into the other modules. If you find yourself adding a second function to `server.py`
that isn't a route handler, it belongs in a module.

## Rules for agents making code changes

1. **One item per run.** Implement exactly what the backlog item says. No bonus refactoring.

2. **Test against dev (port 4001), not prod.** Start dev with:
   ```bash
   AGENTFLOW_PORT=4001 AGENTFLOW_DB=~/.agentflow/dev.sqlite3 \
     python -m uvicorn agentflow_proxy.server:app --host 127.0.0.1 --port 4001 &
   ```

3. **DB schema changes need migrations.** Check existing columns with `PRAGMA table_info`
   before `ALTER TABLE ... ADD COLUMN`; this repo's SQLite rejects `ADD COLUMN IF NOT EXISTS`.
   Never drop a column. Never rename a column. Add a new one.

4. **No new dependencies without justification.** If adding a dep, explain in the commit
   why it's necessary and that no stdlib/already-present alternative exists.

5. **No comments that describe what the code does.** Only add a comment when the *why*
   is non-obvious: a hidden constraint, a workaround, a subtle invariant.

6. **Routing and crunching changes need a before/after token count logged.**
   The metric is tokens, not chars. If we don't measure it, we don't know if it helps.

7. **When splitting a module:** copy the code, update the import in server.py, delete the
   original function, restart dev, run tests. Commit only if tests pass.

## Running tests

```bash
# Quick smoke test against dev
curl -s -X POST http://localhost:4001/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-haiku-4.5","max_tokens":20,"messages":[{"role":"user","content":"Reply: ok"}]}' \
  | python3 -m json.tool

# Full test suite (agents/test.md describes the cases)
claude --print --allowedTools "Bash,Read" < agents/test.md
```

## Unattended recovery

Hourly cron runs call `scripts/run_orchestrator_cron.sh`. The shell guard uses Codex as the
default unattended workhorse; set `AGENTFLOW_WORKER=claude` to switch back to Claude. The
guard records a per-worker cooldown after upstream rate limits, so hourly runs keep checking
local proxy health but skip worker calls until the cooldown expires. If a run exits with
`CODEX_REQUIRED`, the wrapper invokes `scripts/codex_recover.sh` once, subject to a cooldown,
so preserved run worktrees are salvaged promptly instead of blocking later hours indefinitely.
Codex recovery defaults to the original OpenAI base URL even if normal Codex work is routed
through a proxy. Cron and the systemd timer both load `$REPO/.env` before each run when that
file exists, so local worker settings can live there without editing scheduler definitions.

The same cron installer also adds a nightly log archive job. `scripts/archive_orchestrator_logs.py`
deletes quota-only orchestrator logs older than 24 hours and moves work/action logs into
`logs/orchestrator/YYYY/MM/DD/` using only deterministic marker checks. It only archives
completed work logs after a short age buffer so active cron logs stay at their original path
until the wrapper has finished recovery checks.

Controls:

```bash
export AGENTFLOW_WORKER=codex                         # default; set to claude to switch back
export AGENTFLOW_CODEX_MODEL="gpt-5-codex"            # optional explicit Codex worker model
export AGENTFLOW_CODEX_OPENAI_BASE_URL=              # leave empty for Codex OAuth/subscription quota
export AGENTFLOW_OPENAI_AUTH_MODE=client              # preserve client auth when testing API-compatible OpenAI proxying
export AGENTFLOW_OPENAI_API_KEY=sk-...                # optional, only used when AGENTFLOW_OPENAI_AUTH_MODE=proxy
export AGENTFLOW_CODEX_SANDBOX=danger-full-access     # default; avoids bwrap user-namespace failures in unattended Codex
export AGENTFLOW_CODEX_RECOVERY_USE_ORIGINAL_OPENAI=0 # default recovery uses Codex's own profile/auth
export AGENTFLOW_CODEX_ORIGINAL_OPENAI_BASE_URL=https://api.openai.com/v1
export AGENTFLOW_CODEX_AUTO=0                         # disable automatic Codex recovery
export AGENTFLOW_CODEX_RECOVERY_COOLDOWN_MINUTES=180  # default retry cooldown
export AGENTFLOW_CLAUDE_RATE_LIMIT_COOLDOWN_MINUTES=90 # default Claude retry cooldown
export AGENTFLOW_CODEX_RATE_LIMIT_COOLDOWN_MINUTES=90 # default Codex retry cooldown
export AGENTFLOW_ARCHIVE_CRON_HOUR=2                  # default nightly archive hour
export AGENTFLOW_ARCHIVE_CRON_MINUTE=7                # default nightly archive minute
export AGENTFLOW_WORK_LOG_ARCHIVE_MIN_AGE_HOURS=2     # default active-log safety buffer
```

Run provider proxies as separate processes so the served API surface is explicit:

```bash
agentflow-proxy --provider anthropic --host 127.0.0.1 --port 4000
agentflow-proxy --provider openai --host 127.0.0.1 --port 4003
```

OpenAI-mode proxying supports the OpenAI-compatible `/v1/responses` and
`/v1/chat/completions` endpoints plus Responses/file/upload passthrough routes. Routing stays
within OpenAI models only; cross-provider routing is deliberately out of scope.

For Codex OAuth/subscription quota, do not use `openai_base_url`: that forces the public
OpenAI API `/v1/responses` path. Use the experimental app-server relay instead:

```bash
codex app-server --listen ws://127.0.0.1:4014
agentflow-codex-app-proxy --host 127.0.0.1 --port 4013 --upstream ws://127.0.0.1:4014
```

The app-server relay is currently pass-through telemetry only. It records JSON-RPC method names,
message sizes, input item counts, thread IDs, and latency in `codex_app_events`, but it does not
store raw prompts or auth-bearing payloads. Prompt crunching should only be enabled later for a
small allowlist such as `turn/start` after we validate the observed protocol shape.

## Commit message format

```
<scope>: <what changed>

<why it was needed, if not obvious>
```

Examples:
- `store: add actual_input_tokens and actual_output_tokens columns`
- `router: extract route_model into router.py module`
- `agent: implement accurate token counting from response headers`

Orchestrator commits use prefix `agent:`. Human commits use the relevant module name.
