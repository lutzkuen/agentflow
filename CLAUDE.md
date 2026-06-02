# AgentFlow — Codebase Guide

Read this before making any change. It applies to human developers and agent runs equally.

## What this repo is

A local Anthropic-compatible proxy that reduces API cost via crunching, routing, and caching.
See `NORTH_STAR.md` for goals. See `BACKLOG.md` for what to work on.

## Ports

| Instance | Port | DB | Purpose |
|----------|------|----|---------|
| prod | 4000 | `~/.agentflow/agentflow.sqlite3` | Live traffic from Claude Code / Claude CLI |
| dev  | 4001 | `~/.agentflow/dev.sqlite3` | Agent development and testing |

**Never restart prod mid-development.** The developer agent works against port 4001.
Only the orchestrator promotes to prod after tests pass.

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
