# AgentFlow Backlog

Items are ordered by priority within each section. The orchestrator reads this file every run
to decide what to work on. It also writes to this file: adding new findings, updating status,
and moving items between sections.

Format per item:
```
- [STATUS] Title — brief rationale
  Details: any implementation notes
  Metric: how we'll know it worked
```

Statuses: READY | IN-PROGRESS | DONE | BLOCKED | IDEA

---

## P0 — Foundation (do these first, everything else builds on them)

- [DONE] Target architecture documentation (2026-06-02)
  Details: Added `ARCHITECTURE.md` to define the intended split between the local Python
  middleware module and the future managed optimizer server for paying users. The local module
  owns Claude middleware, the read-only dashboard, SQLite logs/cache, and local manual rules
  for model selection, crunching, and exact-match hash matching. The future server is separate,
  opt-in, tenant-aware, and focused on better routing/crunching policies plus broader cache
  and telemetry learning.
  Metric: unattended agents read a clear architecture contract before choosing or implementing
  backlog work.

- [DONE] Dev/prod instance split
  Details: Prod is the proxy on port 4000 serving real Claude Code traffic — never restart
  it mid-development. Dev is a second instance on port 4001 pointing at a separate DB
  (~/.agentflow/dev.sqlite3). The developer agent edits code and tests exclusively against
  the dev instance. Only after run_tester passes against dev does the orchestrator do a
  rolling restart of prod (SIGTERM + start, not kill -9) and re-run a quick smoke test.
  Implement as: AGENTFLOW_PORT env var already exists; add scripts/start_dev.sh that
  launches port 4001 with AGENTFLOW_DB=~/.agentflow/dev.sqlite3. Update run_orchestrator.py
  to target port 4001 for developer/tester agents and port 4000 only for the final prod
  promotion step.
  Metric: real traffic on port 4000 is uninterrupted during an orchestrator run; dev DB
  is separate so test calls don't pollute prod stats.

- [DONE] Accurate token counting from API response headers (2026-06-02)
  Details: Parse `x-request-id`, `input-tokens`, `output-tokens` from Anthropic response.
  Currently using rough chars/4 estimate. Real counts are needed for accurate cost tracking
  and for measuring whether routing/crunching actually saves tokens.
  Metric: calls table has non-null actual_input_tokens, actual_output_tokens columns;
  compare estimates vs. actuals to calibrate.

- [DONE] Session ID tracking (2026-06-02)
  Details: Claude Code sends a stable session identifier in headers (check x-session-id,
  x-request-id patterns, or generate one per connected client IP+port). Group calls into
  sessions so we can track per-session cost, measure agentic workflow phases, and scope
  the cache safely.
  Metric: calls table has session_id; dashboard can show per-session cost.

- [DONE] Dashboard v1 — basic HTML page at /agentflow/dashboard (2026-06-02)
  Details: Serve a simple HTML page (no JS framework, vanilla JS + CSS) that shows:
  - Last 50 calls table (timestamp, model requested, model used, cache hit, latency, cost_est)
  - Summary stats: total calls, total cost, cache hit rate, total saved by routing
  - Auto-refresh every 5 seconds via meta refresh or fetch
  Metric: page loads, data is live, no external dependencies.

- [DONE] Streaming token tracking (2026-06-02)
  Details: Parse SSE stream events to extract usage data from the final `message_delta` event
  which contains `usage.output_tokens`. Store it in the DB. Currently streaming calls log
  no output tokens.
  Metric: streaming calls have non-null output_tokens_est in DB.

---

## P1 — Better Measurement (know what's actually happening)

- [DONE] Fix streaming cost_est_usd (always null) (2026-06-02)
  Details: In server.py streaming `finally` block, `cost_est_usd` is hardcoded to None even
  though `actual_in`/`actual_out` are now populated from SSE events. Fix: call `estimate_cost()`
  with actual tokens the same way the non-streaming path does (server.py:450-452).
  Metric: streaming calls have non-null cost_est_usd; dashboard cost totals include streaming.

- [DONE] Fix dashboard time rendering as NaNd (2026-06-02)
  Details: The dashboard currently shows time values as `NaNd`, which makes recent calls and
  unattended cron/operator history hard to interpret. Inspect the timestamp format returned by
  the stats endpoints and the dashboard JavaScript date formatting path; handle null/invalid
  timestamps gracefully instead of rendering broken text.
  Metric: dashboard time columns render valid local times or a clear placeholder, with no `NaN`
  or `NaNd` values visible.

- [DONE] Dashboard 7-day statistics tab (2026-06-02)
  Details: Add a tab or separate page on the read-only dashboard showing the last 7 days of
  operational statistics: calls received, successful calls, errors, cache hits, average latency,
  estimated cost, baseline cost, savings from routing/crunching/cache, and trends by day.
  Include enough detail to understand what unattended cron/operator sessions did over time.
  Metric: dashboard exposes a 7-day view with daily totals and aggregate success/error/savings
  figures based on stored call data.

- [DONE] Cost comparison baseline (2026-06-02)
  Details: For every call, log what it would have cost at the requested model (before routing).
  This gives us `cost_without_agentflow` vs `cost_with_agentflow` so the dashboard can show
  real savings.
  Metric: dashboard shows "You saved $X today / $Y this week".

- [DONE] Crunch effectiveness metric (2026-06-03)
  Details: Currently we log saved_chars but not saved_tokens. Add actual_tokens_before_crunch
  estimate using real token counts from similar uncrunched calls. Track crunch ratio over time.
  Metric: dashboard shows avg crunch ratio and tokens saved by crunching.

- [DONE] Request categorization (2026-06-02)
  Details: Tag each call with a category: tool-heavy, code-gen, chat, short-completion, etc.
  Use simple heuristics: presence of tool blocks, length, system prompt patterns. This lets
  us see which categories benefit most from which optimizations.
  Metric: calls table has `category` column; routing stats broken down by category.

---

## P2 — Better Crunching

- [DONE] Near-duplicate detection for text blocks (2026-06-03)
  Details: Instead of exact hash comparison, compute a 4-gram shingle similarity. If two
  text blocks share >85% 4-grams, treat them as duplicates and omit the older one.
  Important: be conservative — only apply to large blocks (>2000 chars) in older messages.
  Metric: crunch ratio improvement on real traffic; zero tool call failures.

- [IDEA] Haiku-assisted summarization of old context
  Details: When a message thread is very long (>32k chars), use Haiku to summarize the
  oldest non-tool turns into a compact summary. Replace original turns with summary + notice.
  This is the most powerful crunch technique but also the riskiest — needs careful eval.
  Gating: only enable if exact cache is enabled (for rollback) and behind explicit env flag.
  Metric: token reduction on long sessions; no task completion regressions (measure via
  comparing tool call success rates with and without).

- [IDEA] System prompt deduplication across calls
  Details: Many Claude Code calls repeat the same long system prompt. Cache the system prompt
  separately, assign it a hash, and use Anthropic's prompt caching beta header to mark it as
  cacheable at the API level. This doesn't reduce what we send but reduces what Anthropic charges.
  Metric: Anthropic cache_creation_input_tokens in response headers; cost reduction on repeated
  system prompts.

---

## P3 — Better Routing

- [DONE] YAML routing rules (2026-06-03)
  Details: Replace hardcoded thresholds in route_model() with a routing_rules.yaml file.
  Schema: list of rules with conditions (model_pattern, text_chars_lt, has_tools, category)
  and action (route_to, reason). Rules evaluated top-to-bottom, first match wins.
  Metric: routing config externalizable, no behavior change on existing defaults.

- [IDEA] Phase-aware routing for agentic workflows
  Details: Within a session, classify each turn as: planning, tool-execution, verification,
  or summary. Use Haiku for tool-execution and summary turns, Sonnet for verification,
  keep requested model for planning.
  Phase signals: presence of tool_result blocks = execution; short final turn = summary;
  long system prompt + no prior context = planning.
  Metric: per-session cost reduction without increase in error/retry rate.

- [IDEA] A/B routing experiment framework
  Details: Shadow a fraction of routed-down calls to both models, compare output similarity.
  This measures actual quality impact of routing decisions rather than relying on heuristics.
  Requires storing both responses and a similarity metric (could be embedding cosine sim).
  Metric: can produce routing confidence scores based on empirical output similarity.

---

## P4 — Better Caching

- [DONE] Semantic cache for non-tool requests (2026-06-03)
  Details: For non-tool, non-streaming requests, compute an embedding of the request text
  and check against cached embeddings. If cosine similarity > 0.95, return the cached response
  with a note. Use a lightweight local embedding model (e.g., all-MiniLM-L6-v2 via sentence-
  transformers) to avoid API calls for embeddings.
  Metric: cache hit rate increases; measure false positive rate (wrong cached responses returned).

- [IDEA] Streaming response cache
  Details: Buffer complete streamed responses, store them, and replay as a stream for cache hits.
  The challenge is that streaming clients expect a live stream — we need to replay at a plausible
  rate or just replay instantly with a flag.
  Metric: streaming cache hit rate; zero broken stream clients.

- [IDEA] File-watch cache invalidation
  Details: Monitor the working directory for file changes. When a file changes, invalidate
  cache entries that mention that file path in the request. Makes tool-call caching safer.
  Metric: can enable AGENTFLOW_CACHE_TOOL_CALLS=1 without stale cache issues.

---

## Completed

(Orchestrator moves DONE items here with date)

---

## Agent Findings

(Orchestrator appends new opportunities discovered during analysis runs here)

- [DONE] Normalize dot-notation model aliases before forwarding (2026-06-02)
  Details: Dot-notation aliases (claude-haiku-4.5, claude-sonnet-4.5, claude-opus-4.5) reach
  Anthropic unchanged and return HTTP 404. Normalize them in the handler before forwarding:
  claude-haiku-4.5 → claude-haiku-4-5-20251001, claude-sonnet-4.5 → claude-sonnet-4-5-20240620,
  claude-opus-4.5 → claude-opus-4-5. Tester confirmed 16 prod calls failing with 404.
  Metric: zero 404s from dot-notation aliases; router still routes correctly after normalization.
