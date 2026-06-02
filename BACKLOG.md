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

- [READY] Accurate token counting from API response headers
  Details: Parse `x-request-id`, `input-tokens`, `output-tokens` from Anthropic response.
  Currently using rough chars/4 estimate. Real counts are needed for accurate cost tracking
  and for measuring whether routing/crunching actually saves tokens.
  Metric: calls table has non-null actual_input_tokens, actual_output_tokens columns;
  compare estimates vs. actuals to calibrate.

- [READY] Session ID tracking
  Details: Claude Code sends a stable session identifier in headers (check x-session-id,
  x-request-id patterns, or generate one per connected client IP+port). Group calls into
  sessions so we can track per-session cost, measure agentic workflow phases, and scope
  the cache safely.
  Metric: calls table has session_id; dashboard can show per-session cost.

- [READY] Dashboard v1 — basic HTML page at /agentflow/dashboard
  Details: Serve a simple HTML page (no JS framework, vanilla JS + CSS) that shows:
  - Last 50 calls table (timestamp, model requested, model used, cache hit, latency, cost_est)
  - Summary stats: total calls, total cost, cache hit rate, total saved by routing
  - Auto-refresh every 5 seconds via meta refresh or fetch
  Metric: page loads, data is live, no external dependencies.

- [READY] Streaming token tracking
  Details: Parse SSE stream events to extract usage data from the final `message_delta` event
  which contains `usage.output_tokens`. Store it in the DB. Currently streaming calls log
  no output tokens.
  Metric: streaming calls have non-null output_tokens_est in DB.

---

## P1 — Better Measurement (know what's actually happening)

- [READY] Cost comparison baseline
  Details: For every call, log what it would have cost at the requested model (before routing).
  This gives us `cost_without_agentflow` vs `cost_with_agentflow` so the dashboard can show
  real savings.
  Metric: dashboard shows "You saved $X today / $Y this week".

- [READY] Crunch effectiveness metric
  Details: Currently we log saved_chars but not saved_tokens. Add actual_tokens_before_crunch
  estimate using real token counts from similar uncrunched calls. Track crunch ratio over time.
  Metric: dashboard shows avg crunch ratio and tokens saved by crunching.

- [READY] Request categorization
  Details: Tag each call with a category: tool-heavy, code-gen, chat, short-completion, etc.
  Use simple heuristics: presence of tool blocks, length, system prompt patterns. This lets
  us see which categories benefit most from which optimizations.
  Metric: calls table has `category` column; routing stats broken down by category.

---

## P2 — Better Crunching

- [READY] Near-duplicate detection for text blocks
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

- [READY] YAML routing rules
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

- [READY] Semantic cache for non-tool requests
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
