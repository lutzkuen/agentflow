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

- [DONE] Define free-vs-premium policy bundle boundary (2026-06-05)
  Details: Turn the architecture's product tier boundary into concrete policy metadata and
  file/interface shapes. The free local package should expose low-level manual controls and
  conservative deterministic savings. The premium managed service should own learned policy
  bundles, quality/risk scoring, broader cache/policy intelligence, and continuously updated
  recommendations. Add fields such as policy_source (`local-default`, `local-manual`,
  `managed-recommended`, `managed-enforced`) to decision metadata where appropriate, without
  adding billing, tenant, account, or hosted-server logic to the local proxy.
  Metric: routing/crunch/cache decisions can report policy source; architecture docs and
  code interfaces make clear which savings levers belong to free local vs premium managed.

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

- [DONE] Extract Store class into store.py — architecture priority #1 (2026-06-04)
  Details: Moved utc_now(), stable_json(), cosine_similarity(), and Store class from server.py
  into agentflow_proxy/store.py. server.py imports them via `from agentflow_proxy.store import Store, utc_now, stable_json`.
  Metric: store.py is 141 lines; server.py reduced from 1226 to ~1095 lines; imports work.

- [DONE] Fix tool-result categorization: mixed-content turns not detected (2026-06-04)
  Details: categorize_request used `all(type == "tool_result")` but Claude Code injects system
  reminders as text blocks alongside tool_result blocks. This caused 0% tool-result routing —
  all tool-containing turns fell through to "tool-heavy" category. Fix: changed `all()` to `any()`
  so any turn where the last user message contains at least one tool_result block is categorized
  as "tool-result". Also added `category` field to routing_meta dict for observability in DB.
  Metric: mixed tool_result+text turns now categorized as tool-result and routed to Haiku;
  routing_json includes category field; non-tool-result turns unchanged.

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

- [DONE] Potential bug: dashboard cache savings looks implausibly high (2026-06-05)
  Details: Investigated 2026-06-05. Root cause: the three savings cards (routing, cache,
  prompt-cache) were computed over ALL TIME in stats_full() but displayed alongside today_calls
  and today_cost_usd which are today-only. 98.5M all-time Sonnet cache-read tokens → $273
  all-time savings shown next to $24 today cost. The formula itself was correct (no double-
  counting: actual_input_tokens from Anthropic API is non-cached tokens only, separate from
  cache_creation/cache_read). Fix: added today_ variants of each savings calculation and
  wired the dashboard cards to those. Today's prompt-cache savings (~$107) reconcile with the
  weekly table formula ($106.62) and are consistent with 37M Sonnet cache-read tokens today.
  Metric: confirmed — savings now reconcile with DB token totals and Anthropic pricing.

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

- [DONE] System prompt deduplication across calls (2026-06-03)
  Details: Many Claude Code calls repeat the same long system prompt. Cache the system prompt
  separately, assign it a hash, and use Anthropic's prompt caching beta header to mark it as
  cacheable at the API level. This doesn't reduce what we send but reduces what Anthropic charges.
  Implementation: add AGENTFLOW_PROMPT_CACHE env flag (default on). In the /v1/messages handler,
  after crunching, if system prompt is a string or list with total text > 4096 chars, transform it
  to include cache_control: {type: ephemeral} on the last text block. Add anthropic-beta:
  prompt-caching-2024-07-31 header. Parse cache_creation_input_tokens and cache_read_input_tokens
  from usage in responses. Store in new DB columns. Show prompt cache hit rate in stats.
  Metric: Anthropic cache_creation_input_tokens in response usage; cost reduction on repeated
  system prompts.

---

## P0 — Foundation (module extraction)

- [DONE] Extract crunch.py — crunch_body() and all text-reduction logic (2026-06-04)
  Details: Created agentflow_proxy/crunch.py (188 lines). Moved sha256_text,
  estimate_tokens_from_text, normalize_text, build_embedding, crunch_body,
  inject_prompt_cache, has_cache_control_blocks, TOKEN_CHARS, CRUNCH_ENABLED,
  CRUNCH_THRESHOLD_CHARS, PROMPT_CACHE_ENABLED, PROMPT_CACHE_MIN_CHARS.
  Metric: crunch.py 188 lines; server.py 788 lines; smoke test passes.

- [DONE] Extract pricing.py — MODEL_PRICES, MODEL_ALIASES, estimate_cost() (2026-06-04)
  Details: Created agentflow_proxy/pricing.py (39 lines). Moved MODEL_PRICES, MODEL_ALIASES,
  estimate_cost(). Duplicate MODEL_PRICES key for claude-opus-4-5 removed. server.py
  imports from pricing.py.
  Metric: pricing.py 39 lines; server.py 752 lines; smoke test passes.

- [DONE] Extract cache.py — cache constants, cache_key_for, response_output_text (2026-06-04)
  Details: Move CACHE_ENABLED, CACHE_TOOL_CALLS, SEMANTIC_CACHE_ENABLED,
  SEMANTIC_CACHE_THRESHOLD, cache_key_for(), and response_output_text() from server.py
  into agentflow_proxy/cache.py. server.py imports them. No behavior change.
  This is the last module extraction needed to complete architecture priority #1.
  Metric: cache.py exists; server.py drops by ~15 lines; smoke test passes on dev port 4001.

---

## P3 — Better Routing

- [DONE] Extract router.py — route_model() and all routing logic into its own module (2026-06-04)
  Details: Created agentflow_proxy/router.py (143 lines). Moved extract_text, has_tools,
  categorize_request, _load_routing_rules, route_model and routing constants. server.py drops
  from 1097 to 964 lines. Removed yaml import from server.py.
  Metric: router.py 143 lines; server.py 964 lines; smoke test passes; no behavior change.

- [DONE] YAML routing rules (2026-06-03)
  Details: Replace hardcoded thresholds in route_model() with a routing_rules.yaml file.
  Schema: list of rules with conditions (model_pattern, text_chars_lt, has_tools, category)
  and action (route_to, reason). Rules evaluated top-to-bottom, first match wins.
  Metric: routing config externalizable, no behavior change on existing defaults.

- [DONE] Phase-aware routing for agentic workflows (2026-06-04)
  Details: Within a session, classify each turn as: planning, tool-execution, verification,
  or summary. Use Haiku for tool-execution and summary turns, Sonnet for verification,
  keep requested model for planning.
  Phase signals: presence of tool_result blocks = execution; short final turn = summary;
  long system prompt + no prior context = planning.
  Implementation: added `tool-result` category in categorize_request() when the last message
  is a user role message containing ONLY tool_result blocks. Added routing rule in
  routing_rules.yaml: category=tool-result + model_pattern=sonnet → haiku.
  Metric: >20% of tool-heavy agentic calls routed to Haiku; no increase in error/retry rate.

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

- [DONE] Fix effort parameter stripping: deep-scan request body for nested "effort" key (2026-06-06)
  Details: Analysis 2026-06-06: 10+ Haiku-routed calls per day fail with HTTP 400
  "This model does not support the effort parameter." The stripping code (server.py ~line 238)
  checks `"effort" in crunched` (top-level only) and finds nothing — `stripped_params` is NULL
  in all affected routing_json records — yet Haiku still receives the parameter.
  The `effort` key is not at the top level of the request body as the proxy processes it.
  Two-part fix: (1) enable AGENTFLOW_LOG_BODIES=1 on dev during a cron run to capture the raw
  request structure and identify where `effort` lives; (2) replace the flat top-level check with
  a recursive deep-scan or targeted nested check (e.g., also inspect body["thinking"] for an
  "effort" subkey and strip it there). Also add `interleaved_thinking` to the incompatible list
  as a defensive measure.
  Metric: zero 400 "effort parameter" errors on Haiku-routed calls; routing_json stripped_params
  non-empty when effort-like params are removed.

- [IDEA] Strip historical thinking blocks before Haiku routing to unlock $33/day savings (2026-06-06)
  Details: Analysis 2026-06-06: 1,150 Sonnet calls/day (96%) are blocked from Haiku routing by
  uses_thinking() finding type=="thinking" blocks in message history. These calls have
  thinking_output_tokens=0 — no new thinking is occurring. They are tool-result turns from an
  ongoing thinking-session agent. Haiku rejects requests with thinking blocks in history.
  Potential fix: when routing to Haiku, strip type=="thinking" content blocks from all assistant
  messages in body["messages"] before forwarding. The tool_use/tool_result blocks and text context
  remain; only the model's internal reasoning chain is removed.
  Estimated savings: ~$0.029/call × 1,150 calls/day = ~$33/day.
  Risk: thinking blocks carry the model's reasoning chain. Stripping them could cause Haiku to
  lose coherence in complex multi-turn sessions. Gate behind AGENTFLOW_STRIP_THINKING_HISTORY
  (default: 0, off). Validate on a non-critical session before enabling in prod. Monitor error
  rate and retry count carefully.
  Metric: routing fires on thinking-history tool-result calls; daily cost drops by ~$30+; error
  rate does not increase with AGENTFLOW_STRIP_THINKING_HISTORY=1.

- [IDEA] Dashboard: correct crunch savings estimate to use cache-blended token rate (2026-06-06)
  Details: The dashboard shows crunch savings using full input rate ($3/MTok). For
  thinking sessions where 80-90% of input is served from prompt cache ($0.30/MTok),
  this overstates savings by ~10x. Fix: in stats_full(), compute a blended rate for
  crunch savings = (cache_read_input_tokens * 0.30 + actual_input_tokens * 3.0) /
  (cache_read_input_tokens + actual_input_tokens + 0.001) per MTok, then multiply
  tokens_saved_est by that blended rate instead of the flat input rate.
  Metric: crunch savings estimate within 20% of true value on thinking sessions.

- [DONE] Thinking budget throttle: configurable per-turn budget_tokens cap (2026-06-06)
  Details: Extended thinking sessions cost $44/day vs $5/day without thinking. No
  ceiling exists on per-turn budget_tokens. Add AGENTFLOW_MAX_THINKING_BUDGET_TOKENS
  (default: no cap). In server.py, after crunching, if body["thinking"]["budget_tokens"]
  > threshold, reduce to threshold and log in routing_json as "thinking_capped": true.
  This directly reduces output token cost for thinking calls. Start at 16000 as a safe
  cap vs typical 10000-50000 in traffic. Caveat: may reduce task quality.
  Metric: thinking output tokens/call decrease; no increase in task failure rate.

- [IDEA] Dashboard: per-session thinking token breakdown (2026-06-06)
  Details: The per-session table shows cost and call count but not thinking-token
  consumption. Add thinking_tokens and thinking_cost columns to the per-session panel
  (requires thinking_output_tokens column, active after next prod restart). Lets the
  user see which sessions are thinking-heavy and why costs spiked.
  Metric: per-session dashboard shows thinking_tokens and estimated thinking cost.

- [IDEA] Crunch: thinking block near-duplicate deduplication (2026-06-06)
  Details: Tool-result calls in thinking sessions average 118k chars; the near-duplicate
  detector covers type=="text" blocks but not type=="thinking" blocks. Thinking blocks
  can be 10-50k chars of reasoning repeated across turns. Before enabling, audit whether
  thinking blocks carry semantic state that later tool_use depends on. If safe, extend
  near-duplicate detection to type=="thinking" blocks at >0.95 jaccard (more conservative
  than the 0.85 used for text blocks). Estimate: additional 5-10% crunch ratio on
  thinking sessions, saving ~$0.30-0.60/day at cache-read rates.
  Metric: additional saved_chars on thinking sessions; zero tool-call failures.

- [DONE] Capture actual input tokens for streaming calls — currently 0 for ~400 calls/day (2026-06-06)
  Details: Analysis 2026-06-06: 392 successful streaming calls (status=200, stream=1) have
  actual_input_tokens=0. The streaming SSE path extracts output tokens from message_delta events,
  but input tokens may not be parsed from the message_start event (which carries usage.input_tokens).
  Without actual_input_tokens, cost_est_usd for streaming calls relies on the pre-call estimate
  (chars/4), causing a systematic undercount for calls where crunching changed the payload.
  Fix: in the streaming handler, parse the first SSE event (type="message_start") and extract
  usage.input_tokens. Store it to actual_input_tokens. This mirrors how the non-streaming path
  reads the response usage block.
  Metric: actual_input_tokens > 0 on >95% of streaming calls; cost_est_usd drift vs estimate
  shrinks for crunched streaming calls.
  Verified 2026-06-06: fix was already applied (parse_sse_usage in server.py parses message_start).
  281/281 (100%) of today's successful streaming calls have actual_input_tokens > 0.
  The 392 historical successful streaming calls with 0 were from June 2-5, mostly before
  the SSE parsing code was active in production. (2026-06-06)

- [IDEA] Dashboard: show prompt-cache write vs read ratio and cost to warm cache (2026-06-06)
  Details: Analysis 2026-06-06: prompt cache is the dominant savings driver — $24.5 savings on
  June 5, $15.8 on June 4. But cache write tokens cost 3.75× input rate; June 6 shows 523 M
  write tokens vs 17 M read tokens (ratio 30:1 write-heavy), meaning the cache may not yet be
  paying for itself on shorter sessions. Surfacing write/read ratio per session would help identify
  sessions that write large caches but terminate before the reads recover the cost.
  Fix: add a "Prompt cache write/read ratio" metric to the per-session breakdown in the dashboard.
  Show cache_creation_cost / cache_read_savings per session so expensive cache warmups are visible.
  Metric: dashboard shows per-session write/read cost ratio; identifies sessions where cache warmup
  was not recouped.

- [IDEA] Routing rule: route non-thinking Sonnet calls with text_chars 8k–30k and no tools to Haiku (2026-06-06)
  Details: Analysis 2026-06-06: 23 non-tool Sonnet calls in the 15k–30k char range and 11 in
  the 8k–15k range stayed on Sonnet (total ~34 calls, cost ~$0.27). These are "keep requested
  model" calls that passed the tool check but exceeded the 8k threshold raised in the last run.
  The large-context but non-tool calls in this band are most likely context-heavy tool_result
  processing that was not caught by the tool_result category, or long-context reads (grep output,
  file contents). Routing them to Haiku risks quality on complex reasoning but likely fine for
  read/summarize tasks. Suggest adding a new rule: non-tool Sonnet, category NOT in (code-gen,
  thinking), text_chars 8000–30000 → haiku. Gate behind a separate AGENTFLOW_ROUTE_MIDSIZE env
  flag initially, so it can be disabled if quality issues appear.
  Metric: 30+ additional calls/day routed to Haiku; monitor error rate and quality signals before
  making permanent.

- [IDEA] Per-session spending summary in daily log output (2026-06-06)
  Details: Analysis 2026-06-06: session 360d cost $44 on June 5 (217k tokens saved by prompt cache
  but still expensive). The session cost alert LOG WARNING fires at $5 threshold, but there's no
  end-of-day summary that lists total per-session cost with breakdown (thinking cost, routing
  savings, cache savings). This makes it hard to review unattended run impact without opening the
  dashboard.
  Fix: at server startup and on SIGTERM/graceful shutdown, emit a one-line summary per session
  active in the last 24h: session_id[:8], call count, actual cost, savings breakdown. Output to
  stderr so it's captured by the cron wrapper log.
  Metric: cron log shows per-session cost lines on shutdown; operator can audit cost without
  opening the dashboard.

- [DONE] Strip model-incompatible params when routing Sonnet→Haiku (2026-06-05)
  Details: Analysis 2026-06-05: a Sonnet→Haiku routed code-gen call returned 400
  "This model does not support the effort parameter". The `effort` param (sent by
  Claude Code CLI for Sonnet 4.6) is forwarded unchanged to Haiku, which rejects it.
  Fix: in server.py or router.py, after route_model() returns a different model than
  requested, strip known Haiku-incompatible top-level params from the request body
  before forwarding. Known incompatible: `effort`. Also strip `thinking` / `budget_tokens`
  as a defense-in-depth (uses_thinking() guards routing but stripping is a safe belt-and-
  suspenders for future code paths). Do not strip params for calls staying on their
  requested model. Log stripped params in routing_json for observability.
  Metric: zero 400 "does not support" errors on Haiku-routed calls; routing_json records
  stripped params when they are removed.

- [IDEA] Per-tier concurrency cap to prevent initial burst collisions (2026-06-05)
  Details: Rate limits at 7.6% (94/1233 calls today) despite global tier backoff. Pattern:
  4 concurrent agent calls hit the same tier simultaneously, all receive 429, and the
  backoff then serializes retries — but the initial burst already consumed quota. Per-call
  backoff doesn't prevent concurrent requests from colliding before any 429 is seen.
  Fix: add an asyncio.Semaphore per tier (haiku/sonnet) with a max of 2 concurrent
  forwarded requests. Requests queue behind the semaphore rather than racing to the API.
  This adds some latency for bursts but trades it for fewer exhausted-retry failures.
  The semaphore limit should be configurable via env var (AGENTFLOW_MAX_CONCURRENT_PER_TIER,
  default 2). No new dependencies required.
  Metric: 429 rate drops below 3%; retry_count=3 exhausted failures drop significantly.

- [IDEA] Capture error body for all non-200 responses (2026-06-05)
  Details: 6 of 12 today's 400 errors have error=NULL in the DB. These are the pre-fix
  "adaptive thinking not supported" cases but the error text wasn't saved. Audit the
  non-streaming error capture path: ensure the response body is read and stored in the
  `error` column for ALL non-200 status codes, not just some. If the current code only
  saves error text for certain paths (e.g. after raise), add a catch-all that reads
  response.text and stores it before returning the error to the client.
  Metric: zero non-200 calls with NULL error field going forward.

- [DONE] Retry on network errors (ConnectError/DNS) to fix ~15 transient 500s/day (2026-06-05)
  Details: Analysis 2026-06-05: 15 500s in last 2 days all have error="ConnectError('[Errno -3]
  Temporary failure in name resolution')", clustered at midnight during unattended cron runs. The
  current retry logic only handles 429/529 — network-level failures propagate immediately as 500.
  Fix: in both the streaming and non-streaming forwarding paths in server.py, catch httpx.NetworkError
  inside the while-True retry loop. On NetworkError, sleep 2s and retry up to 2 times before re-raising.
  Use a separate net_retries counter (don't conflate with rate-limit retry_count). Print a
  "network_error: ... retry=N" log line for observability.
  Metric: zero ConnectError 500s on subsequent runs; net_retries logged when retries fire.

- [DONE] Route tool-light Sonnet calls to Haiku (2026-06-05)
  Details: 4 tool-light calls/day on Sonnet at avg 9247 chars. Category "tool-light" = has tools
  but <16k chars and last message is not a tool_result turn. These are small tool-setup or short
  responses that don't need Sonnet reasoning power. Add routing_rules.yaml rule:
  model_pattern=sonnet, category=tool-light → route_to: haiku.
  Metric: tool-light calls routed to Haiku; no increase in error rate.

- [IDEA] Raise small-Sonnet-→-Haiku text threshold from 6000 to 10000 chars
  Details: 25 code-gen calls/day on Sonnet, avg 7010 chars, none routed. Current non-tool
  Sonnet rule fires at text_chars_lt: 6000. Raising to 10000 would catch these code-gen calls.
  However, code quality from Haiku may be lower — needs monitoring. Consider adding category
  exclusion: only route if category != "code-gen" to avoid routing code generation.
  Metric: more non-tool Sonnet calls routed; no regression in code quality signals.

- [DONE] Normalize dot-notation model aliases before forwarding (2026-06-02)
  Details: Dot-notation aliases (claude-haiku-4.5, claude-sonnet-4.5, claude-opus-4.5) reach
  Anthropic unchanged and return HTTP 404. Normalize them in the handler before forwarding:
  claude-haiku-4.5 → claude-haiku-4-5-20251001, claude-sonnet-4.5 → claude-sonnet-4-5-20240620,
  claude-opus-4.5 → claude-opus-4-5. Tester confirmed 16 prod calls failing with 404.
  Metric: zero 404s from dot-notation aliases; router still routes correctly after normalization.

- [DONE] Fix routing rule: max_tokens_lte condition blocks Sonnet→Haiku routing (2026-06-03)
  Details: The Sonnet→Haiku rule requires `max_tokens_lte: 2048`, but Claude Code never sets
  max_tokens in requests — all 59 inspected non-tool Sonnet calls had max_tokens=NULL. NULL<=2048
  is false, so the rule never fires. Only 1 of 1,323 calls was rerouted (0.08%).
  Fix: remove `max_tokens_lte` from the Sonnet→Haiku rule in routing_rules.yaml, or update the
  rule evaluator in agentflow_proxy/server.py to treat a missing max_tokens_lte condition as always-true (already
  the case) AND treat an explicit max_tokens_lte condition as matching when request max_tokens is
  absent (i.e., treat NULL max_tokens as unconstrained). The latter is safer. 37 calls with
  text_chars<6000 and no tools should have gone to Haiku.
  Metric: >30 calls/day routed to Haiku on small non-tool Sonnet requests; routing_json.reason
  shows "small non-tool Sonnet request routed to Haiku"; zero increase in error rate.

- [DONE] Fix: streaming handler drops prompt cache stats when Claude Code injects cache headers (2026-06-03)
  Details: inject_prompt_cache() returns (body, False) when the incoming request already contains
  cache_control blocks (Claude Code sends them). The streaming handler (server.py ~line 583) then
  skips cache stat capture because `if prompt_cached:` is False. Result: all 1,323 calls have
  cache_creation_input_tokens=0, cache_read_input_tokens=0 even though Anthropic is almost certainly
  returning non-zero values for the large tool-heavy calls (avg 25k chars each).
  Fix: in the streaming handler, move cache token extraction to an unconditional block that always
  reads cache_creation_input_tokens and cache_read_input_tokens from the final SSE usage event —
  mirror what the non-streaming path does at lines 698-700. The `if prompt_cached:` guard should
  only control whether AgentFlow adds the beta header, not whether we capture the stats.
  Metric: cache_creation_input_tokens > 0 on calls with large system prompts; dashboard prompt
  cache stats become non-zero; confirms or disproves actual prompt cache activity.

- [DONE] Fix session tracking: 70% of calls have null session_id (2026-06-03)
  Details: Only 403 of 1,323 calls (30%) have a session_id. One session accounts for all of them.
  The other 920 calls from Claude Code CLI/API never receive a session_id. Current extraction likely
  depends on a header that is only present in certain connection modes.
  Fix: audit the session_id extraction logic — check x-session-id, x-request-id, and client IP:port
  extraction. Consider assigning a synthetic session_id based on a short time window + client IP
  when no header is present (e.g., calls within 5 minutes from the same IP share a session).
  Metric: >80% of calls have a non-null session_id; dashboard per-session cost table shows multiple
  active sessions.

- [DONE] Request pacing: 6.5% error rate from 429/529 bursts during agent runs (2026-06-03)
  Details: 68 rate limits (429) and 18 overloaded (529) across 2 days. Pattern: 26 rate limits
  in one hour during a June 2 agent run, 10 in another. Both Haiku and Sonnet affected. Agent runs
  send concurrent requests that collide in the same rate-limit window. Each 429 likely triggers a
  client retry, doubling request count and cost for failed turns.
  Fix: add AGENTFLOW_MIN_REQUEST_INTERVAL_MS (default 0, so no-op unless set) that imposes a
  minimum delay between forwarded requests. Add exponential backoff with jitter when proxying a
  429/529: wait and re-forward rather than returning the error immediately to the client. Cap at
  3 retries. Log backoff events in the calls table (new column: retry_count).
  Metric: 429+529 rate drops below 1%; no increase in p95 latency for successful calls.

- [DONE] Rate-limit fallback routing: retry with original model on Haiku/routed-model 429s (2026-06-04)
  Details: 116 429 errors in last 24h (16.3%) — well above the <1% target. Phase-aware routing
  now funnels tool-result turns to Haiku, causing bursts on a model with lower rate limits.
  Current backoff retries the SAME routed model (e.g., Haiku) up to 3 times with exponential
  backoff, but if Haiku is broadly rate-limited during an agent burst, all retries fail too.
  Pattern: retry_count=3 calls with 429 status in the DB have ~12s latency = all 3 retries burned.
  Fix: in server.py, when a 429/529 occurs on a call where AgentFlow applied routing (routed_model
  != requested_model), on the FIRST retry switch to the originally-requested model instead of the
  routed one. Subsequent retries also use the requested model. Update routing_json to record the
  fallback: add "fallback_reason": "rate_limited". This prevents Haiku saturation from cascading
  into full failure — at worst we pay Sonnet prices instead of failing the call entirely.
  For Sonnet 429s (where the requested model IS Sonnet), continue retrying as-is since there's
  no cheaper fallback target that would help with rate limits.
  Metric: 429+529 rate drops below 2%; retry_count=3 failures drop significantly; routing_json
  shows fallback_reason=rate_limited for affected calls.

- [DONE] Expand request categorization: 70% of calls uncategorized (2026-06-03)
  Details: Only tool-heavy, code-gen, short-completion are assigned. 930 of 1,323 calls get
  category=NULL. Chat and long-context categories have no heuristics. Routing rules that target
  category can't function on uncategorized calls.
  Fix: add heuristics for chat (no tools, no code fences, < 2000 chars) and long-context (> 20k
  chars regardless of tools). Verify coverage against DB after deploy.
  Metric: <20% NULL category across new calls; category breakdown visible in dashboard.

- [DONE] Fix cost_est_usd: missing prompt-cache token costs understates actual bill by ~4.4× (2026-06-04)
  Details: estimate_cost() receives only actual_input_tokens (uncached portion, avg 281 tokens)
  and actual_output_tokens. It ignores cache_creation_input_tokens and cache_read_input_tokens.
  Over 24 h: 3.08 M cache_creation tokens (cost 1.25× input rate = $11.54) and 49.2 M
  cache_read tokens (cost 0.10× input rate = $14.77) are invisible to the cost model.
  Dashboard shows $7.75/day; actual Anthropic bill is ~$34/day.
  Fix: add optional cache_creation and cache_read parameters to estimate_cost() in pricing.py.
  In both streaming and non-streaming handlers, pass the DB-stored cache token counts when
  computing cost_est_usd and cost_baseline_usd. Also fix cost_baseline_usd to include what
  the cache tokens would have cost without prompt caching (baseline = all tokens at full price).
  Metric: cost_est_usd within 5% of actual Anthropic invoice; dashboard shows real $ saved
  by prompt cache as a separate line item.

- [DONE] Dashboard: add prompt-cache savings line to cost summary (2026-06-04)
  Details: Prompt cache is saving ~$133/day (49.2 M cache_read tokens × $2.70 savings per MTok
  vs full input price) but this is invisible in the dashboard. The "savings" section only shows
  routing savings (currently $0 since routing never fires).
  Fix: after fixing cost_est_usd, add a "Prompt cache saved" row to the dashboard summary
  section: SUM(cache_read_input_tokens) × (input_price − cache_read_price). Also show prompt
  cache hit rate (calls where cache_read_input_tokens > 0 / total calls).
  Metric: dashboard shows prompt cache savings ≥ $100/day; cache hit rate ≥ 40%.

- [DONE] Tool-result turns: verify routing fires after prod restart and add coverage metric (2026-06-05)
  Details: Verified 2026-06-05: 10 of 39 tool-result Sonnet calls routed to Haiku (26%).
  The remaining 29 are thinking requests (reason="keep requested model for thinking request",
  text_chars ~100k) — correctly kept on Sonnet since Haiku lacks extended-thinking support.
  Effective non-thinking routing rate: 10/10 = 100%.
  Metric: met — routing fires correctly; unrouted calls all have valid thinking-mode reason.

- [DONE] Global tier backoff using retry-after header to fix persistent ~5% rate-limit rate (2026-06-05)
  Details: Rate limits remain at ~5% (40 in last 24h, 40 exhausting all 3 retries). Pattern:
  agent bursts send 4 concurrent requests simultaneously; all hit 429 and all retry independently
  without coordination, so retries collide again. Per-call exponential backoff does not help when
  concurrent requests share the same overloaded tier.
  Fix: track a per-model-tier global backoff timestamp. When any request receives a 429, read the
  `retry-after` header (or default to 60s) and set a global asyncio.Event / timestamp for that
  tier. All subsequent forwarded requests to that tier check the global backoff and await it before
  sending. A simple asyncio.Lock per tier (haiku/sonnet) is enough — no new dependencies.
  Metric: 429+529 rate drops below 2%; retry_count=3 exhausted-retry calls drop significantly.

- [DONE] Remove max_tokens_lte from small-Sonnet-→-Haiku routing rule (2026-06-05)
  Details: routing_rules.yaml still contains `max_tokens_lte: 2048` on the small non-tool
  Sonnet rule. The evaluator treats absent max_tokens as unconstrained (matching), so this
  condition only blocks calls that explicitly set max_tokens > 2048. Currently 100% of Sonnet
  calls have has_tools=true so the rule can't fire anyway, but once tool-result routing
  separates tool-result turns the non-tool rule becomes relevant again.
  Fix: remove the max_tokens_lte condition from the Sonnet→Haiku rule in routing_rules.yaml.
  Metric: rule fires on short, non-tool Sonnet calls (any that appear); no new errors.

- [DONE] Fix tool-result routing: 100% failure rate when sessions have thinking blocks in message history (2026-06-05)
  Details: 10/10 tool-result turns routed to Haiku today returned HTTP 400 with
  "adaptive thinking is not supported on this model". Root cause: uses_thinking() in router.py
  checks only the top-level body["thinking"] param. When a session uses extended thinking,
  Sonnet's responses include {"type":"thinking","thinking":"..."} content blocks in assistant
  messages. On the next tool-result turn the client may omit or disable the top-level
  "thinking" param (no new thinking needed), but the message history still contains those
  blocks. Haiku rejects the request on receipt.
  Fix: extend uses_thinking(body) to also scan body["messages"] for any assistant message
  whose content list contains a block with type=="thinking". If found, treat the request as
  a thinking session and keep it on the requested model.
  Example check: any(isinstance(b,dict) and b.get("type")=="thinking" for msg in
  body.get("messages",[]) if isinstance(msg.get("content"), list) for b in msg["content"])
  Metric: zero 400 errors with "adaptive thinking" message; tool-result routing fires without
  failure on non-thinking sessions; routing_json reflects correct reason.

- [DONE] Investigate zero exact-match cache hit rate (2026-06-05)
  Details: Investigated 2026-06-05. Findings: (a) 92% of calls are streaming — streaming
  path returns early before cache lookup (by design, line 207 comment). (b) CACHE_TOOL_CALLS
  defaults to "0" — tool-heavy and tool-result calls (majority of non-streaming) skip cache.
  (c) SEMANTIC_CACHE_ENABLED defaults to "0" — semantic cache is disabled entirely.
  (d) 18 cache entries exist but never match because every request body contains unique
  session/message history context. Conclusion: caching is structurally inapplicable to
  current streaming + tool-heavy traffic. Exact cache would only help for repeated identical
  short non-tool requests, which don't occur in agentic workflows. The streaming cache IDEA
  would be the path to meaningful hit rates; file-watch invalidation would be a prerequisite.
  Metric: confirmed — caching is structurally inapplicable to current traffic pattern.

- [DONE] Dashboard: show per-session cost and phase breakdown for today (2026-06-05)
  Details: Session tracking is working (3 sessions identified, largest at $8.60 over 168 calls
  in one day). The dashboard currently shows per-session cost but not phase breakdown (how
  many calls were tool-result, tool-heavy, thinking, etc. per session). Adding this would
  make it easier to spot sessions with unusually high thinking overhead or tool-call density.
  Metric: per-session panel shows call count by category; helps identify expensive agentic
  patterns worth targeting with routing or crunching rules.

- [DONE] Capture error body for all non-200 responses (2026-06-05)
  Details: 5 non-200 calls today have NULL error field. 4 are pre-fix thinking-routing 400s;
  1 is a post-restart code-gen→Haiku 400 at 18:18 UTC with completely unknown cause because
  the error body was not stored. Audit the non-streaming error capture path: ensure the
  response body is read and stored in the `error` column for ALL non-200 status codes.
  Two gaps fixed: (1) non-streaming non-JSON error responses now log to calls before returning;
  (2) streaming finally block now sets error="upstream_error: status=N" when status >= 400 and
  no exception was raised.
  Metric: zero non-200 calls with NULL error field going forward; the 18:18 Haiku 400 root
  cause can be identified from DB after fix.

- [DONE] Implement per-tier concurrency cap (2026-06-05)
  Details: Error rate at 6.9% (7-day), 6.4% today despite global tier backoff. 8 calls today
  burned all 3 retries (avg 30.9s latency). Some 429 responses carry retry-after of 2769s
  (46 min) — tier is being fully exhausted before backoff can prevent collisions. Fix: add
  asyncio.Semaphore per tier (haiku/sonnet) with default 2 concurrent forwarded requests;
  configurable via AGENTFLOW_MAX_CONCURRENT_PER_TIER. Queue behind semaphore rather than
  racing to the API.
  Metric: 429+529 rate drops below 3%; retry_count=3 exhausted failures drop significantly.

- [DONE] Dashboard: thinking budget visibility as a cost line item (2026-06-06)
  Details: Added `thinking_output_tokens` column to the `calls` table. For non-streaming
  calls: thinking chars parsed from response body content blocks. For streaming calls:
  `thinking_delta` SSE events accumulated during stream, token count estimated at TOKEN_CHARS
  chars/token. stats_full() now returns thinking_output_tokens, today_thinking_output_tokens,
  thinking_cost_usd, today_thinking_cost_usd (computed per model via estimate_cost). Dashboard
  summary panel gains a yellow "Thinking cost today" card showing today's thinking token count
  and estimated cost.
  Metric: stats_full returns thinking_output_tokens and thinking_cost_usd > 0 when the DB has
  thinking-session rows; dashboard shows Thinking cost line item.

- [DONE] Session cost alert: log warning when session exceeds daily $ threshold (2026-06-06)
  Details: Analysis 2026-06-05: one session cost $34 of today's $35 bill. The user has no
  real-time signal that a session is burning unusually high budget. Fix: in server.py, after
  storing each call, check the rolling per-session cost for today. If it exceeds a configurable
  threshold (AGENTFLOW_SESSION_COST_ALERT_USD, default 5.0), log a WARNING line with session
  ID, current cost, and call count. This is a local log-only alert — no external notification,
  no SaaS dependency. The threshold should default to a high-enough value ($5) that it only
  fires for genuinely expensive sessions, not normal usage.
  Metric: WARNING log emitted when a session crosses the threshold; visible in server logs
  during unattended runs; does not fire on typical <$1 sessions.

- [DONE] Raise small-Sonnet-→-Haiku text threshold from 6000 to 8000 chars for code-gen (2026-06-06)
  Details: Analysis 2026-06-05: the small-non-tool rule fires at <6000 chars. Post-restart,
  2 code-gen calls at 6614 and 9048 chars stay on Sonnet. Raising the threshold to 8000
  (not 10000 as previously IDEAd) would catch the 6614-char call without reaching into the
  range where code quality risk is unknown. Exclude category=code-gen from an 8k-10k range
  until Haiku code quality on that window is measured, or add a separate code-gen rule with
  a lower confidence threshold.
  Metric: additional code-gen calls routed at 6k-8k chars; no increase in error rate or
  quality complaints on short code-gen tasks.
