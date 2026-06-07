# AgentFlow North Star

## What This Is

AgentFlow is a local Anthropic-compatible proxy that sits between Claude Code / Claude CLI
and the Anthropic API. Its job is to continuously reduce the cost of running AI workloads
without degrading result quality — and to improve its own ability to do that over time,
driven by an autonomous agentic development loop.

See `ARCHITECTURE.md` for the target architecture contract. In short: the local Python module
must provide the Claude middleware, read-only dashboard, and local manual rules; a separate
future managed optimizer can later serve paying users with better routing, crunching, and
broader cache/policy learning.

The product split is intentional: the free local package should provide transparent low-level
controls and conservative deterministic savings, roughly the first 20% of the attainable value.
The premium managed optimizer should provide the strongest savings through learned policies,
quality evaluation, and broader cache/policy intelligence that one local install cannot build
well on its own.

## The Three Core Levers

### 1. Prompt Crunching
Reduce the number of tokens sent upstream by making context smaller before it leaves the proxy.
Current state: conservative whitespace normalization + exact-duplicate block removal.

Target capabilities (in order of ambition):
- Whitespace normalization (done)
- Exact duplicate block removal (done)
- Long block head/tail shortening (done)
- Semantic deduplication: detect near-duplicate blocks and merge them
- Model-assisted summarization: use Haiku to summarize old context turns before sending to Sonnet/Opus
- Agentic workflow phase compression: detect repeated tool scaffolding and strip it
- Instruction deduplication: detect system prompt sections repeated verbatim across calls

### 2. Model Routing
Route each request to the cheapest model that can handle it correctly.
Current state: static thresholds — small non-tool requests go to Haiku, medium Opus goes to Sonnet.

Target capabilities:
- Configurable YAML routing rules (replace hardcoded thresholds)
- Request complexity classification: use content signals (code? math? tool-heavy?) to route better
- Per-session routing memory: if a session is doing file ops, it probably needs Sonnet minimum
- Agentic phase detection: planning turns vs. execution turns vs. summary turns each have different model needs
- Outcome-based routing refinement: track quality signals and tighten downgrade thresholds over time
- A/B routing experiments: run fraction of traffic through cheaper model and compare outcomes

### 3. Caching
Serve repeated requests from cache instead of hitting the API.
Current state: exact SHA-256 hash cache for non-streaming, non-tool requests.

Target capabilities:
- Exact cache (done)
- Semantic cache: embed requests, match by cosine similarity above threshold (requires embedding model)
- Partial cache: cache expensive substrings (long system prompts) and stitch responses
- Session-scoped cache: within one Claude Code session, repeated identical reads are safe to cache
- TTL and invalidation: file-watching to invalidate cache when underlying files change
- Streaming cache: buffer, store, and replay streamed responses

## The Dashboard

A read-only live web UI at `http://localhost:4002/agentflow/dashboard` showing:

- Real-time request feed (model, route decision, cache hit/miss, latency, cost)
- Cost totals: what was actually spent, what would have been spent without routing/crunching
- Savings breakdown: $ saved by routing, $ saved by crunching, $ saved by caching
- Cumulative savings over time (chart)
- Top patterns: most common request shapes, cache hit rates by category
- Routing distribution: where traffic ended up vs. where it was requested
- Crunch efficiency: average token reduction %, by model tier
- Recent agent run summaries: what the orchestrator did last cycle

## Data Model Goals

All requests should be stored with enough detail to:
1. Replay and re-analyze past traffic
2. Measure quality of routing decisions
3. Identify new optimization opportunities
4. Train future heuristics

Key fields to capture:
- Raw request (opt-in, local only) + crunched request
- Actual tokens used (from Anthropic response headers, not estimates)
- Wall-clock latency
- Full routing decision chain with reasons
- Session ID (group Claude Code turns into sessions)
- Whether response was used (tool calls that complete vs. are abandoned)

## Agentic Workflow Focus

Standard chatbot workloads are easy to optimize. The real prize is **agentic workflows**
(Claude Code, multi-step coding agents, research agents) because:

- They generate large, repetitive context windows
- Different phases have very different model requirements:
  - **Planning**: needs Sonnet/Opus reasoning quality
  - **Tool execution**: simple read/write ops can often use Haiku
  - **Verification**: moderate complexity, Sonnet-range
  - **Summary/reporting**: low complexity, Haiku
- They make many repeated calls with nearly identical context
- They're long-running, so caching and crunching compound

Target: phase-aware routing for agentic sessions that cuts per-session cost by 40-60%
without reducing task completion rate.

## The Agentic Development Loop

AgentFlow improves itself. Every two hours, a Claude Code orchestrator is invoked via cron.
It reads current state, decides what to work on, spawns focused sub-agents, and commits
the result. Human intervention is needed only to approve major architectural decisions
or unblock the agent when it gets stuck.

### Orchestrator Responsibilities
1. Read current stats from the proxy DB — understand recent traffic patterns
2. Read GitHub Issues — know what's planned and prioritized
3. Decide: implement a backlog item, analyze for new opportunities, or run evals
4. Invoke the right sub-agent(s) with focused prompts
5. Update GitHub Issues with results and new findings
6. Commit all changes with clear messages
7. Write a run summary to `runs/YYYY-MM-DD_HH-MM.md`

### Sub-Agents
Each sub-agent is a vanilla `claude -p` invocation with a focused prompt and appropriate tools.
No framework, no SDK wrapping — just Claude with context.

- **analyze**: reads DB stats, identifies optimization opportunities, creates GitHub Issues
- **develop**: implements a specific backlog item end-to-end (code + basic test)
- **test**: runs the test suite, validates proxy behavior, reports regressions
- **research**: searches for new crunching/routing/caching techniques, creates GitHub Issues
- **dashboard**: implements or improves the dashboard UI

### Human Touchpoints
- Review `runs/` summaries to see what was done
- Edit GitHub Issues to reprioritize or add items
- Restart proxy after significant changes
- Occasionally review the dashboard to spot issues

## Success Metrics

Primary: **$ saved per 1000 API calls** (tracked in dashboard)

Secondary:
- Cache hit rate (target: >20% on steady-state Claude Code sessions)
- Average token reduction from crunching (target: >15% on large contexts)
- Routing downgrade rate (what % of Opus requests run on Sonnet or less)
- Zero regression rate (no broken tool calls attributable to proxy changes)

## Constraints and Principles

1. **Correctness over savings**: a broken tool call costs more than it saves. When in doubt, pass through unchanged.
2. **Transparency**: every transformation is logged with the reason. The dashboard shows everything.
3. **Local first**: no data leaves the machine except through the Anthropic API. The proxy stores data locally only.
4. **Incremental**: each orchestrator run should make one small improvement, not a big rewrite.
5. **Reversible**: every change is committed to git. Rollback is always available.
6. **Self-measuring**: any new optimization must be accompanied by a metric that proves it works.
