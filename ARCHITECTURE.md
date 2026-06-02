# AgentFlow Target Architecture

This document is the product and architecture contract for unattended agent runs. When a
backlog item is ambiguous, prefer the shape described here.

## Product Shape

AgentFlow has two related products:

1. **Local AgentFlow module**: a Python package that runs on the user's machine as a local
   Anthropic-compatible middleware for Claude Code and Claude CLI.
2. **Future managed optimizer**: a separate server for later paying users. It receives
   telemetry and policy inputs from opted-in local installations, learns better routing and
   crunching strategies, and maintains a broader cache/policy base.

The local module must remain useful and safe without the managed optimizer. The future server
is an optional extension point, not a dependency for basic Claude middleware behavior.

## Runtime Topology

| Component | Bind | Purpose | Exposure |
|-----------|------|---------|----------|
| Local Claude proxy | `127.0.0.1:4000` | Anthropic-compatible `/v1/messages` middleware for Claude | Localhost only |
| Local dev proxy | `127.0.0.1:4001` | Development and smoke testing | Localhost only |
| Read-only dashboard | `0.0.0.0:4002` | LAN-visible observability dashboard | Read-only, no Claude endpoint |
| Future managed optimizer | Hosted separately | Shared policy, evaluation, optional cache metadata | Authenticated, tenant-aware |

Never expose the local Claude proxy endpoint on the LAN. It forwards real Anthropic
credentials and request bodies upstream.

## Local Module Responsibilities

The local module owns the user-facing middleware experience:

- Run the Anthropic-compatible proxy.
- Preserve Claude Code compatibility.
- Store local call logs and cache data in SQLite by default.
- Provide the read-only dashboard.
- Provide manual controls for:
  - model-selection rules,
  - crunching rules,
  - exact-match hash cache rules.
- Apply only conservative defaults. Savings must not come at the cost of broken tool calls.
- Work fully offline except for forwarding the user's Anthropic requests upstream.

Manual controls should be stored locally in versionable files, not hidden in the database:

```text
config/
  routing_rules.yaml
  crunch_rules.yaml
  cache_rules.yaml
```

The dashboard may later edit these files through a local-only admin surface, but the current
LAN dashboard must stay read-only.

## Local Module Boundaries

The local module should not become a SaaS backend. It should not implement billing, tenant
management, hosted user accounts, organization-wide shared caches, or fleet-wide learning.

It may define interfaces for a future optimizer, for example:

- export anonymized aggregates,
- import signed routing/crunch/cache policy bundles,
- compare local outcomes against recommended policies,
- maintain a local policy version and rollback history.

Any cloud or managed-server integration must be opt-in and must default to off.

## Future Managed Optimizer Responsibilities

The future managed server is a separate service for paying users. Its job is to improve
optimization quality beyond what one local installation can learn alone.

It should provide:

- better model routing based on aggregate outcomes,
- better crunching strategies and policy evaluation,
- a wider base for safe cache hits and repeated prompt structures,
- tenant-aware policy distribution,
- fleet-level dashboards and billing,
- privacy controls for what telemetry may leave a local machine.

The managed server should not require raw prompts for its first useful version. Prefer
aggregates, hashes, request categories, policy IDs, model decisions, token counts, latency,
error rates, and outcome signals.

## Policy Interfaces

Build routing, crunching, and caching around explicit policy interfaces:

- `router`: requested model plus request features -> routed model plus reason.
- `cruncher`: request body plus crunch rules -> transformed body plus before/after metrics.
- `cache`: normalized request plus cache rules -> exact hit, miss, or bypass plus reason.
- `store`: append-only call log and cache metadata.
- `dashboard`: read-only views over store and policy state.

Every decision must produce machine-readable metadata. The dashboard and analyzer should be
able to answer: what changed, why, how much it saved, and whether errors increased.

## Development Priorities

Near-term unattended runs should build toward this order:

1. Split the local proxy into testable modules without changing behavior.
2. Add file-backed manual rules for routing, crunching, and exact-match caching.
3. Show effective rules and decision reasons in the dashboard.
4. Add safe local edit/reload flow for rules, while keeping the LAN dashboard read-only.
5. Define export/import policy bundle shapes for the future managed optimizer.
6. Only then prototype the managed optimizer server.

Do not start by building the managed server. The local module needs clean interfaces first.

## Safety Rules

- Correctness beats savings.
- Local Claude proxy stays localhost-only.
- LAN dashboard is read-only.
- Body logging stays opt-in.
- Tool-call caching stays off unless invalidation is proven safe.
- Any managed-server communication is opt-in and documented.
- Each optimization needs metrics before it is treated as successful.
