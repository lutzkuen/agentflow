# Client/server responsibility boundary

TokenClaw has two intentionally separate roles:

- **TokenClaw client**: local proxy, local telemetry, dashboard, cache mechanics,
  crunch execution, local hard gates, explicit user rules, managed-server
  communication when opted in, safe fallback when the server is unavailable, and
  metadata-only outcome reporting.
- **`tokenclaw_server`**: policy brain for measurement contracts, route and
  action proposals, canary and holdout sizing, routing/crunch/cache policy
  discovery, research rollups, lifecycle learning, signing, expiry, and tenant
  policy.

The client is the boring executor. The server is the policy brain.

## Local client owns

- Provider-compatible proxying for Anthropic-compatible and OpenAI-compatible
  traffic.
- Provider credential handling and upstream forwarding from the user's machine.
- Local SQLite telemetry, local dashboard views, and local privacy controls.
- Exact cache lookup, storage, replay, bypass, and invalidation mechanics.
- Crunch execution and request mutation mechanics.
- Explicit local routing, crunch, and cache hard rules from local files.
- Opt-in/opt-out flags, local hard gates, verification of managed decisions, and
  fallback to local policy when managed communication is unavailable or unsafe.
- Metadata-only outcome feedback after local execution.

## Managed server owns

- The measurement contract that tells the client which derived metadata is
  useful to collect.
- Feasible route and action proposals from metadata, not raw provider bodies.
- Canary, holdout, and rollout sizing.
- Routing, crunch, and cache policy discovery.
- Research rollups, lifecycle learning, promotion criteria, and safety stop
  analysis.
- Server-side policy signing, expiry, tenant policy, and managed billing.

## Future request flow

1. The client starts local-only. Managed communication is disabled unless the
   user opts in and configures a server URL.
2. The client requests or caches a server-owned measurement contract.
3. For eligible provider requests, the client collects only requested derived
   metadata and hashed grouping identifiers.
4. The server returns a signed, expiring, metadata-only policy decision.
5. The client verifies provenance and compatibility, checks local opt-in flags,
   applies local hard gates, and executes only supported local actions.
6. The client forwards the provider request itself.
7. The client reports metadata-only outcomes such as status, latency, token
   usage, retry/fallback counts, applied families, vetoed families, and policy
   ids.

## Privacy boundary

Managed server decisions must remain metadata-only. The client must not send the
server provider request bodies, provider response bodies, raw prompts, raw
responses, secrets, local file paths, cache keys, raw session identifiers, raw
request ids, or request-replay payloads.

If a proposed optimization would require server-generated provider content or a
server-side body rewrite, the server should omit that action and explain why.
The local client may still apply explicit local rules when the user configured
them.

## Client non-goals

- Route discovery and learned/adaptive routing policy.
- Savings research bench behavior or cross-install optimization learning.
- Managed policy candidate generation.
- Hosted account, tenant, billing, fleet policy, or organization-wide shared
  cache ownership.
- Server-side provider forwarding or server-generated provider request content.
