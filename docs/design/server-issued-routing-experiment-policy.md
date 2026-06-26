# Design note: server-issued routing-experiment / canary policy

Status: proposed · Scope: cross-repo (`tokenclaw` proxy + `tokenclaw_server`)

## Problem

The routing **experiment setup** — which route-down candidates to shadow, at what
sample fraction, with what budget, eligibility, and safety stops — currently lives in a
local `routing_experiments.yaml` that the proxy loads **once at process start**. The live
decision path (`routing_experiment_decision` → `_effective_experiment_controls`) reads the
in-memory `ROUTING_EXPERIMENT_POLICY` global and never re-reads the file, so:

- Changing the experiment plan (e.g. raising `max_text_chars` so large Opus calls become
  shadow-eligible) requires a **prod restart**; edits do not hot-apply.
- The plan is **static** — it cannot adapt as evidence accrues (raise/lower sampling, open a
  candidate, tighten a budget, trip a safety stop).
- It is **architecturally misplaced.** `ARCHITECTURE.md` assigns *"Canary management —
  deciding when to shadow or A/B a candidate route, at what fraction, with safety stops, and
  promoting only on evidence"* to `tokenclaw_server`. The local module is meant to *apply*,
  not *decide*. The YAML header already says so: *"Managed servers discover and mint new
  pathways; the local client keeps only guardrails, explicit preferences, and a small
  deterministic fallback."* The existing **backed-or-off** gate points the same way: the
  local proxy must not mint canaries without server backing.

## Goal

The server **owns and issues** the routing-experiment / canary policy at runtime; the proxy
**fetches, caches, and applies** it (hot, no restart). The local YAML collapses to
**guardrails + a tiny deterministic fallback** so the proxy stays useful with the server off.
This is delivery only — the proxy still *executes* the shadow request; only *who decides the
plan* moves.

## Non-goals

- Not removing local guardrails (blocklist / preferred pathways) or the off-server fallback;
  local must remain useful and safe with the server switched off (architecture requirement).
- Not changing shadow *execution* (the proxy still sends the counterfactual request and logs
  the comparison) or the outcome-feedback channel (`/v1/policy-events`).
- Not raw-prompt egress: the issued policy is metadata/policy, not request content.

## Design

### Delivery (reuse the client-contract pattern)
Issue the policy as a **signed, TTL'd bundle** over the existing managed channel — the same
machinery as `/v1/client-contract` (fetch → validate → cache by scope → hot-apply, with a
local fallback when unreachable/expired). Either a new `/v1/routing-experiment-policy`
endpoint or an extension of the policy-bundle surface (`tokenclaw_server/policy_bundles.py`).
Per-request candidate *forcing* already exists via the policy-decision response
(`selected_for_shadow_evaluation` / `shadow_model`); this bundle supplies the standing
candidate matrix + controls that today live in YAML.

### Envelope (sketch)
```jsonc
{
  "schema": "tokenclaw.routing_experiment_policy.v1",
  "version": "1",
  "generated_at": "...", "expires_at": "...",            // TTL'd
  "provenance": { "signed": true, "algorithm": "hmac-sha256", "signature": "..." },
  "controls": {
    "enabled": true, "kill_switch": false,
    "sample_rate": 0.1, "daily_budget_usd": 10.0,
    "min_text_chars": 0, "max_text_chars": 0             // 0 = unlimited
  },
  "candidates": [
    {
      "candidate_id": "anthropic-opus48-to-sonnet46-chat",
      "requested_model": "claude-opus-4-8", "routed_model": "claude-sonnet-4-6",
      "provider": "anthropic", "source_surface": "anthropic_messages",
      "app_family": "claude_code", "category": "chat",
      "eligibility": { "min_text_chars": 0, "max_text_chars": 0, "stream": false },
      "sample_rate": 0.1, "sample_weight": 1, "daily_budget_usd": 5.0
    }
  ],
  "safety_stops": [ /* error-rate / similarity / spend thresholds that disable a candidate */ ],
  "privacy_summary": { "metadata_only": true, "raw_payload_included": false }
}
```

### Decision precedence (the core contract)
`routing_experiment_decision` resolves the active plan per request in this order:

1. **Server-issued policy** — when the managed feed is backed, active, and unexpired, it is
   authoritative for candidate selection, fraction, eligibility, and budget.
2. **Local guardrails** — `blocklist` and explicit `preferred_pathways` are always applied as
   a **local veto/override** on top of the server plan. Local safety wins; the proxy can
   always refuse a server-proposed pathway. The local `kill_switch` also always wins.
3. **Off / fallback** — with no backed server policy, fall back to a tiny deterministic local
   set (`fallback_routes`) or off entirely. No local-minted canary matrix. This is the
   backed-or-off rule, unchanged.

### Safety & budget
- Local `kill_switch` and `blocklist` override the server unconditionally.
- The proxy enforces budget locally (it already tracks daily shadow spend); the server budget
  is an **upper bound** the proxy respects, never a license to overspend.
- Safety-stop thresholds may be *expressed* by the server but are *enforced* locally.

### Closing the loop
The proxy already emits experiment outcomes to `/v1/policy-events`; the server consumes those
to adjust the issued policy. The operational friction that motivated this note — hand-editing
`max_text_chars` and restarting to sample large Opus calls — becomes an evidence-driven server
decision pushed at runtime.

## Phased migration (maps to the issues)

- **Phase 0 (interim, proxy):** hot-reload the local experiment policy inside
  `routing_experiment_decision` (call `refresh_experiment_policy_if_changed()` in the hot
  path) so config edits stop requiring a restart. Bridges until server delivery lands.
- **Phase 1 (server):** define the `tokenclaw.routing_experiment_policy.v1` envelope + the
  signed/TTL'd issuing endpoint in `tokenclaw_server`, behind the opt-in managed boundary.
- **Phase 2 (proxy):** fetch + cache + hot-apply the server policy, reusing the client-contract
  fetch/cache machinery; consume it in `routing_experiment_decision`.
- **Phase 3 (proxy):** demote `routing_experiments.yaml` to guardrails + tiny fallback;
  implement the precedence (server → guardrail → off); keep local useful with the server off.

## Open questions

- Separate endpoint vs. extending the existing client-contract / policy-bundle surface.
- Standing TTL'd bundle (candidate matrix + controls) **plus** per-request policy-decision
  forcing (already present) — recommended split — vs. a single per-request decision.
- Where per-tenant budgets and safety-stop authority live (server-expressed, locally enforced).
