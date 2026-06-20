import json
import tempfile
import unittest
from pathlib import Path

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.session_memory_hints import build_session_memory_optimization_hints
from tokenclaw.store import Store, stable_json


def _log_call(
    store,
    suffix,
    *,
    session_id="secret-session-alpha",
    category="summary",
    text_chars=40_000,
    status_code=200,
    retry_count=0,
    routing=None,
    cache_json=None,
    crunch_json=None,
    adversarial_raw_fields=False,
):
    routing_json = {
        "category": category,
        "workflow_phase": "summary" if category == "summary" else "tool-execution",
        "text_chars": text_chars,
        "has_tools": category.startswith("tool"),
    }
    if routing:
        routing_json.update(routing)
    raw_fields = {}
    if adversarial_raw_fields:
        raw_fields = {
            "prompt": "SECRET_HINT_MEMORY_PROMPT",
            "messages": [{"role": "user", "content": "SECRET_HINT_MEMORY_MESSAGE"}],
            "content": "SECRET_HINT_MEMORY_CONTENT",
            "tool_payload": {"arguments": "SECRET_HINT_MEMORY_TOOL_PAYLOAD"},
            "request_id": "req_hint_memory_secret",
            "cache_key": "cache-key-hint-memory-secret",
            "file_path": "/tmp/hint-memory-secret.py",
            "session_id": "secret-session-hint-raw-field",
            "raw_request": {"messages": [{"content": "SECRET_HINT_MEMORY_PROMPT"}]},
        }
        routing_json.update({
            "workflow_phase": "SECRET_HINT_MEMORY_PROMPT",
            "category": "SECRET_HINT_MEMORY_MESSAGE",
            **raw_fields,
        })
        cache_json = {"status": "SECRET_HINT_MEMORY_CONTENT", "reason": "cache-key-hint-memory-secret", **raw_fields}
        crunch_json = {"changed": True, "tokens_saved_est": 1500, **raw_fields}
    store.log_call(
        id=f"call-{suffix}",
        created_at=f"2026-06-10T10:01:{int(suffix):02d}+00:00",
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=0,
        cache_hit=0,
        status_code=status_code,
        latency_ms=1200,
        input_tokens_est=text_chars // 4,
        output_tokens_est=100,
        actual_input_tokens=text_chars // 4,
        actual_output_tokens=100,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        cost_est_usd=0.02,
        cost_baseline_usd=0.03,
        crunch_json=stable_json(crunch_json or {"changed": True, "tokens_saved_est": 1500}),
        routing_json=stable_json(routing_json),
        cache_json=stable_json(cache_json or {"status": "skipped", "reason": "streaming"}),
        error="SECRET_ERROR_BODY" if status_code >= 400 else None,
        request_json=stable_json({"messages": [{"content": "SECRET_HINT_MEMORY_PROMPT"}], **raw_fields})
        if adversarial_raw_fields
        else stable_json({"messages": [{"content": "SECRET_PROMPT_BODY /tmp/secret-file.py"}]}),
        response_json=stable_json({"content": [{"text": "SECRET_HINT_MEMORY_RESPONSE"}]})
        if adversarial_raw_fields
        else stable_json({"content": [{"text": "SECRET_RESPONSE_BODY"}]}),
        session_id=session_id,
        category="SECRET_HINT_MEMORY_MESSAGE" if adversarial_raw_fields else category,
        retry_count=retry_count,
        thinking_output_tokens=0,
        provider="anthropic",
    )


def _crunch_policy(**overrides):
    policy = {
        "session_memory_hints": {
            "enabled": True,
            "rule_id": "test-plateau-crunch",
            "crunch_profile": "test-plateau-profile",
            "old_context_summary_canary": True,
            "min_call_count": 4,
            "min_plateau_pairs": 3,
            "min_text_chars": 8000,
            "max_error_rate": 0.0,
            "allowed_phases": ["summary", "planning", "verification"],
            "block_tool_results": True,
            "block_thinking": True,
            "projected_savings_ratio": 0.20,
        }
    }
    policy["session_memory_hints"].update(overrides)
    return policy


def _cache_policy(**overrides):
    policy = {
        "session_memory_hints": {
            "enabled": True,
            "rule_id": "test-plateau-cache",
            "min_call_count": 4,
            "min_plateau_pairs": 3,
            "min_text_chars": 8000,
            "max_error_rate": 0.0,
            "allowed_phases": ["summary", "planning", "verification"],
            "block_tool_results": True,
            "block_thinking": True,
            "require_safe_invalidation": True,
            "require_reviewed_pattern_rule": True,
            "allow_tool_calls": False,
            "allow_streaming_replay": False,
        }
    }
    policy["session_memory_hints"].update(overrides)
    return policy


class SessionMemoryHintTests(unittest.TestCase):
    def test_plateau_session_produces_metadata_only_crunch_and_cache_dry_run_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for index, chars in enumerate((40_000, 40_500, 39_800, 40_100), start=1):
                    _log_call(store, f"0{index}", text_chars=chars)

                hints = build_session_memory_optimization_hints(
                    store_obj=store,
                    session_id="secret-session-alpha",
                    stream=False,
                    has_tool_blocks=False,
                    category="summary",
                    text_chars=41_000,
                    routing_meta={"workflow_phase": "summary"},
                    crunch_policy=_crunch_policy(),
                    crunch_policy_source="local-manual",
                    crunch_rule_path="/tmp/crunch_rules.yaml",
                    cache_policy=_cache_policy(),
                    cache_policy_source="local-manual",
                    cache_rule_path="/tmp/cache_rules.yaml",
                    safe_invalidation_evidence=True,
                    reviewed_cache_pattern_rule=True,
                )
            finally:
                store.conn.close()

        self.assertEqual(hints["schema"], "agentflow.session_memory_optimization_hints.v1")
        self.assertTrue(hints["privacy"]["metadata_only"])
        self.assertFalse(hints["privacy"]["request_json_read"])
        self.assertFalse(hints["privacy"]["cache_mutation"])
        self.assertEqual(managed_egress_violations(hints), [])

        crunch = hints["crunch"]
        self.assertEqual(crunch["status"], "eligible")
        self.assertEqual(crunch["rule_id"], "test-plateau-crunch")
        self.assertEqual(crunch["policy_source"], "local-manual")
        self.assertEqual(crunch["crunch_profile"], "test-plateau-profile")
        self.assertTrue(crunch["old_context_summary_canary_candidate"])
        self.assertGreater(crunch["projected_tokens_saved_est"], 0)
        self.assertFalse(crunch["mutation_applied"])

        cache = hints["cache"]
        self.assertEqual(cache["status"], "eligible")
        self.assertEqual(cache["rule_id"], "test-plateau-cache")
        self.assertEqual(cache["cacheability_hint"], "dry-run-exact-replay-group")
        self.assertEqual(cache["replayability_level"], "local-exact-response-dry-run")
        self.assertFalse(cache["cache_mutation"])
        self.assertTrue(cache["dry_run_projection"]["exact_replay_grouping_candidate"])
        proposal = cache["dry_run_replay_proposal"]
        self.assertEqual(proposal["schema"], "agentflow.session_memory_cache_replay_proposal.v1")
        self.assertEqual(proposal["status"], "session-plateau-dry-run-eligible")
        self.assertEqual(proposal["rule_id"], "test-plateau-cache")
        self.assertTrue(proposal["proposal_fingerprint"].startswith("sha256:"))
        self.assertIn("confirm exact replay", " ".join(proposal["review_steps"]))
        self.assertNotEqual(proposal["projected_savings_bucket"], "none")
        self.assertFalse(proposal["mutation_applied"])
        self.assertFalse(proposal["cache_mutation"])
        self.assertFalse(proposal["policy_files_written"])
        self.assertFalse(proposal["policy_rule_path_included"])

        rendered = json.dumps(hints, sort_keys=True)
        self.assertIn("sha256:", rendered)
        self.assertNotIn("secret-session-alpha", rendered)
        self.assertNotIn("SECRET_PROMPT_BODY", rendered)
        self.assertNotIn("SECRET_RESPONSE_BODY", rendered)
        self.assertNotIn("SECRET_ERROR_BODY", rendered)
        self.assertNotIn("/tmp/secret-file.py", rendered)

    def test_missing_memory_blocks_hints_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                hints = build_session_memory_optimization_hints(
                    store_obj=store,
                    session_id="secret-session-missing",
                    stream=False,
                    has_tool_blocks=False,
                    category="summary",
                    text_chars=40_000,
                    routing_meta={"workflow_phase": "summary"},
                    crunch_policy=_crunch_policy(),
                    crunch_policy_source="local-manual",
                    crunch_rule_path="/tmp/crunch_rules.yaml",
                    cache_policy=_cache_policy(),
                    cache_policy_source="local-manual",
                    cache_rule_path="/tmp/cache_rules.yaml",
                    safe_invalidation_evidence=True,
                    reviewed_cache_pattern_rule=True,
                )
            finally:
                store.conn.close()

        self.assertFalse(hints["crunch"]["memory"]["available"])
        self.assertEqual(hints["crunch"]["status"], "blocked")
        self.assertIn("no_session_memory", hints["crunch"]["blockers"])
        self.assertFalse(hints["crunch"]["mutation_applied"])
        self.assertEqual(hints["cache"]["status"], "blocked")
        self.assertIn("no_session_memory", hints["cache"]["blockers"])
        self.assertFalse(hints["cache"]["cache_mutation"])
        self.assertEqual(hints["cache"]["dry_run_replay_proposal"]["status"], "blocked")
        self.assertIn("no_session_memory", hints["cache"]["dry_run_replay_proposal"]["blockers"])
        self.assertNotIn("secret-session-missing", json.dumps(hints, sort_keys=True))
        self.assertEqual(managed_egress_violations(hints), [])

    def test_adversarial_memory_rows_are_sanitized_in_hint_payloads(self):
        forbidden = (
            "SECRET_HINT_MEMORY_PROMPT",
            "SECRET_HINT_MEMORY_MESSAGE",
            "SECRET_HINT_MEMORY_CONTENT",
            "SECRET_HINT_MEMORY_TOOL_PAYLOAD",
            "SECRET_HINT_MEMORY_RESPONSE",
            "req_hint_memory_secret",
            "cache-key-hint-memory-secret",
            "/tmp/hint-memory-secret.py",
            "secret-session-alpha",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for index, chars in enumerate((40_000, 40_500, 39_800, 40_100), start=1):
                    _log_call(store, f"0{index}", text_chars=chars, adversarial_raw_fields=True)
                hints = build_session_memory_optimization_hints(
                    store_obj=store,
                    session_id="secret-session-alpha",
                    stream=False,
                    has_tool_blocks=False,
                    category="summary",
                    text_chars=41_000,
                    routing_meta={"workflow_phase": "summary"},
                    crunch_policy=_crunch_policy(allowed_phases=["unknown"]),
                    crunch_policy_source="local-manual",
                    crunch_rule_path="/tmp/crunch_rules.yaml",
                    cache_policy=_cache_policy(allowed_phases=["unknown"]),
                    cache_policy_source="local-manual",
                    cache_rule_path="/tmp/cache_rules.yaml",
                    safe_invalidation_evidence=True,
                    reviewed_cache_pattern_rule=True,
                )
            finally:
                store.conn.close()

        self.assertEqual(hints["crunch"]["status"], "eligible")
        self.assertEqual(hints["crunch"]["memory"]["dominant_phase"], "unknown")
        self.assertIn({"value": "unknown", "count": 4}, hints["crunch"]["memory"]["cache_status_counts"])
        self.assertEqual(managed_egress_violations(hints), [])
        rendered = json.dumps(hints, sort_keys=True)
        for value in forbidden:
            self.assertNotIn(value, rendered)

    def test_policy_disabled_records_skip_without_enabling_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for index in range(1, 5):
                    _log_call(store, f"0{index}")
                hints = build_session_memory_optimization_hints(
                    store_obj=store,
                    session_id="secret-session-alpha",
                    stream=False,
                    has_tool_blocks=False,
                    category="summary",
                    text_chars=40_000,
                    routing_meta={"workflow_phase": "summary"},
                    crunch_policy=_crunch_policy(enabled=False),
                    crunch_policy_source="local-default",
                    crunch_rule_path="bundled",
                    cache_policy=_cache_policy(enabled=False),
                    cache_policy_source="local-default",
                    cache_rule_path="bundled",
                    safe_invalidation_evidence=True,
                    reviewed_cache_pattern_rule=True,
                )
            finally:
                store.conn.close()

        self.assertEqual(hints["crunch"]["status"], "skipped")
        self.assertEqual(hints["crunch"]["reason"], "policy-disabled")
        self.assertFalse(hints["crunch"]["mutation_applied"])
        self.assertEqual(hints["cache"]["status"], "skipped")
        self.assertEqual(hints["cache"]["reason"], "policy-disabled")
        self.assertFalse(hints["cache"]["cache_mutation"])

    def test_unsafe_plateau_cases_are_blocked_with_explicit_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for index, status in enumerate((200, 200, 429, 200), start=1):
                    _log_call(
                        store,
                        f"0{index}",
                        category="tool-result",
                        text_chars=50_000 + index * 100,
                        status_code=status,
                        retry_count=1 if status == 429 else 0,
                        routing={"reason": "keep requested model for thinking request"},
                    )
                hints = build_session_memory_optimization_hints(
                    store_obj=store,
                    session_id="secret-session-alpha",
                    stream=True,
                    has_tool_blocks=True,
                    category="tool-result",
                    text_chars=50_000,
                    routing_meta={"reason": "keep requested model for thinking request"},
                    crunch_policy=_crunch_policy(),
                    crunch_policy_source="local-manual",
                    crunch_rule_path="/tmp/crunch_rules.yaml",
                    cache_policy=_cache_policy(),
                    cache_policy_source="local-manual",
                    cache_rule_path="/tmp/cache_rules.yaml",
                    safe_invalidation_evidence=False,
                    reviewed_cache_pattern_rule=False,
                    current_thinking=True,
                )
            finally:
                store.conn.close()

        crunch_blockers = set(hints["crunch"]["blockers"])
        self.assertEqual(hints["crunch"]["status"], "blocked")
        self.assertIn("tool_result_state_dependence", crunch_blockers)
        self.assertIn("thinking_blocks_present", crunch_blockers)
        self.assertIn("recent_errors", crunch_blockers)
        self.assertIn("recent_retries", crunch_blockers)

        cache_blockers = set(hints["cache"]["blockers"])
        self.assertEqual(hints["cache"]["status"], "blocked")
        self.assertIn("tool_result_state_dependence", cache_blockers)
        self.assertIn("tool_call_cache_disabled", cache_blockers)
        self.assertIn("streaming_replay_reviewed_rule_required", cache_blockers)
        self.assertIn("missing_invalidation_evidence", cache_blockers)
        self.assertIn("reviewed_pattern_rule_required", cache_blockers)
        self.assertFalse(hints["cache"]["dry_run_projection"]["eligible"])
        proposal = hints["cache"]["dry_run_replay_proposal"]
        self.assertEqual(proposal["status"], "blocked")
        self.assertIn("streaming_replay_reviewed_rule_required", proposal["blockers"])
        self.assertIn("tool_call_cache_disabled", proposal["blockers"])
        self.assertIn("missing_invalidation_evidence", proposal["blockers"])
        self.assertTrue(proposal["blocker_families"]["streaming"])
        self.assertTrue(proposal["blocker_families"]["tool"])
        self.assertTrue(proposal["blocker_families"]["thinking"])
        self.assertTrue(proposal["blocker_families"]["safe_invalidation"])
        self.assertTrue(proposal["blocker_families"]["reviewed_pattern_rule"])
        self.assertFalse(proposal["cache_mutation"])


if __name__ == "__main__":
    unittest.main()
