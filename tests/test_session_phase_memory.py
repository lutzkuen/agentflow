import io
import json
import tempfile
import unittest
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.session_phase_memory import build_session_phase_memory
from agentflow_proxy.store import Store, stable_json

FORBIDDEN_MEMORY_VALUES = (
    "SECRET_PHASE_PROMPT_BODY",
    "SECRET_PHASE_MESSAGE_BODY",
    "SECRET_PHASE_TOOL_PAYLOAD",
    "SECRET_PHASE_RESPONSE_BODY",
    "SECRET_PHASE_ERROR_BODY",
    "req_phase_memory_secret",
    "cache-key-phase-memory-secret",
    "/tmp/private-phase-memory.py",
    "secret-session",
)

FORBIDDEN_MEMORY_KEYS = (
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages"',
    '"prompt"',
    '"raw_request"',
    '"request_id"',
    '"session_id"',
    '"tool_payload"',
)


def _assert_session_memory_privacy_clean(testcase, payload):
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_MEMORY_VALUES:
        testcase.assertNotIn(forbidden, rendered)
    for forbidden_key in FORBIDDEN_MEMORY_KEYS:
        testcase.assertNotIn(forbidden_key, rendered)


def _log_call(store, suffix, *, session_id="secret-session-alpha", category="tool-result", routing=None, **overrides):
    routing_json = {
        "category": category,
        "text_chars": overrides.pop("text_chars", 40_000),
        "has_tools": category.startswith("tool"),
    }
    if routing:
        routing_json.update(routing)
    store.log_call(
        id=f"call-{suffix}",
        created_at=f"2026-06-10T10:00:{int(suffix):02d}+00:00",
        path="/v1/messages",
        requested_model=overrides.pop("requested_model", "claude-sonnet-4-6"),
        routed_model=overrides.pop("routed_model", "claude-sonnet-4-6"),
        stream=overrides.pop("stream", 1),
        cache_hit=0,
        status_code=overrides.pop("status_code", 200),
        latency_ms=overrides.pop("latency_ms", 1200),
        input_tokens_est=overrides.pop("input_tokens_est", 10_000),
        output_tokens_est=overrides.pop("output_tokens_est", 100),
        actual_input_tokens=overrides.pop("actual_input_tokens", 10_000),
        actual_output_tokens=overrides.pop("actual_output_tokens", 100),
        cache_creation_input_tokens=overrides.pop("cache_creation_input_tokens", 0),
        cache_read_input_tokens=overrides.pop("cache_read_input_tokens", 0),
        cost_est_usd=overrides.pop("cost_est_usd", 0.01),
        cost_baseline_usd=overrides.pop("cost_baseline_usd", 0.02),
        crunch_json=stable_json(overrides.pop("crunch_json", {"changed": True, "tokens_saved_est": 1200})),
        routing_json=stable_json(routing_json),
        cache_json=stable_json(overrides.pop("cache_json", {"status": "skipped", "reason": "streaming"})),
        error=overrides.pop("error", None),
        request_json=stable_json({"messages": [{"content": "SECRET_PROMPT_BODY /tmp/secret-file.py"}]}),
        response_json=stable_json({"content": [{"text": "SECRET_RESPONSE_BODY"}]}),
        session_id=session_id,
        category=category,
        retry_count=overrides.pop("retry_count", 0),
        thinking_output_tokens=overrides.pop("thinking_output_tokens", 0),
        provider=overrides.pop("provider", "anthropic"),
    )


def _adversarial_raw_fields():
    return {
        "prompt": "SECRET_PHASE_PROMPT_BODY",
        "messages": [{"role": "user", "content": "SECRET_PHASE_MESSAGE_BODY"}],
        "content": "SECRET_PHASE_MESSAGE_BODY",
        "tool_payload": {"arguments": "SECRET_PHASE_TOOL_PAYLOAD"},
        "request_id": "req_phase_memory_secret",
        "cache_key": "cache-key-phase-memory-secret",
        "file_path": "/tmp/private-phase-memory.py",
        "session_id": "secret-session-raw-field",
        "raw_request": {"messages": [{"content": "SECRET_PHASE_PROMPT_BODY"}]},
    }


class SessionPhaseMemoryTests(unittest.TestCase):
    def test_builds_metadata_only_rollups_with_phases_plateaus_and_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(store, "01", text_chars=40_000)
                _log_call(store, "02", text_chars=40_800)
                _log_call(
                    store,
                    "03",
                    text_chars=40_200,
                    routing={"fallback_reason": "rate_limited"},
                    retry_count=1,
                    status_code=429,
                    error="SECRET_ERROR_BODY",
                )
                _log_call(
                    store,
                    "04",
                    session_id="secret-session-beta",
                    category="code-gen",
                    routing={"workflow_phase": "verification"},
                    text_chars=7_000,
                    crunch_json={"changed": False, "tokens_saved_est": 0},
                    cache_json={"status": "miss", "reason": "exact-miss"},
                )
                _log_call(
                    store,
                    "05",
                    session_id=None,
                    category="short-completion",
                    routing={"workflow_phase": "summary"},
                    text_chars=900,
                    cache_json={"status": "hit", "reason": "exact-match"},
                )

                result = build_session_phase_memory(store, limit=50, window_size=10)
            finally:
                store.conn.close()

        self.assertEqual(result["schema"], "agentflow.session_phase_memory.v1")
        self.assertEqual(result["lookback"]["sampled_call_count"], 5)
        self.assertEqual(result["summary"]["session_count"], 3)
        self.assertEqual(result["summary"]["unknown_session_call_count"], 1)
        self.assertEqual(result["summary"]["plateau_session_count"], 1)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertFalse(result["privacy"]["raw_session_ids_included"])
        self.assertFalse(result["privacy"]["request_json_read"])

        by_kind = {row["session_key_kind"]: row for row in result["sessions"]}
        self.assertEqual(by_kind["missing_session"]["session_key"], "missing-session")
        hashed = [row for row in result["sessions"] if row["session_key_kind"] == "sha256_session_id"]
        self.assertEqual(len(hashed), 2)
        alpha = next(row for row in hashed if row["window"]["call_count"] == 3)
        self.assertEqual(alpha["dominant_phase"], "tool-execution")
        self.assertEqual(alpha["readiness"], "blocked")
        self.assertEqual(alpha["dominant_phase_count"], 3)
        self.assertEqual(alpha["phase_stability"], 1.0)
        self.assertEqual(alpha["model_family_floor"], "sonnet")
        self.assertIn("plateau", alpha["classifications"])
        self.assertTrue(alpha["context_plateau"]["active"])
        self.assertEqual(alpha["context_plateau"]["pairs"], 2)
        self.assertEqual(alpha["projected_savings_bucket"], "1k_10k_tokens")
        self.assertIn("recent_errors", alpha["blocker_reasons"])
        self.assertIn("recent_retries", alpha["blocker_reasons"])
        self.assertIn("recent_routing_fallback", alpha["blocker_reasons"])
        self.assertGreater(alpha["retry_rate"], 0)
        self.assertGreater(alpha["fallback_rate"], 0)
        self.assertIn({"value": "skipped", "count": 3}, alpha["cache_status_counts"])
        self.assertIn({"value": "rate_limited", "count": 1}, alpha["status_bucket_counts"])
        self.assertEqual(result["summary"]["memory_ready_session_count"], 0)
        self.assertEqual(result["summary"]["blocked_session_count"], 3)

        payload = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-session", payload)
        self.assertNotIn("SECRET_PROMPT_BODY", payload)
        self.assertNotIn("SECRET_RESPONSE_BODY", payload)
        self.assertNotIn("SECRET_ERROR_BODY", payload)
        self.assertNotIn("/tmp/secret-file.py", payload)
        self.assertNotIn("call-01", payload)

    def test_adversarial_metadata_rows_are_bucketed_before_memory_output(self):
        raw_fields = _adversarial_raw_fields()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for index in range(1, 4):
                    store.log_call(
                        id=f"adversarial-call-{index}",
                        created_at=f"2026-06-10T10:02:{index:02d}+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=1,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=10_000,
                        output_tokens_est=100,
                        actual_input_tokens=10_000,
                        actual_output_tokens=100,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        cost_est_usd=0.01,
                        cost_baseline_usd=0.02,
                        crunch_json=stable_json({"changed": True, "tokens_saved_est": 1200, **raw_fields}),
                        routing_json=stable_json(
                            {
                                "workflow_phase": "SECRET_PHASE_PROMPT_BODY",
                                "category": "SECRET_PHASE_MESSAGE_BODY",
                                "text_chars": 40_000 + index * 100,
                                "has_tools": True,
                                **raw_fields,
                            }
                        ),
                        cache_json=stable_json(
                            {
                                "status": "SECRET_PHASE_TOOL_PAYLOAD",
                                "reason": "cache-key-phase-memory-secret",
                                **raw_fields,
                            }
                        ),
                        error="SECRET_PHASE_ERROR_BODY",
                        request_json=stable_json({"messages": [{"content": "SECRET_PHASE_PROMPT_BODY"}], **raw_fields}),
                        response_json=stable_json({"content": [{"text": "SECRET_PHASE_RESPONSE_BODY"}]}),
                        session_id="secret-session-adversarial",
                        category="SECRET_PHASE_MESSAGE_BODY",
                        retry_count=0,
                        thinking_output_tokens=0,
                        provider="anthropic",
                        source_surface="SECRET_PHASE_PROMPT_BODY",
                    )

                result = build_session_phase_memory(store, limit=20, window_size=10)
            finally:
                store.conn.close()

        self.assertEqual(result["summary"]["session_count"], 1)
        session = result["sessions"][0]
        self.assertEqual(session["dominant_phase"], "unknown")
        self.assertEqual(session["source_surface"], "unknown")
        self.assertIn({"value": "unknown", "count": 3}, session["category_counts"])
        self.assertIn({"value": "unknown", "count": 3}, session["cache_status_counts"])
        self.assertIn({"value": "other", "count": 3}, session["cache_reason_counts"])
        self.assertEqual(managed_egress_violations(result), [])
        _assert_session_memory_privacy_clean(self, result)

    def test_cli_reads_seeded_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "01",
                    session_id="secret-session-cli",
                    category="short-completion",
                    routing={"workflow_phase": "summary"},
                    text_chars=1200,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.session_phase_memory_cli(["--db", db_path, "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.session_phase_memory.v1")
        self.assertEqual(result["summary"]["hashed_session_count"], 1)
        self.assertEqual(result["sessions"][0]["dominant_phase"], "summary")
        self.assertNotIn("secret-session-cli", stdout.getvalue())
        self.assertEqual(managed_egress_violations(result), [])


if __name__ == "__main__":
    unittest.main()
