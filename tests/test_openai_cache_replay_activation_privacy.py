"""Privacy and failure fixtures for dependency-aware cache replay activation.

Covers every outcome class: hit, holdout, bypass, invalidated, safety-stop,
rollback recommendation, provider-adoption-gated hold.  Each fixture asserts
that no raw content, file paths, cache keys, request IDs, session IDs, tool IDs,
or policy file contents appear in activation-health rollups, lifecycle-feedback
payloads, or dashboard API responses.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from tokenclaw.cache import build_cache_replay_lifecycle_feedback
from tokenclaw.dashboard_app import create_dashboard_app
from tokenclaw.managed_egress import RAW_FEATURE_KEYS, assert_managed_egress_safe
from tokenclaw.stats import stats_cache_replay_activation_health
from tokenclaw.store import SQLiteStore, stable_json, utc_now


# ---------------------------------------------------------------------------
# Forbidden values and keys – nothing raw should escape the activation path.
# ---------------------------------------------------------------------------

FORBIDDEN_VALUES = (
    "raw-activation-prompt-secret",
    "raw-activation-message-secret",
    "raw-activation-content-secret",
    "raw-activation-tool-payload-secret",
    "req_activation_raw_secret",
    "cache-key-activation-secret",
    "/home/lutz/private/activation_secret.py",
    "sk-activation-secret",
    "tenant-activation-secret",
    "raw-activation-session-id",
    "raw-activation-tool-id-secret",
    "raw-activation-sse-data-secret",
)

FORBIDDEN_KEYS = (
    '"api_key"',
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages"',
    '"prompt"',
    '"raw_request"',
    '"request_id"',
    '"session_id"',
    '"tenant_id"',
    '"tool_payload"',
)


def _assert_activation_privacy_clean(tc: unittest.TestCase, payload: object) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_VALUES:
        tc.assertNotIn(forbidden, rendered, f"raw value leaked: {forbidden!r}")
    for forbidden_key in FORBIDDEN_KEYS:
        tc.assertNotIn(forbidden_key, rendered, f"raw key leaked: {forbidden_key!r}")


# ---------------------------------------------------------------------------
# Helpers for building pattern-rule and canary metadata blobs that the
# stats function reads from cache_json in the calls table.
# ---------------------------------------------------------------------------

_RULE_ID = "managed-activation-cache-replay-rule"
_CANDIDATE_ID = "activation-cache-replay-candidate"


def _canary_meta(*, cohort: str = "canary_applied", selected: bool = True) -> dict:
    return {
        "schema": "tokenclaw.pattern_canary_decision.v1",
        "enabled": True,
        "selected": selected,
        "cohort": cohort,
        "fraction": 0.5,
        "holdout_fraction": 0.5,
        "unit": "session",
        "status": "applied" if cohort == "canary_applied" else "holdout",
    }


def _pattern_rule(
    *,
    cohort: str = "canary_applied",
    selected: bool = True,
    reason: str | None = None,
    safety_stop: dict | None = None,
) -> dict:
    rule: dict = {
        "rule_id": _RULE_ID,
        "candidate_id": _CANDIDATE_ID,
        "policy_source": "managed-recommended",
        "scope": "session",
        "canary": _canary_meta(cohort=cohort, selected=selected),
    }
    if reason:
        rule["reason"] = reason
    if safety_stop:
        rule["safety_stop"] = safety_stop
    return rule


def _raw_like_extra() -> dict:
    """Fields that simulate raw-like pollution injected into cache metadata."""
    return {
        "prompt": "raw-activation-prompt-secret",
        "messages": [{"role": "user", "content": "raw-activation-message-secret"}],
        "content": "raw-activation-content-secret",
        "tool_payload": "raw-activation-tool-payload-secret",
        "request_id": "req_activation_raw_secret",
        "cache_key": "cache-key-activation-secret",
        "file_path": "/home/lutz/private/activation_secret.py",
        "session_id": "raw-activation-session-id",
        "tenant_id": "tenant-activation-secret",
        "api_key": "sk-activation-secret",
    }


class DependencyAwareCacheReplayActivationPrivacyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Store helper – logs a single call with the given cache_json extras.
    # ------------------------------------------------------------------

    def _log_call(
        self,
        *,
        cache_status: str = "miss",
        cache_reason: str = "exact-miss",
        cache_hit: int = 0,
        status_code: int = 200,
        stream: int = 0,
        retry_count: int = 0,
        cost: float = 0.01,
        cost_baseline: float = 0.03,
        cache_extra: dict | None = None,
        session_id: str = "raw-activation-session-id",
    ) -> str:
        call_id = str(uuid.uuid4())
        cache_json: dict = {
            "status": cache_status,
            "reason": cache_reason,
            "policy_source": "managed-recommended",
            "replayability_level": "local-exact-response",
        }
        if cache_extra:
            cache_json.update(cache_extra)
        self.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            stream=stream,
            cache_hit=cache_hit,
            status_code=status_code,
            latency_ms=120,
            input_tokens_est=300,
            output_tokens_est=40,
            actual_input_tokens=300,
            actual_output_tokens=40,
            cost_est_usd=cost,
            cost_baseline_usd=cost_baseline,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({
                "enabled": False,
                "provider": "openai",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "text_chars": 2400,
                "has_tools": True,
                "category": "tool-result",
            }),
            cache_json=stable_json(cache_json),
            error=None,
            request_json=stable_json({
                "input": "raw-activation-prompt-secret",
                "cache_key": "cache-key-activation-secret",
                "session_id": "raw-activation-session-id",
            }),
            response_json=stable_json({"output_text": "raw-activation-content-secret"}),
            session_id=session_id,
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )
        return call_id

    # ------------------------------------------------------------------
    # Convenience: run activation health and assert basic invariants.
    # ------------------------------------------------------------------

    def _activation_health(self) -> dict:
        result = asyncio.run(stats_cache_replay_activation_health(self.store, limit=50, scan_limit=500))
        self.assertEqual(result["schema"], "tokenclaw.cache_replay_activation_health.v1")
        self.assertTrue(result["read_only"])
        privacy = result["privacy"]
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["raw_messages_included"])
        self.assertFalse(privacy["raw_request_bodies_included"])
        self.assertFalse(privacy["raw_responses_included"])
        self.assertFalse(privacy["raw_tool_payloads_included"])
        self.assertFalse(privacy["file_paths_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["raw_session_ids_included"])
        self.assertFalse(privacy["cache_keys_included"])
        self.assertFalse(privacy["pattern_hashes_included"])
        self.assertFalse(privacy["policy_file_contents_included"])
        return result

    # ==================================================================
    # 1. Stable dependency replay hit → widen-candidate or canary-active
    # ==================================================================

    def test_stable_dependency_replay_hit_is_metadata_only(self) -> None:
        rule = _pattern_rule()
        replay_canary = {
            "schema": "tokenclaw.cache_replay_canary_decision.v1",
            "rule_id": _RULE_ID,
            "candidate_id": _CANDIDATE_ID,
            "policy_source": "managed-recommended",
            "status": "applied",
            "reason": "dependency-stable",
            "canary": _canary_meta(),
            **_raw_like_extra(),
        }
        # Applied / hit rows
        for _ in range(3):
            self._log_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=0.03,
                cache_extra={
                    "pattern_rule": rule,
                    "pattern_rules": {"configured_count": 1, "matched_count": 1, "rules": [rule]},
                    "cache_replay_canary": replay_canary,
                    "estimated_saved_cost_usd": 0.03,
                    **_raw_like_extra(),
                },
            )
        # Holdout row → needed to reach "widen candidate"
        holdout_rule = _pattern_rule(cohort="canary_holdout", selected=False, reason="canary_holdout")
        self._log_call(
            cache_status="skipped",
            cache_reason="canary_holdout",
            cache_extra={
                "pattern_rule": holdout_rule,
                "pattern_rules": {
                    "configured_count": 1,
                    "matched_count": 0,
                    "skip_reasons": [{**holdout_rule}],
                },
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": _RULE_ID,
                    "candidate_id": _CANDIDATE_ID,
                    "status": "holdout",
                    "reason": "canary_holdout",
                    "canary": _canary_meta(cohort="canary_holdout", selected=False),
                    **_raw_like_extra(),
                },
                **_raw_like_extra(),
            },
        )

        result = self._activation_health()
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        states = {c["state"] for c in cohorts}
        self.assertTrue(
            states & {"widen candidate", "canary active"},
            f"expected widen/canary-active state, got {states}",
        )
        hit_cohort = next(c for c in cohorts if c["candidate_id"] == _CANDIDATE_ID)
        self.assertGreater(hit_cohort["hit_count"], 0)
        self.assertFalse(hit_cohort["cache_keys_included"])
        self.assertFalse(hit_cohort["pattern_hashes_included"])
        _assert_activation_privacy_clean(self, result)

        # Verify lifecycle feedback event is metadata-only
        event = build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "hit",
                "reason": "exact-match",
                "pattern_rule": rule,
                "cache_replay_canary": replay_canary,
                "estimated_saved_cost_usd": 0.03,
                **_raw_like_extra(),
            },
            provider="openai",
            source_surface="openai_responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            latency_ms=8,
            retry_count=0,
            cost_est_usd=0.0,
            cost_baseline_usd=0.03,
            category="tool-result",
            stream=False,
        )
        self.assertIsNotNone(event)
        assert event is not None
        assert_managed_egress_safe(event)
        self.assertEqual(event["cohort"], "replayed")
        self.assertFalse(event["privacy"]["file_paths_included"])
        self.assertFalse(event["privacy"]["cache_keys_included"])
        self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(set(event.keys())))
        _assert_activation_privacy_clean(self, event)

    # ==================================================================
    # 2. Stale dependency → invalidated, activation state = "hold"
    # ==================================================================

    def test_stale_dependency_bypass_activation_state_is_hold(self) -> None:
        rule = _pattern_rule(cohort="canary_applied", reason="dependency-changed")
        dep_audit = {
            "schema": "tokenclaw.cache_file_dependency_audit.v1",
            "file_watch_enabled": True,
            "snapshot_root_policy": "stored-local-paths",
            "root_path_included": False,
            "snapshot_count": 1,
            "changed_path_count": 1,
            "deleted_path_count": 0,
            "missing_path_count": 0,
            "cap_exceeded": False,
            "invalidation_reason": "dependency-changed",
            "safe_invalidation_evidence": False,
            "file_dependency_evidence_available": False,
            "paths_included": False,
        }
        self._log_call(
            cache_status="invalidated",
            cache_reason="dependency-changed",
            cache_extra={
                "pattern_rule": rule,
                "file_dependency_audit": dep_audit,
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": _RULE_ID,
                    "candidate_id": _CANDIDATE_ID,
                    "status": "invalidated",
                    "reason": "dependency-changed",
                    "canary": _canary_meta(),
                    "dependency_audit": dep_audit,
                    **_raw_like_extra(),
                },
                **_raw_like_extra(),
            },
        )

        result = self._activation_health()
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        cohort = next(c for c in cohorts if c["candidate_id"] == _CANDIDATE_ID)
        self.assertEqual(cohort["state"], "hold")
        self.assertGreater(cohort["invalidation_count"], 0)
        _assert_activation_privacy_clean(self, result)

        # Lifecycle feedback for invalidated event
        event = build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "invalidated",
                "reason": "dependency-changed",
                "invalidated": True,
                "invalidation_reason": "dependency-changed",
                "pattern_rule": rule,
                "file_dependency_audit": dep_audit,
                "cache_replay_canary": {
                    "status": "invalidated",
                    "reason": "dependency-changed",
                    "dependency_audit": dep_audit,
                },
                **_raw_like_extra(),
            },
            provider="openai",
            source_surface="openai_responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            latency_ms=10,
            retry_count=0,
            cost_est_usd=0.03,
            cost_baseline_usd=0.03,
            category="tool-result",
            stream=False,
        )
        self.assertIsNotNone(event)
        assert event is not None
        assert_managed_egress_safe(event)
        self.assertEqual(event["cohort"], "invalidated")
        self.assertIn("dependency-changed", event["invalidation_reason_codes"])
        _assert_activation_privacy_clean(self, event)

    # ==================================================================
    # 3. Missing dependency evidence → bypass, activation state = "hold"
    # ==================================================================

    def test_missing_dependency_evidence_fails_closed(self) -> None:
        rule = _pattern_rule(cohort="canary_applied", reason="file-dependency-missing")
        missing_audit = {
            "schema": "tokenclaw.cache_file_dependency_audit.v1",
            "file_watch_enabled": True,
            "snapshot_root_policy": "stored-local-paths",
            "root_path_included": False,
            "snapshot_count": 0,
            "changed_path_count": 0,
            "deleted_path_count": 0,
            "missing_path_count": 0,
            "cap_exceeded": False,
            "invalidation_reason": None,
            "safe_invalidation_evidence": False,
            "file_dependency_evidence_available": False,
            "dependency_capture_reason": "file-dependency-missing",
            "paths_included": False,
        }
        self._log_call(
            cache_status="bypassed",
            cache_reason="file-dependency-missing",
            cache_extra={
                "pattern_rule": rule,
                "file_dependency_audit": missing_audit,
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": _RULE_ID,
                    "candidate_id": _CANDIDATE_ID,
                    "status": "bypassed",
                    "reason": "file-dependency-missing",
                    "canary": _canary_meta(),
                    "current_dependency_evidence": {
                        "safe_invalidation_evidence": False,
                        "reason": "file-dependency-missing",
                        "paths_included": False,
                    },
                    **_raw_like_extra(),
                },
                **_raw_like_extra(),
            },
        )

        result = self._activation_health()
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        cohort = next(c for c in cohorts if c["candidate_id"] == _CANDIDATE_ID)
        self.assertEqual(cohort["state"], "hold")
        _assert_activation_privacy_clean(self, result)
        # Ensure no dependency paths leaked
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-activation-session-id", rendered)

    # ==================================================================
    # 4. Malformed cached SSE frames → bypass, no raw SSE data in output
    # ==================================================================

    def test_malformed_sse_frame_bypasses_without_raw_payload(self) -> None:
        rule = _pattern_rule(cohort="canary_applied", reason="malformed-stream-cache")
        self._log_call(
            cache_status="bypassed",
            cache_reason="malformed-stream-cache",
            stream=1,
            cache_extra={
                "pattern_rule": rule,
                "malformed_stream_cache": {
                    "reason": "sse-data-missing",
                    "raw_payload_included": False,
                    "sse_frame_count": 3,
                    "sse": {"raw_activation_sse_data_secret": "raw-activation-sse-data-secret"},
                },
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": _RULE_ID,
                    "candidate_id": _CANDIDATE_ID,
                    "status": "bypassed",
                    "reason": "malformed-stream-cache",
                    "canary": _canary_meta(),
                    **_raw_like_extra(),
                },
                **_raw_like_extra(),
            },
        )

        result = self._activation_health()
        _assert_activation_privacy_clean(self, result)
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-activation-sse-data-secret", rendered)
        self.assertNotIn("sse-data-missing", rendered)

        # Lifecycle feedback for streaming bypass
        event = build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "bypassed",
                "reason": "malformed-stream-cache",
                "pattern_rule": rule,
                "cache_replay_canary": {
                    "status": "bypassed",
                    "reason": "malformed-stream-cache",
                    "canary": _canary_meta(),
                },
                "malformed_stream_cache": {
                    "reason": "sse-data-missing",
                    "raw_payload_included": False,
                },
                **_raw_like_extra(),
            },
            provider="openai",
            source_surface="openai_responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            latency_ms=5,
            retry_count=0,
            cost_est_usd=0.03,
            cost_baseline_usd=0.03,
            category="tool-result",
            stream=True,
        )
        self.assertIsNotNone(event)
        assert event is not None
        assert_managed_egress_safe(event)
        _assert_activation_privacy_clean(self, event)
        rendered_event = json.dumps(event, sort_keys=True)
        self.assertNotIn("raw-activation-sse-data-secret", rendered_event)

    # ==================================================================
    # 5. Tool protocol mismatch → unsafe-tool-cache-pattern skip
    # ==================================================================

    def test_tool_protocol_mismatch_fails_closed_without_raw_tool_ids(self) -> None:
        skip_rule = {
            "rule_id": _RULE_ID,
            "candidate_id": _CANDIDATE_ID,
            "policy_source": "managed-recommended",
            "reason": "unsafe-tool-cache-pattern",
            "canary": _canary_meta(),
        }
        self._log_call(
            cache_status="skipped",
            cache_reason="unsafe-tool-cache-pattern",
            cache_extra={
                "pattern_rules": {
                    "configured_count": 1,
                    "matched_count": 0,
                    "skip_reasons": [
                        {
                            **skip_rule,
                            "tool_id": "raw-activation-tool-id-secret",
                            "tool_payload": "raw-activation-tool-payload-secret",
                        }
                    ],
                },
                **_raw_like_extra(),
            },
        )

        result = self._activation_health()
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-activation-tool-id-secret", rendered)
        self.assertNotIn("raw-activation-tool-payload-secret", rendered)
        _assert_activation_privacy_clean(self, result)

    # ==================================================================
    # 6. Canary holdout → activation state "hold" (holdout-only)
    # ==================================================================

    def test_canary_holdout_activation_state_and_no_raw_session_leak(self) -> None:
        holdout_rule = _pattern_rule(cohort="canary_holdout", selected=False, reason="canary_holdout")
        for _ in range(2):
            self._log_call(
                cache_status="skipped",
                cache_reason="canary_holdout",
                cache_extra={
                    "pattern_rule": holdout_rule,
                    "pattern_rules": {
                        "configured_count": 1,
                        "matched_count": 0,
                        "skip_reasons": [{**holdout_rule}],
                    },
                    "cache_replay_canary": {
                        "schema": "tokenclaw.cache_replay_canary_decision.v1",
                        "rule_id": _RULE_ID,
                        "candidate_id": _CANDIDATE_ID,
                        "status": "holdout",
                        "reason": "canary_holdout",
                        "canary": _canary_meta(cohort="canary_holdout", selected=False),
                        **_raw_like_extra(),
                    },
                    **_raw_like_extra(),
                },
                session_id="raw-activation-session-id",
            )

        result = self._activation_health()
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        cohort = next(c for c in cohorts if c["candidate_id"] == _CANDIDATE_ID)
        self.assertIn(cohort["state"], {"hold", "needs evidence"})
        self.assertGreater(cohort["holdout_count"], 0)
        _assert_activation_privacy_clean(self, result)

        # Lifecycle feedback for holdout
        event = build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "skipped",
                "reason": "canary_holdout",
                "pattern_rule": holdout_rule,
                "pattern_rules": {
                    "skip_reasons": [{**holdout_rule}],
                },
                **_raw_like_extra(),
            },
            provider="openai",
            source_surface="openai_responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=200,
            latency_ms=130,
            retry_count=0,
            cost_est_usd=0.03,
            cost_baseline_usd=0.03,
            category="tool-result",
            stream=False,
        )
        self.assertIsNotNone(event)
        assert event is not None
        assert_managed_egress_safe(event)
        self.assertEqual(event["cohort"], "holdout")
        _assert_activation_privacy_clean(self, event)

    # ==================================================================
    # 7. Safety stop → activation state "rollback recommended"
    # ==================================================================

    def test_safety_stop_recommends_rollback_and_is_metadata_only(self) -> None:
        safety_stop_meta = {
            "reason": "error-rate-regression",
            "decision": "stop",
            "sample_count": 8,
            "error_rate": 0.4,
            "retry_rate": 0.5,
            "pattern_hash": "sha256:" + "f" * 64,
            "path": "/home/lutz/private/activation_secret.py",
        }
        stop_rule = _pattern_rule(reason="local-canary-safety-stop", safety_stop=safety_stop_meta)
        self._log_call(
            cache_status="bypassed",
            cache_reason="local-canary-safety-stop",
            cache_extra={
                "pattern_rules": {
                    "configured_count": 1,
                    "matched_count": 0,
                    "skip_reasons": [{**stop_rule}],
                },
                "safety_stop": safety_stop_meta,
                **_raw_like_extra(),
            },
        )

        result = self._activation_health()
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        cohort = next(c for c in cohorts if c["candidate_id"] == _CANDIDATE_ID)
        self.assertEqual(cohort["state"], "rollback recommended")
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("sha256:" + "f" * 64, rendered)
        self.assertNotIn("/home/lutz/private/activation_secret.py", rendered)
        _assert_activation_privacy_clean(self, result)

        # Lifecycle feedback for safety-stopped event
        event = build_cache_replay_lifecycle_feedback(
            cache_meta={
                "status": "skipped",
                "reason": "local-canary-safety-stop",
                "policy_source": "managed-recommended",
                "pattern_rules": {"skip_reasons": [stop_rule]},
                **_raw_like_extra(),
            },
            provider="openai",
            source_surface="openai_responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            status_code=500,
            latency_ms=25,
            retry_count=3,
            cost_est_usd=0.03,
            cost_baseline_usd=0.03,
            category="tool-result",
            stream=False,
        )
        self.assertIsNotNone(event)
        assert event is not None
        assert_managed_egress_safe(event)
        self.assertEqual(event["cohort"], "safety_stopped")
        self.assertEqual(event["safety_stop"]["decision"], "stop")
        rendered_event = json.dumps(event, sort_keys=True)
        self.assertNotIn("sha256:" + "f" * 64, rendered_event)
        self.assertNotIn("/home/lutz/private/activation_secret.py", rendered_event)
        _assert_activation_privacy_clean(self, event)

    # ==================================================================
    # 8. Provider adoption regression → activation state "hold"
    # ==================================================================

    def test_provider_adoption_regression_gates_activation_without_raw_tool_ids(self) -> None:
        rule = _pattern_rule()
        adoption_gate = {
            "status": "blocked",
            "blocking": True,
            "reason_codes": ["provider-adoption-regression"],
            "tool_id": "raw-activation-tool-id-secret",
            "session_id": "raw-activation-session-id",
        }
        for _ in range(2):
            self._log_call(
                cache_status="hit",
                cache_reason="exact-match",
                cache_hit=1,
                cost=0.0,
                cost_baseline=0.03,
                cache_extra={
                    "pattern_rule": rule,
                    "cache_replay_canary": {
                        "schema": "tokenclaw.cache_replay_canary_decision.v1",
                        "rule_id": _RULE_ID,
                        "candidate_id": _CANDIDATE_ID,
                        "status": "applied",
                        "reason": "dependency-stable",
                        "canary": _canary_meta(),
                    },
                    "provider_adoption_gate": adoption_gate,
                    **_raw_like_extra(),
                },
            )

        result = self._activation_health()
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        cohort = next(c for c in cohorts if c["candidate_id"] == _CANDIDATE_ID)
        self.assertEqual(cohort["state"], "hold")
        self.assertEqual(cohort["provider_adoption_gate"]["status"], "blocked")
        self.assertIn("provider-adoption-regression", cohort["reason_codes"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw-activation-tool-id-secret", rendered)
        self.assertNotIn("raw-activation-session-id", rendered)
        _assert_activation_privacy_clean(self, result)

    # ==================================================================
    # 9. Aggregate-only managed lifecycle feedback (no raw content)
    # ==================================================================

    def test_aggregate_managed_lifecycle_feedback_is_metadata_only(self) -> None:
        rule = _pattern_rule()
        for cohort, reason, status in [
            ("canary_applied", "dependency-stable", "applied"),
            ("canary_applied", "dependency-stable", "applied"),
            ("canary_holdout", "canary_holdout", "holdout"),
        ]:
            pattern_rule = _pattern_rule(cohort=cohort, selected=(cohort == "canary_applied"), reason=reason)
            event = build_cache_replay_lifecycle_feedback(
                cache_meta={
                    "status": "hit" if status == "applied" else "skipped",
                    "reason": reason,
                    "pattern_rule": pattern_rule,
                    "cache_replay_canary": {
                        "status": status,
                        "reason": reason,
                        "canary": _canary_meta(cohort=cohort, selected=(cohort == "canary_applied")),
                    },
                    **_raw_like_extra(),
                },
                provider="openai",
                source_surface="openai_responses",
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                status_code=200,
                latency_ms=15,
                retry_count=0,
                cost_est_usd=0.0 if status == "applied" else 0.03,
                cost_baseline_usd=0.03,
                category="tool-result",
                stream=False,
            )
            if event is None:
                continue
            assert_managed_egress_safe(event)
            self.assertFalse(event["privacy"]["cache_keys_included"])
            self.assertFalse(event["privacy"]["file_paths_included"])
            self.assertTrue(RAW_FEATURE_KEYS.isdisjoint(set(event.keys())))
            _assert_activation_privacy_clean(self, event)

    # ==================================================================
    # 10. Raw-like payload fields are rejected / sanitized in rollup
    # ==================================================================

    def test_raw_like_candidate_id_is_sanitized_in_activation_health(self) -> None:
        raw_candidate = "raw candidate / cache_key session_id /home/lutz/private"
        rule = {
            "rule_id": "raw rule / cache_key request_id",
            "candidate_id": raw_candidate,
            "policy_source": "managed-recommended",
            "scope": "session",
            "canary": _canary_meta(),
        }
        self._log_call(
            cache_status="hit",
            cache_reason="exact-match",
            cache_hit=1,
            cost=0.0,
            cost_baseline=0.03,
            cache_extra={
                "pattern_rule": rule,
                "cache_replay_canary": {
                    "rule_id": "raw rule / cache_key request_id",
                    "candidate_id": raw_candidate,
                    "status": "applied",
                    "reason": "dependency-stable",
                    "canary": _canary_meta(),
                    **_raw_like_extra(),
                },
            },
        )

        result = self._activation_health()
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(raw_candidate, rendered)
        self.assertNotIn("raw rule / cache_key request_id", rendered)
        cohorts = result["cohorts"]
        self.assertGreater(len(cohorts), 0)
        sanitized_cohort = cohorts[0]
        # The candidate_id must be redacted (starts with "candidate-id:" or "redacted-")
        self.assertNotEqual(sanitized_cohort["candidate_id"], raw_candidate)
        _assert_activation_privacy_clean(self, result)

    # ==================================================================
    # 11. Dashboard /cache-replay-activation-health endpoint privacy
    # ==================================================================

    def test_dashboard_cache_replay_activation_health_endpoint_privacy(self) -> None:
        rule = _pattern_rule()
        replay_canary = {
            "schema": "tokenclaw.cache_replay_canary_decision.v1",
            "rule_id": _RULE_ID,
            "candidate_id": _CANDIDATE_ID,
            "status": "applied",
            "reason": "dependency-stable",
            "canary": _canary_meta(),
            **_raw_like_extra(),
        }
        holdout_rule = _pattern_rule(cohort="canary_holdout", selected=False, reason="canary_holdout")

        # Applied hit
        self._log_call(
            cache_status="hit",
            cache_reason="exact-match",
            cache_hit=1,
            cost=0.0,
            cost_baseline=0.03,
            cache_extra={
                "pattern_rule": rule,
                "cache_replay_canary": {**replay_canary, **_raw_like_extra()},
                **_raw_like_extra(),
            },
        )
        # Holdout
        self._log_call(
            cache_status="skipped",
            cache_reason="canary_holdout",
            cache_extra={
                "pattern_rule": holdout_rule,
                "pattern_rules": {"skip_reasons": [{**holdout_rule}]},
                "cache_replay_canary": {
                    "rule_id": _RULE_ID,
                    "candidate_id": _CANDIDATE_ID,
                    "status": "holdout",
                    "reason": "canary_holdout",
                    "canary": _canary_meta(cohort="canary_holdout", selected=False),
                    **_raw_like_extra(),
                },
                **_raw_like_extra(),
            },
        )

        app = create_dashboard_app(
            store_obj=lambda: self.store,
            default_db=self.db_path,
            upstream="https://openai.test",
            limiter_status=lambda: [],
            limiter_config={
                "min_request_interval_ms": 0,
                "max_tier_backoff_wait_s": 30,
                "max_concurrent_per_tier": 2,
            },
        )
        client = TestClient(app, raise_server_exceptions=True)
        response = client.get("/tokenclaw/stats/cache-replay-activation-health?limit=50&scan_limit=500")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["schema"], "tokenclaw.cache_replay_activation_health.v1")
        self.assertTrue(payload["read_only"])

        privacy = payload["privacy"]
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["raw_messages_included"])
        self.assertFalse(privacy["raw_request_bodies_included"])
        self.assertFalse(privacy["raw_responses_included"])
        self.assertFalse(privacy["raw_tool_payloads_included"])
        self.assertFalse(privacy["file_paths_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["raw_session_ids_included"])
        self.assertFalse(privacy["cache_keys_included"])
        self.assertFalse(privacy["pattern_hashes_included"])
        self.assertFalse(privacy["policy_file_contents_included"])
        self.assertTrue(privacy["dashboard_read_only"])

        _assert_activation_privacy_clean(self, payload)


if __name__ == "__main__":
    unittest.main()
