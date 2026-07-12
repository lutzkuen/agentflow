from __future__ import annotations

import unittest

from tokenclaw.managed_session_tier import (
    SESSION_TIER_DECISION_SCHEMAS,
    _normalize_decision,
    _session_tier_payload,
)


def _decision(schema: str) -> dict:
    return {
        "schema": schema,
        "tier": "sonnet",
        "confidence": 0.9,
        "session_type": "coding-agent-file-ops",
        "session_tier_source": "managed",
        "reason_codes": ["coding-agent-file-ops"],
        "feature_only": True,
        "locally_executed": True,
        "server_content_processing": False,
        "provider_forwarding": False,
    }


class SessionTierRequestContractTests(unittest.TestCase):
    """Lock the /v1/session-tier wire contract the managed server enforces.

    The server's SessionTierRequest model is strict (extra="forbid") and uses the
    agentflow.* wire vocabulary. Sending tokenclaw.* or the duplicate/extra fields
    phase / input_tokens_est / privacy_summary produced a 422 and silently disabled
    all managed routing (every decision fell back to noop). Guard against re-drift.
    """

    def _payload(self) -> dict:
        unit = {
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "requested_model": "claude-opus-4-8",
            "input_features": {
                "category": "code",
                "workflow_phase": "tool-execution",
                "text_chars": 42000,
                "input_tokens_est": 12000,
            },
            "tool_features": {"has_tools": True},
            "grouping_identifiers": {"session_id_hash": "abc123"},
        }
        return _session_tier_payload(unit, tool_count=8)

    def test_request_uses_agentflow_wire_schema(self):
        self.assertEqual(self._payload()["schema"], "agentflow.session_tier_request.v1")

    def test_request_omits_server_forbidden_extra_fields(self):
        payload = self._payload()
        for forbidden in ("phase", "input_tokens_est", "privacy_summary"):
            self.assertNotIn(forbidden, payload)

    def test_request_uses_server_field_names(self):
        payload = self._payload()
        # Server expects workflow_phase + input_tokens (not phase / input_tokens_est).
        self.assertEqual(payload["workflow_phase"], "tool-execution")
        self.assertEqual(payload["input_tokens"], 12000)

    def test_decision_normalizer_dual_accepts_agentflow_and_tokenclaw(self):
        for schema in ("agentflow.session_tier_decision.v1", "tokenclaw.session_tier_decision.v1"):
            self.assertIn(schema, SESSION_TIER_DECISION_SCHEMAS)
            norm, err = _normalize_decision(_decision(schema))
            self.assertIsNone(err, f"schema {schema} should be accepted")
            self.assertEqual((norm or {}).get("tier"), "sonnet")

    def test_decision_normalizer_rejects_unknown_schema(self):
        _, err = _normalize_decision(_decision("acme.session_tier_decision.v1"))
        self.assertEqual(err, "unsupported-schema")


if __name__ == "__main__":
    unittest.main()
