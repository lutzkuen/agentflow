from __future__ import annotations

import json
import unittest

from agentflow_proxy.optimization_coordinator import build_optimization_coordinator


FORBIDDEN_VALUES = (
    "raw-coordinator-prompt-secret",
    "raw-coordinator-response-secret",
    "raw-coordinator-session-secret",
    "cache-key-coordinator-secret",
    "/home/lutz/private/coordinator_secret.py",
    "terminal output raw line",
    "tool payload secret",
    "local-salt-secret",
)

FORBIDDEN_KEYS = (
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages":',
    '"prompt"',
    '"raw_request"',
    '"request_id"',
    '"response_json"',
    '"session_id"',
    '"tool_payload"',
)


def ledger_with(*entries: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "agentflow.optimization_action_ledger.v1",
        "entry_count": len(entries),
        "entries": list(entries),
        "privacy": {
            "metadata_only": True,
            "provider_bodies_included": False,
        },
    }


def entry(
    family: str,
    *,
    status: str = "eligible",
    reason_codes: list[str] | None = None,
    candidate_id: str | None = None,
    policy_source: str = "managed-recommended",
) -> dict[str, object]:
    return {
        "schema": "agentflow.optimization_action_ledger_entry.v1",
        "family": family,
        "source_surface": "openai_responses",
        "provider_family": "openai",
        "endpoint": "responses",
        "category": "tool-result",
        "phase": "tool-execution",
        "text_bucket": "8k_30k",
        "input_token_bucket": "2k_8k",
        "policy_source": policy_source,
        "status": status,
        "reason_codes": reason_codes or [],
        "candidate_id": candidate_id or f"{family}-candidate",
        "projected_savings_bucket": "0_01_0_10",
    }


class OptimizationCoordinatorTests(unittest.TestCase):
    def assert_private(self, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True)
        for value in FORBIDDEN_VALUES:
            self.assertNotIn(value, rendered)
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(key, rendered)

    def test_selects_one_highest_priority_family_and_suppresses_conflicts(self) -> None:
        decision = build_optimization_coordinator(
            ledger=ledger_with(
                entry("routing", candidate_id="routing-candidate"),
                entry("cache_replay", reason_codes=["dependency-stable"], candidate_id="cache-candidate"),
                entry("terminal_output_compaction", candidate_id="terminal-candidate"),
            ),
            local_salt="coordinator-test",
        )

        self.assertEqual(decision["schema"], "agentflow.optimization_coordinator.v1")
        self.assertEqual(decision["selected_family"], "cache_replay")
        self.assertEqual(decision["selected_candidate"]["candidate_id"], "cache-candidate")
        self.assertEqual(decision["family_status"]["cache_replay"]["selected"], True)
        suppressed = {item["family"]: item for item in decision["suppressed_families"]}
        self.assertEqual(suppressed["routing"]["reason_codes"], ["conflicts-with-selected-family"])
        self.assertEqual(suppressed["terminal_output_compaction"]["reason_codes"], ["conflicts-with-selected-family"])
        self.assertTrue(decision["conservative_single_mutation"])
        self.assertFalse(decision["provider_body_changed"])
        self.assertFalse(decision["policy_files_changed"])

    def test_holdout_is_deterministic_and_selects_no_family(self) -> None:
        ledger = ledger_with(entry("routing"))

        first = build_optimization_coordinator(
            ledger=ledger,
            local_salt="coordinator-test",
            holdout_fraction=1.0,
            canary_fraction=0.0,
        )
        second = build_optimization_coordinator(
            ledger=ledger,
            local_salt="coordinator-test",
            holdout_fraction=1.0,
            canary_fraction=0.0,
        )

        self.assertEqual(first["selected_family"], "none")
        self.assertEqual(first["canary"]["cohort"], "coordinator_holdout")
        self.assertEqual(first["suppressed_families"][0]["reason_codes"], ["coordinator-holdout"])
        self.assertEqual(first["decision_hash"], second["decision_hash"])
        self.assertEqual(first["canary"]["cohort_key_hash"], second["canary"]["cohort_key_hash"])

    def test_noop_when_no_eligible_families(self) -> None:
        decision = build_optimization_coordinator(
            ledger=ledger_with(entry("routing", status="suppressed", reason_codes=["disabled"])),
            local_salt="coordinator-test",
        )

        self.assertEqual(decision["selected_family"], "none")
        self.assertEqual(decision["candidate_count"], 0)
        self.assertEqual(decision["reason_codes"], ["no-eligible-families"])
        self.assertEqual(decision["family_status"]["routing"]["selected"], False)

    def test_stale_evidence_suppresses_candidate_without_selection(self) -> None:
        decision = build_optimization_coordinator(
            ledger=ledger_with(entry("routing", reason_codes=["stale-evidence"])),
            local_salt="coordinator-test",
        )

        self.assertEqual(decision["selected_family"], "none")
        self.assertEqual(decision["candidate_count"], 0)
        self.assertEqual(decision["suppressed_families"][0]["family"], "routing")
        self.assertEqual(decision["suppressed_families"][0]["reason_codes"], ["stale-evidence"])

    def test_cache_replay_requires_dependency_freshness_evidence(self) -> None:
        decision = build_optimization_coordinator(
            ledger=ledger_with(entry("cache_replay")),
            local_salt="coordinator-test",
        )

        self.assertEqual(decision["selected_family"], "none")
        self.assertEqual(decision["suppressed_families"][0]["family"], "cache_replay")
        self.assertEqual(decision["suppressed_families"][0]["reason_codes"], ["missing-dependency-freshness-evidence"])

    def test_safety_stop_entry_takes_priority_and_suppresses_other_families(self) -> None:
        decision = build_optimization_coordinator(
            ledger=ledger_with(
                entry("routing", reason_codes=["safety-stop-tripped"], candidate_id="rollback-action"),
                entry("cache_replay", reason_codes=["dependency-stable"], candidate_id="cache-candidate"),
            ),
            local_salt="coordinator-test",
        )

        self.assertEqual(decision["selected_family"], "routing")
        self.assertEqual(decision["selected_candidate"]["candidate_id"], "rollback-action")
        self.assertEqual(decision["reason_codes"], ["safety-stop-priority"])
        self.assertEqual(decision["suppressed_families"][0]["family"], "cache_replay")
        self.assertEqual(decision["suppressed_families"][0]["reason_codes"], ["conflicts-with-selected-family"])

    def test_builds_from_metadata_without_leaking_raw_values(self) -> None:
        decision = build_optimization_coordinator(
            row={
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "actual_input_tokens": 600,
                "cost_est_usd": 0.003,
                "cost_baseline_usd": 0.012,
                "request_json": {
                    "request_id": "raw-coordinator-session-secret",
                    "messages": [{"content": "raw-coordinator-prompt-secret"}],
                    "cache_key": "cache-key-coordinator-secret",
                    "file_path": "/home/lutz/private/coordinator_secret.py",
                },
                "response_json": {"content": "raw-coordinator-response-secret"},
                "session_id": "raw-coordinator-session-secret",
            },
            routing_meta={
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "tool-execution",
                "text_chars": 2400,
                "enabled": True,
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "reason": "small-request",
            },
            crunch_meta={
                "terminal_output_compaction": {
                    "status": "eligible",
                    "candidate_id": "raw-coordinator-session-secret",
                    "terminal_line": "terminal output raw line",
                },
            },
            cache_meta={
                "status": "miss",
                "reason": "exact-miss",
                "cache_key": "cache-key-coordinator-secret",
                "tool_payload": "tool payload secret",
            },
            local_salt="local-salt-secret",
        )

        self.assertEqual(decision["selected_family"], "routing")
        self.assertTrue(decision["decision_hash"].startswith("sha256:"))
        self.assertFalse(decision["privacy"]["local_salt_included"])
        self.assertFalse(decision["privacy"]["provider_bodies_included"])
        self.assert_private(decision)


if __name__ == "__main__":
    unittest.main()
