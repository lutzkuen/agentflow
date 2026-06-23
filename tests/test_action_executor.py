from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import yaml

from tokenclaw.action_executor import ActionExecutor
from tokenclaw.managed_action_outcome_feedback import build_managed_action_feedback


class ActionExecutorTests(unittest.TestCase):
    managed_live_env = {"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "live"}

    def _routing_decision(self, **overrides):
        decision = {
            "enabled": True,
            "status": "received",
            "policy_id": "managed-routing-policy",
            "decision_id": "decision-1",
            "provider": "openai",
            "source_surface": "openai_responses",
            "routing": {
                "target_model": "gpt-5-mini",
                "route_proposal": {
                    "target_model": "gpt-5-mini",
                    "traffic_treatment": "canary",
                    "route_selected": True,
                    "canary_fraction": 0.20,
                    "holdout_fraction": 0.10,
                    "server_selected_canary_membership": True,
                },
            },
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }
        for key, value in overrides.items():
            if key == "route_proposal":
                decision["routing"]["route_proposal"].update(value)
            elif key == "routing":
                decision["routing"].update(value)
            else:
                decision[key] = value
        return decision

    def _crunch_decision(self, *, treatment: str = "widen", fraction: float = 0.25, **overrides):
        decision = {
            "enabled": True,
            "status": "received",
            "policy_id": "managed-crunch-policy",
            "decision_id": "managed-crunch-decision-1",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "crunch": {
                "status": "recommended",
                "profile": "managed",
                "candidate_id": "repeated-context-thinking-tool-result-gte-128k",
                "traffic_treatment": treatment,
                "canary_fraction": fraction,
                "holdout_fraction": 0.33,
            },
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }
        for key, value in overrides.items():
            if key == "crunch":
                decision["crunch"].update(value)
            else:
                decision[key] = value
        return decision

    def _write_crunch_rules(self, config: Path, *, fraction: float = 0.05, holdout: float = 0.10) -> Path:
        config.mkdir(parents=True, exist_ok=True)
        path = config / "crunch_rules.yaml"
        path.write_text(
            f"""
enabled: true
anthropic_thinking_history_compaction:
  enabled: true
  policy_source: local-manual
  canary:
    enabled: true
    canary_fraction: {fraction}
    holdout_fraction: {holdout}
  rules:
    - id: local-repeated-context-thinking-tool-result-canary
      enabled: true
      policy_source: local-manual
      candidate_id: repeated-context-thinking-tool-result-gte-128k
      canary:
        enabled: true
        canary_fraction: {fraction}
        holdout_fraction: {holdout}
      safety_stop:
        enabled: true
""",
            encoding="utf-8",
        )
        return path

    def _execute_crunch(self, *, config: Path, decision: dict, env: dict[str, str] | None = None, **executor_kwargs):
        body = {"model": "claude-sonnet-4-6", "messages": []}
        with patch.dict(os.environ, env or self.managed_live_env, clear=False):
            result = ActionExecutor(provider="anthropic", config_dir=str(config), **executor_kwargs).execute(
                body=body,
                routing_meta={"routed_model": "claude-sonnet-4-6", "source_surface": "anthropic_messages"},
                decision=decision,
                application_enabled=True,
                source_surface="anthropic_messages",
            )
        return result

    def test_applies_only_provider_body_model_rewrite_for_routing(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta=routing_meta,
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "route-policy",
                    "provider": "openai",
                    "source_surface": "openai_responses",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertEqual(routing_meta["routed_model"], "gpt-5-mini")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["routing"]["status"], "applied")
        self.assertEqual(result["routing"]["apply_reason"], "provider-body-model-rewrite")
        self.assertEqual(result["outcome_feedback"]["status"], "applied")
        self.assertEqual(result["product_mode"]["mode"], "live")

    def test_canary_selected_by_server_applies_route_and_records_treatment(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta=routing_meta,
                decision=self._routing_decision(),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-mini")
        self.assertEqual(routing_meta["routed_model"], "gpt-5-mini")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["server_traffic_treatment"], "canary")
        self.assertTrue(result["server_route_selected"])
        self.assertEqual(result["canary_fraction"], 0.20)
        self.assertEqual(result["holdout_fraction"], 0.10)
        self.assertEqual(result["routing"]["status"], "applied")
        self.assertIn("routing", result["outcome_feedback"]["applied_families"])

    def test_canary_holdout_preserves_body_and_records_heldout_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        routing_meta = {"routed_model": "gpt-5-codex", "source_surface": "openai_responses"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta=routing_meta,
                decision=self._routing_decision(route_proposal={
                    "traffic_treatment": "holdout",
                    "route_selected": False,
                    "server_selected_canary_membership": False,
                }),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(routing_meta["routed_model"], "gpt-5-codex")
        self.assertEqual(result["status"], "heldout")
        self.assertEqual(result["apply_reason"], "server-canary-holdout")
        self.assertEqual(result["routing"]["status"], "heldout")
        self.assertEqual(result["routing"]["target_model"], "gpt-5-mini")
        self.assertEqual(result["outcome_feedback"]["status"], "heldout")
        self.assertIn("routing", result["outcome_feedback"]["heldout_families"])

        feedback = build_managed_action_feedback(result, source_surface="openai_responses")
        self.assertEqual(feedback["local_result"], "heldout")
        self.assertEqual(feedback["server_traffic_treatment"], "holdout")
        self.assertFalse(feedback["server_route_selected"])
        self.assertIn("routing", feedback["heldout_families"])

    def test_observe_treatment_preserves_body_even_when_live_mode_allows_application(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex", "source_surface": "openai_responses"},
                decision=self._routing_decision(route_proposal={
                    "traffic_treatment": "observe",
                    "route_selected": False,
                }),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "observe-only")
        self.assertEqual(result["routing"]["status"], "held")

    def test_expired_decision_is_vetoed_before_application(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai", now=datetime(2026, 1, 2, tzinfo=timezone.utc)).execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex", "source_surface": "openai_responses"},
                decision=self._routing_decision(expires_at="2026-01-01T00:00:00+00:00"),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["apply_reason"], "expired-policy")
        self.assertEqual(result["outcome_feedback"]["status"], "vetoed")

    def test_unsupported_target_model_is_vetoed(self):
        body = {"model": "claude-sonnet-4-6", "messages": []}
        decision = {
            "enabled": True,
            "status": "received",
            "policy_id": "unsupported-target",
            "decision_id": "unsupported-1",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "routing": {
                "target_model": "claude-future-unknown",
                "route_proposal": {
                    "target_model": "claude-future-unknown",
                    "traffic_treatment": "live",
                    "route_selected": True,
                },
            },
            "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
        }
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="anthropic").execute(
                body=body,
                routing_meta={"routed_model": "claude-sonnet-4-6", "source_surface": "anthropic_messages"},
                decision=decision,
                application_enabled=True,
                source_surface="anthropic_messages",
            )

        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["routing"]["veto_reason"], "unsupported-target-model")
        self.assertIn("routing", result["outcome_feedback"]["vetoed_families"])

    def test_unsupported_actions_are_vetoed_with_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "unsafe-policy",
                    "provider": "openai",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "actions": [{"type": "replacement_prompt"}],
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["apply_reason"], "unsupported-action-type")
        self.assertEqual(result["unsupported_actions"][0]["reason"], "unsupported-action-type")
        self.assertEqual(result["outcome_feedback"]["status"], "vetoed")

    def test_locally_disabled_action_family_is_vetoed(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, self.managed_live_env, clear=False):
            result = ActionExecutor(provider="openai", supported_action_families=("crunch", "cache")).execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "routing-disabled",
                    "provider": "openai",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["routing"]["status"], "vetoed")
        self.assertIn("routing", result["outcome_feedback"]["vetoed_families"])

    def test_product_family_opt_out_vetoes_routing_with_feedback(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, {**self.managed_live_env, "TOKENCLAW_MANAGED_ROUTING": "0"}, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "routing-product-disabled",
                    "provider": "openai",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["routing"]["veto_reason"], "local-action-family-disabled")
        self.assertIn("routing", result["outcome_feedback"]["vetoed_families"])
        self.assertFalse(result["outcome_feedback"]["raw_payload_included"])

    def test_observe_only_holds_without_changing_body(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(os.environ, {"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "observe_only"}, clear=False):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "observe-policy",
                    "provider": "openai",
                    "target_model": "gpt-5-mini",
                    "routing": {"target_model": "gpt-5-mini"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "managed-mode-observe_only")
        self.assertEqual(result["product_mode"]["mode"], "observe_only")

    def test_local_rules_only_holds_without_server_application(self):
        body = {"model": "gpt-5-codex", "input": "hello"}
        with patch.dict(
            os.environ,
            {**self.managed_live_env, "TOKENCLAW_LOCAL_RULES_ONLY": "1"},
            clear=False,
        ):
            result = ActionExecutor(provider="openai").execute(
                body=body,
                routing_meta={"routed_model": "gpt-5-codex", "source_surface": "openai_responses"},
                decision=self._routing_decision(),
                application_enabled=True,
                source_surface="openai_responses",
            )

        self.assertEqual(body["model"], "gpt-5-codex")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "local-rules-only")
        self.assertEqual(result["product_mode"]["mode"], "local_only")
        self.assertFalse(result["outcome_feedback"]["raw_payload_included"])

    def test_executor_disabled_holds_without_changing_forwarding_body(self):
        body = {"model": "claude-sonnet-4-6"}
        with patch.dict(os.environ, {**self.managed_live_env, "TOKENCLAW_ACTION_EXECUTOR_ENABLED": "0"}):
            result = ActionExecutor(provider="anthropic").execute(
                body=body,
                routing_meta={"routed_model": "claude-sonnet-4-6"},
                decision={
                    "enabled": True,
                    "status": "received",
                    "policy_id": "held-policy",
                    "provider": "anthropic",
                    "target_model": "claude-haiku-4-5-20251001",
                    "routing": {"target_model": "claude-haiku-4-5-20251001"},
                    "privacy_summary": {"metadata_only": True, "raw_payload_included": False},
                },
                application_enabled=True,
            )

        self.assertEqual(body["model"], "claude-sonnet-4-6")
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "action-executor-disabled")
        self.assertEqual(result["fallback"], "local-policy")

    def test_managed_crunch_hold_treatment_leaves_thinking_compaction_rule_unchanged(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="hold", fraction=0.25),
            )
            after = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "held")
        self.assertEqual(result["crunch"]["status"], "held")
        self.assertEqual(before, after)
        self.assertIn("crunch", result["outcome_feedback"]["held_families"])

    def test_managed_crunch_widen_updates_canary_fraction_and_preserves_holdout(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="widen", fraction=0.25),
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["crunch"]["status"], "applied")
        self.assertEqual(result["crunch"]["traffic_treatment_policy_file"]["status"], "applied")
        self.assertAlmostEqual(rule["canary"]["canary_fraction"], 0.25)
        self.assertAlmostEqual(rule["canary"]["holdout_fraction"], 0.10)
        self.assertEqual(rule["policy_source"], "managed-recommended")
        self.assertEqual(rule["managed_controller"]["server_traffic_treatment"], "widen")
        self.assertIn("crunch", result["outcome_feedback"]["applied_families"])

    def test_managed_crunch_canary_updates_policy_file_without_route_membership(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="canary", fraction=0.15),
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertEqual(result["status"], "applied")
        self.assertAlmostEqual(rule["canary"]["canary_fraction"], 0.15)
        self.assertEqual(rule["managed_controller"]["server_traffic_treatment"], "canary")

    def test_managed_crunch_rollback_disables_candidate_and_sets_safety_stop(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.25, holdout=0.10)
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="rollback", fraction=0.0),
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        section = data["anthropic_thinking_history_compaction"]
        rule = section["rules"][0]
        self.assertEqual(result["status"], "applied")
        self.assertFalse(section["enabled"])
        self.assertFalse(rule["enabled"])
        self.assertAlmostEqual(rule["canary"]["canary_fraction"], 0.0)
        self.assertEqual(rule["safety_stop"]["last_managed_rollback_reason"], "server-rollback")
        self.assertEqual(rule["managed_controller"]["server_traffic_treatment"], "rollback")

    def test_managed_crunch_dry_run_reports_exact_policy_file_diff_without_writing(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="widen", fraction=0.20),
                env={"TOKENCLAW_MANAGED": "1", "TOKENCLAW_MANAGED_MODE": "dry_run"},
            )
            after = path.read_text(encoding="utf-8")

        policy_file = result["crunch"]["traffic_treatment_policy_file"]
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["apply_reason"], "managed-mode-dry_run")
        self.assertEqual(result["crunch"]["status"], "dry-run")
        self.assertEqual(policy_file["status"], "planned")
        self.assertIn("+    canary_fraction: 0.2", policy_file["diff"])
        self.assertEqual(before, after)

    def test_managed_crunch_family_opt_out_vetoes_with_feedback_and_no_file_write(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="widen", fraction=0.25),
                env={**self.managed_live_env, "TOKENCLAW_MANAGED_CRUNCH": "0"},
            )
            after = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["crunch"]["veto_reason"], "local-action-family-disabled")
        self.assertIn("crunch", result["outcome_feedback"]["vetoed_families"])
        self.assertEqual(before, after)

    def test_managed_crunch_unsigned_and_expired_decisions_are_vetoed_without_file_write(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            unsigned = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="widen", fraction=0.25),
                require_signature=True,
            )
            expired = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(
                    treatment="widen",
                    fraction=0.25,
                    expires_at="2026-01-01T00:00:00+00:00",
                ),
                now=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            after = path.read_text(encoding="utf-8")

        self.assertEqual(unsigned["status"], "vetoed")
        self.assertEqual(unsigned["apply_reason"], "unsigned-policy")
        self.assertEqual(expired["status"], "vetoed")
        self.assertEqual(expired["apply_reason"], "expired-policy")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
