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
from tokenclaw.store import Store


class ActionExecutorTests(unittest.TestCase):
    managed_live_env = {
        "TOKENCLAW_MANAGED": "1",
        "TOKENCLAW_MANAGED_MODE": "live",
        "TOKENCLAW_LOCAL_RULES_ONLY": "0",
        "TOKENCLAW_MANAGED_ROUTING": "1",
        "TOKENCLAW_MANAGED_CRUNCH": "1",
        "TOKENCLAW_MANAGED_CACHE": "1",
    }

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

    def _crunch_schedule(
        self,
        *,
        target: str = "widen",
        current: float = 0.10,
        cap: float = 0.50,
        increment: float = 0.40,
        holdout: float = 0.05,
        expires_at: str = "2026-01-02T00:00:00+00:00",
    ) -> dict[str, object]:
        return {
            "schema": "agentflow.thinking_tail_widening_schedule.v1",
            "action_family": "crunch",
            "candidate_id": "thinking-tail-compaction",
            "policy_id": "managed-crunch-policy",
            "schedule_status": "widen-ready",
            "treatment_target": target,
            "current_fraction": current,
            "next_fraction_cap": cap,
            "max_fraction_increment": increment,
            "holdout_fraction": holdout,
            "minimum_applied_samples": 20,
            "minimum_holdout_samples": 3,
            "freshness_window_seconds": 604800,
            "expires_at": expires_at,
            "reason_codes": ["thinking-tail-widening-schedule-bounded"],
            "feature_only": True,
            "metadata_only": True,
            "locally_executed": True,
        }

    def _crunch_decision_with_schedule(self, *, schedule: dict[str, object] | None = None, **overrides):
        schedule = schedule or self._crunch_schedule()
        extra_crunch = overrides.pop("crunch", {})
        if not isinstance(extra_crunch, dict):
            extra_crunch = {}
        crunch_section = {
            "candidate_id": "thinking-tail-compaction",
            "thinking_tail_readiness": {
                "schema": "agentflow.thinking_tail_readiness_summary.v1",
                "source": "policy-decision",
                "action_family": "crunch",
                "candidate_id": "thinking-tail-compaction",
                "policy_id": "managed-crunch-policy",
                "readiness_state": "widen-ready",
                "requested_next_local_action": "widen",
                "minimum_local_client_version": "0.1.0",
                "required_local_capabilities": ["crunch"],
                "traffic_treatment": str(schedule.get("treatment_target") or "widen"),
                "canary_fraction": float(schedule.get("next_fraction_cap") or 0.0),
                "holdout_fraction": float(schedule.get("holdout_fraction") or 0.0),
                "widening_schedule": schedule,
                "feature_only": True,
                "metadata_only": True,
                "locally_executed": True,
            },
        }
        crunch_section.update(extra_crunch)
        decision = self._crunch_decision(
            treatment=str(schedule.get("treatment_target") or "widen"),
            fraction=float(schedule.get("next_fraction_cap") or 0.0),
            crunch=crunch_section,
            **overrides,
        )
        return decision

    def _write_crunch_rules(self, config: Path, *, fraction: float = 0.05, holdout: float = 0.10, manual_disabled: bool = False) -> Path:
        config.mkdir(parents=True, exist_ok=True)
        path = config / "crunch_rules.yaml"
        enabled = not manual_disabled
        path.write_text(
            f"""
enabled: true
anthropic_thinking_history_compaction:
  enabled: {str(enabled).lower()}
  policy_source: local-manual
  canary:
    enabled: true
    canary_fraction: {fraction}
    holdout_fraction: {holdout}
  rules:
    - id: local-repeated-context-thinking-tool-result-canary
      enabled: {str(enabled).lower()}
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

    def _assignment_store(self, tmp: str):
        return Store(str(Path(tmp) / "assignments.sqlite3"))

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

    def test_managed_crunch_widening_schedule_caps_fraction_and_records_metadata(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.10, holdout=0.10)
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision_with_schedule(
                    schedule=self._crunch_schedule(current=0.10, cap=0.90, increment=0.20, holdout=0.07)
                ),
                now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        policy_file = result["crunch"]["traffic_treatment_policy_file"]
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["server_traffic_treatment"], "widen")
        self.assertAlmostEqual(rule["canary"]["canary_fraction"], 0.30)
        self.assertAlmostEqual(rule["canary"]["holdout_fraction"], 0.07)
        self.assertAlmostEqual(policy_file["recommended_fraction"], 0.30)
        self.assertAlmostEqual(policy_file["schedule_next_fraction_cap"], 0.90)
        self.assertEqual(
            rule["managed_controller"]["widening_schedule"]["schema"],
            "agentflow.thinking_tail_widening_schedule.v1",
        )
        self.assertIn("crunch", result["outcome_feedback"]["applied_families"])

    def test_expired_managed_crunch_widening_schedule_vetoes_without_file_write(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.10, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision_with_schedule(
                    schedule=self._crunch_schedule(expires_at="2026-01-01T00:00:00+00:00")
                ),
                now=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            after = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["apply_reason"], "expired-widening-schedule")
        self.assertEqual(result["crunch"]["status"], "vetoed")
        self.assertEqual(result["crunch"]["veto_reason"], "expired-widening-schedule")
        self.assertIn("crunch", result["outcome_feedback"]["vetoed_families"])
        self.assertEqual(before, after)

    def test_widening_schedule_respects_local_crunch_gate_without_assignment(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.10, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            store = self._assignment_store(tmp)
            try:
                result = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision_with_schedule(),
                    env={**self.managed_live_env, "TOKENCLAW_MANAGED_CRUNCH": "0"},
                    store_obj=store,
                    session_id="raw-schedule-disabled",
                    now=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
                rows = store.managed_thinking_tail_assignment_rows()
            finally:
                store.conn.close()
            after = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "vetoed")
        self.assertEqual(result["crunch"]["veto_reason"], "local-action-family-disabled")
        self.assertEqual(before, after)
        self.assertEqual(rows, [])

    def test_rollback_assignment_retains_stop_for_later_widening_schedule_action(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.25, holdout=0.10)
            store = self._assignment_store(tmp)
            try:
                rollback = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision_with_schedule(
                        schedule=self._crunch_schedule(target="rollback", current=0.25, cap=0.0, increment=0.0),
                        crunch={"action_id": "rollback-action"},
                    ),
                    store_obj=store,
                    session_id="raw-schedule-rollback",
                    now=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
                widen = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision_with_schedule(
                        schedule=self._crunch_schedule(target="widen", current=0.0, cap=0.50, increment=0.25),
                        crunch={"action_id": "new-widen-action"},
                    ),
                    store_obj=store,
                    session_id="raw-schedule-rollback",
                    now=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
                rows = store.managed_thinking_tail_assignment_rows()
            finally:
                store.conn.close()
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertEqual(rollback["server_traffic_treatment"], "rollback")
        self.assertEqual(widen["server_traffic_treatment"], "rollback")
        self.assertEqual(widen["managed_thinking_tail_assignment"]["reason"], "rollback-retained")
        self.assertFalse(rule["enabled"])
        self.assertAlmostEqual(rule["canary"]["canary_fraction"], 0.0)
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row["treatment"] == "rollback" for row in rows))

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

    def test_managed_crunch_sticky_assignment_reuses_session_treatment_despite_later_widen(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            store = self._assignment_store(tmp)
            try:
                first = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision(treatment="canary", fraction=0.15),
                    store_obj=store,
                    session_id="raw-session-id-1",
                )
                second = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision(treatment="widen", fraction=0.55),
                    store_obj=store,
                    session_id="raw-session-id-1",
                )
                rows = store.managed_thinking_tail_assignment_rows()
            finally:
                store.conn.close()
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertEqual(first["managed_thinking_tail_assignment"]["status"], "recorded")
        self.assertEqual(second["managed_thinking_tail_assignment"]["status"], "reused")
        self.assertEqual(second["server_traffic_treatment"], "canary")
        self.assertEqual(second["crunch"]["traffic_treatment_policy_file"]["server_traffic_treatment"], "canary")
        self.assertAlmostEqual(rule["canary"]["canary_fraction"], 0.15)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["treatment"], "canary")
        self.assertAlmostEqual(row["canary_fraction"], 0.15)
        self.assertTrue(str(row["session_key_hash"]).startswith("sha256:"))
        self.assertNotIn("raw-session-id-1", str(row))
        self.assertNotIn("thinking text", str(row).lower())
        self.assertNotIn("/home/", str(row))

    def test_managed_crunch_rollback_overrides_sticky_assignment(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            store = self._assignment_store(tmp)
            try:
                self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision(treatment="canary", fraction=0.15),
                    store_obj=store,
                    session_id="raw-session-id-2",
                )
                result = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision(treatment="rollback", fraction=0.0),
                    store_obj=store,
                    session_id="raw-session-id-2",
                )
                rows = store.managed_thinking_tail_assignment_rows()
            finally:
                store.conn.close()
            data = yaml.safe_load(path.read_text(encoding="utf-8"))

        rule = data["anthropic_thinking_history_compaction"]["rules"][0]
        self.assertEqual(result["server_traffic_treatment"], "rollback")
        self.assertEqual(result["managed_thinking_tail_assignment"]["status"], "recorded")
        self.assertFalse(rule["enabled"])
        self.assertEqual(rows[0]["treatment"], "rollback")
        self.assertAlmostEqual(rows[0]["canary_fraction"], 0.0)

    def test_managed_crunch_sticky_assignment_skips_disabled_crunch_gate(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            store = self._assignment_store(tmp)
            try:
                result = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision(treatment="widen", fraction=0.25),
                    env={**self.managed_live_env, "TOKENCLAW_MANAGED_CRUNCH": "0"},
                    store_obj=store,
                    session_id="raw-session-id-3",
                )
                rows = store.managed_thinking_tail_assignment_rows()
            finally:
                store.conn.close()
            after = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "vetoed")
        self.assertNotIn("managed_thinking_tail_assignment", result)
        self.assertEqual(before, after)
        self.assertEqual(rows, [])

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

    def test_managed_crunch_treatment_respects_local_manual_disabled_rule(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.0, holdout=0.10, manual_disabled=True)
            before = path.read_text(encoding="utf-8")
            store = self._assignment_store(tmp)
            try:
                result = self._execute_crunch(
                    config=config,
                    decision=self._crunch_decision(treatment="widen", fraction=0.25),
                    store_obj=store,
                    session_id="raw-session-id-manual-disabled",
                )
                rows = store.managed_thinking_tail_assignment_rows()
            finally:
                store.conn.close()
            after = path.read_text(encoding="utf-8")

        policy_file = result["crunch"]["traffic_treatment_policy_file"]
        self.assertEqual(result["status"], "noop")
        self.assertEqual(result["apply_reason"], "no-local-actions-applied")
        self.assertEqual(result["crunch"]["status"], "held")
        self.assertEqual(result["crunch"]["apply_reason"], "local-manual-disabled")
        self.assertTrue(policy_file["manual_disabled_authoritative"])
        self.assertEqual(policy_file["reason"], "local-manual-disabled")
        self.assertEqual(before, after)
        self.assertEqual(rows[0]["local_veto_reason"], "local-manual-disabled")
        self.assertNotIn("raw-session-id-manual-disabled", str(rows[0]))

    def test_managed_crunch_dry_run_reports_exact_policy_file_diff_without_writing(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            path = self._write_crunch_rules(config, fraction=0.05, holdout=0.10)
            before = path.read_text(encoding="utf-8")
            result = self._execute_crunch(
                config=config,
                decision=self._crunch_decision(treatment="widen", fraction=0.20),
                env={**self.managed_live_env, "TOKENCLAW_MANAGED_MODE": "dry_run"},
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
