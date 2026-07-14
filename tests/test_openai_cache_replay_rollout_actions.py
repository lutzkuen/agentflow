import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import yaml

from tokenclaw import cli
from tokenclaw.openai_cache_replay_rollout_actions import (
    attach_openai_cache_replay_rollout_provenance,
)


class ManagedFeedbackFlushClient:
    calls = []

    def __init__(self, *, timeout=None, **kwargs):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {}), "timeout": self.timeout})
        return httpx.Response(200, text='{"ok":true}')


class OpenAICacheReplayRolloutActionsTests(unittest.TestCase):
    def setUp(self):
        ManagedFeedbackFlushClient.calls = []
        self.tmp = TemporaryDirectory()
        self.old_event_log = os.environ.get("TOKENCLAW_POLICY_EVENTS_LOG")
        os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")
        self.old_secret = os.environ.get("TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET")
        os.environ["TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET"] = ""

    def tearDown(self):
        if self.old_event_log is None:
            os.environ.pop("TOKENCLAW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = self.old_event_log
        if self.old_secret is None:
            os.environ.pop("TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET", None)
        else:
            os.environ["TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET"] = self.old_secret
        self.tmp.cleanup()

    def _policy_path(self, config_dir: str) -> Path:
        pattern_hash = "sha256:" + "a" * 64
        path = Path(config_dir) / "cache_canary_policy.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema": "tokenclaw.openai_cache_replay_canary_policy.v1",
                    "policy_source": "managed-recommended",
                    "pattern_rules": [
                        {
                            "id": "managed-openai-cache-replay-rule",
                            "candidate_id": "openai-cache-replay-candidate",
                            "enabled": True,
                            "policy_source": "managed-recommended",
                            "conditions": {
                                "pattern_hashes": [pattern_hash],
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "has_tools": False,
                                "stream": False,
                                "replayability_levels": ["local-exact-response"],
                            },
                            "action": {
                                "type": "exact_cache_pattern",
                                "allow_tool_calls": False,
                                "safe_invalidation": True,
                                "scope": "session",
                            },
                            "rollout": {
                                "schema": "tokenclaw.pattern_policy_rollout.v1",
                                "recommendation_mode": "canary",
                                "canary_enabled": True,
                                "canary_fraction": 0.25,
                                "holdout_fraction": 0.75,
                                "canary_salt": "local-cache-replay",
                                "canary_unit": "request_fingerprint",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _bundle(self, **overrides):
        action = {
            "schema": "tokenclaw.optimization_rollout_action.v1",
            "action_id": "openai-cache-replay-rollout-action",
            "action_type": "widen",
            "target_candidate_id": "openai-cache-replay-candidate",
            "action_family": "cache",
            "candidate_family": "cache-policy-rule",
            "policy_section": "cache",
            "source_surface": "openai_responses",
            "provider_endpoint": "responses",
            "confidence": 0.82,
            "generated_at": "2026-06-12T15:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "required_local_review": True,
            "managed_enforced": False,
            "local_executor_compatibility": {
                "minimum_local_client_version": "0.1.0",
                "compatible": True,
                "supported_local_action_families": ["cache"],
                "local_review_required": True,
                "requires_cache_rule_support": True,
            },
            "evidence_summary": {
                "rollout_gate": {"status": "pass", "reason_codes": []},
                "openai_cache_replay": {"sample_counts": {"applied": 4, "holdout": 4}},
            },
            "action": {
                "schema": "tokenclaw.openai_cache_replay_rollout_review_action.v1",
                "status": "review-local-openai-cache-replay-rule",
                "local_action": "cache",
                "locally_executable": True,
                "locally_executed": True,
                "requires_local_review": True,
                "managed_enforced": False,
                "provider_forwarding": False,
                "server_content_processing": False,
                "proposed_edit": {
                    "policy_section": "cache",
                    "policy_source": "managed-recommended",
                    "rule_id": "managed-openai-cache-replay-rule",
                    "candidate_id": "openai-cache-replay-candidate",
                    "source_surface": "openai_responses",
                    "provider_endpoint": "responses",
                    "conditions": {
                        "source_surface": "openai_responses",
                        "provider_endpoint": "responses",
                        "requires_dependency_evidence": True,
                        "requires_quality_gate": True,
                        "requires_canary_holdout": True,
                    },
                    "action": {
                        "type": "exact_cache_replay",
                        "mode": "openai-local-review",
                        "enabled": True,
                        "review_only": True,
                        "canary_fraction": 0.5,
                        "holdout_fraction": 0.5,
                        "dependency_requirements": {
                            "safe_invalidation_evidence": True,
                            "stable_dependency_snapshot": True,
                            "local_dependency_identifiers_only": True,
                        },
                    },
                },
            },
            "privacy_summary": {
                "metadata_only": True,
                "feature_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "tool_payloads_returned": False,
                "request_ids_returned": False,
                "tenant_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
                "locally_executed": True,
                "provider_forwarding": False,
                "managed_enforced": False,
            },
        }
        bundle = {
            "schema": "tokenclaw.openai_cache_replay_rollout_actions.v1",
            "generated_at": "2026-06-12T15:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "summary": {
                "action_count": 1,
                "managed_enforced": False,
                "required_local_review": True,
                "feature_only": True,
                "locally_executed": True,
                "provider_forwarding": False,
                "server_content_processing": False,
            },
            "local_executor_compatibility": {
                "minimum_local_client_version": "0.1.0",
                "compatible": True,
                "supported_local_action_families": ["cache"],
                "local_review_required": True,
                "requires_cache_rule_support": True,
            },
            "actions": [action],
            "omitted_actions": [],
            "privacy_summary": {
                "metadata_only": True,
                "feature_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "tool_payloads_returned": False,
                "request_ids_returned": False,
                "tenant_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
                "provider_forwarding": False,
                "managed_enforced": False,
            },
        }
        for key, value in overrides.items():
            if key == "action_updates":
                action.update(value)
            elif key == "proposed_updates":
                action["action"]["proposed_edit"].update(value)
            else:
                bundle[key] = value
        return bundle

    def _signed(self, bundle):
        return attach_openai_cache_replay_rollout_provenance(
            bundle,
            secret="cache-replay-secret",
            issuer="tokenclaw-server",
            server_id="managed-test",
            key_id="cache-replay-key",
            generated_at="2026-06-12T15:00:00+00:00",
        )

    def test_signed_openai_cache_replay_actions_review_dry_run_and_apply_overlay(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET": "cache-replay-secret"}, clear=False):
            policy_path = self._policy_path(tmp)
            before = policy_path.read_text(encoding="utf-8")
            signed = self._signed(self._bundle())

            review_out = io.StringIO()
            review_code = cli.managed_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(signed)),
                stdout=review_out,
                stderr=io.StringIO(),
            )
            self.assertEqual(review_code, 0)
            review = json.loads(review_out.getvalue())
            self.assertEqual(review["schema"], "tokenclaw.openai_cache_replay_rollout_actions_review.v1")
            self.assertEqual(review["provenance"]["status"], "verified")
            self.assertEqual(review["planned_action_count"], 1)
            self.assertEqual(review["actions"][0]["proposed_edit"]["recommended_fraction"], 0.5)

            dry_out = io.StringIO()
            dry_code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(signed)),
                stdout=dry_out,
            )
            self.assertEqual(dry_code, 0)
            dry_run = json.loads(dry_out.getvalue())
            self.assertEqual(dry_run["schema"], "tokenclaw.openai_cache_replay_rollout_actions_dry_run.v1")
            self.assertTrue(dry_run["dry_run"])
            self.assertTrue(dry_run["files"][0]["changed"])
            self.assertEqual(policy_path.read_text(encoding="utf-8"), before)

            apply_out = io.StringIO()
            apply_code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(signed)),
                stdout=apply_out,
            )
            self.assertEqual(apply_code, 0)
            applied = json.loads(apply_out.getvalue())
            self.assertEqual(applied["schema"], "tokenclaw.openai_cache_replay_rollout_actions_apply.v1")
            self.assertTrue(applied["wrote_policy_files"])
            self.assertEqual(applied["applied_sections"], ["cache"])
            self.assertEqual(len(list(Path(tmp).glob("cache_canary_policy.yaml.bak-*"))), 1)
            written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
            rule = written["pattern_rules"][0]
            self.assertEqual(rule["rollout"]["canary_fraction"], 0.5)
            self.assertEqual(rule["rollout"]["holdout_fraction"], 0.5)
            self.assertEqual(rule["rollout_action"]["action_type"], "widen")
            self.assertEqual(rule["rollout_action"]["target_candidate_id"], "openai-cache-replay-candidate")

    def test_openai_cache_replay_actions_fail_closed_without_writing(self):
        cases = []
        unsigned = self._bundle()
        cases.append(("unsigned", unsigned))
        expired = self._bundle(expires_at="2000-01-01T00:00:00+00:00")
        cases.append(("expired", self._signed(expired)))
        incompatible = self._bundle()
        incompatible["local_executor_compatibility"]["compatible"] = False
        cases.append(("incompatible", self._signed(incompatible)))
        raw_like = self._bundle(action_updates={"raw_request": {"prompt": "raw prompt must not be written"}})
        cases.append(("raw-like", self._signed(raw_like)))
        unknown = self._bundle(
            action_updates={"target_candidate_id": "unknown-cache-candidate"},
            proposed_updates={"rule_id": "unknown-cache-rule", "candidate_id": "unknown-cache-candidate"},
        )
        cases.append(("unknown-rule", self._signed(unknown)))

        for name, bundle in cases:
            with self.subTest(name=name):
                with TemporaryDirectory() as tmp, patch.dict(os.environ, {"TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET": "cache-replay-secret"}, clear=False):
                    policy_path = self._policy_path(tmp)
                    before = policy_path.read_text(encoding="utf-8")
                    out = io.StringIO()
                    code = cli.managed_rollout_actions_apply_cli(
                        ["--config-dir", tmp, "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=out,
                    )
                    self.assertEqual(code, 1)
                    self.assertEqual(policy_path.read_text(encoding="utf-8"), before)
                    self.assertEqual(list(Path(tmp).glob("cache_canary_policy.yaml.bak-*")), [])
                    payload = json.loads(out.getvalue())
                    self.assertFalse(payload["ok"])
                    rendered = json.dumps(payload)
                    self.assertNotIn("raw prompt must not be written", rendered)

    def test_openai_cache_replay_lifecycle_feedback_is_metadata_only(self):
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "TOKENCLAW_MANAGED_POLICY_VERIFICATION_SECRET": "cache-replay-secret",
                "TOKENCLAW_RECOMMENDATION_ENABLED": "1",
                "TOKENCLAW_RECOMMENDATION_SERVER_URL": "http://managed.test",
            },
            clear=False,
        ):
            self._policy_path(tmp)
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            signed = self._signed(self._bundle())
            out = io.StringIO()
            with patch("tokenclaw.http_client.httpx.AsyncClient", ManagedFeedbackFlushClient):
                code = cli.managed_rollout_actions_apply_cli(
                    ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                    stdin=io.StringIO(json.dumps(signed)),
                    stdout=out,
                )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent["event_type"], "dry-run")
        self.assertEqual(sent["policy_sections"], ["cache"])
        self.assertEqual(sent["metadata"]["action_type_counts"]["widen"], 1)
        self.assertFalse(sent["metadata"]["privacy"]["file_paths_included"])
        rendered = json.dumps(sent, sort_keys=True)
        self.assertNotIn("cache_canary_policy.yaml", rendered)
        self.assertNotIn("raw_request", rendered)
        self.assertNotIn("raw prompt", rendered)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "sent")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])


if __name__ == "__main__":
    unittest.main()
