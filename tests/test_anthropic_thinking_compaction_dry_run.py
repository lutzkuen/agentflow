from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentflow_proxy import cli
from agentflow_proxy.anthropic_thinking_compaction_dry_run import (
    build_anthropic_thinking_compaction_dry_run,
)
from agentflow_proxy.store import Store, stable_json


def _thinking_text(secret: str, marker: str = "alpha beta gamma delta epsilon") -> str:
    return "\n".join(
        f"private thinking {secret} /workspace/private/plan.py {marker} step-{index} observed-token-{index % 37}"
        for index in range(420)
    )


def _safe_tool_result_body(*thinking_texts: str) -> dict:
    messages = []
    for text in thinking_texts:
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": text},
                {"type": "text", "text": "public assistant continuity fallback"},
            ],
        })
        messages.append({"role": "user", "content": "continue"})
    messages.extend([
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "using the read tool"},
                {
                    "type": "tool_use",
                    "id": "raw-tool-use-id-must-not-leak",
                    "name": "Read",
                    "input": {"file_path": "/workspace/private/secret.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "raw-tool-use-id-must-not-leak",
                    "content": "raw tool payload secret must not leak",
                }
            ],
        },
    ])
    return {"model": "claude-sonnet-4-6", "messages": messages}


def _unsafe_active_thinking_body(secret: str) -> dict:
    body = _safe_tool_result_body(_thinking_text(secret), _thinking_text(secret))
    body["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    return body


def _unresolved_tool_dependency_body(secret: str) -> dict:
    thinking = _thinking_text(secret)
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking},
                    {"type": "text", "text": "assistant fallback before unresolved tool"},
                    {
                        "type": "tool_use",
                        "id": "raw-unresolved-tool-id-must-not-leak",
                        "name": "Read",
                        "input": {"file_path": "/workspace/private/unresolved.py"},
                    },
                ],
            },
            {"role": "user", "content": "not the matching tool result"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": thinking},
                    {"type": "text", "text": "newer duplicate fallback"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "different-tool-id",
                        "content": "raw delayed tool payload must not leak",
                    }
                ],
            },
        ],
    }


def _log_call(
    store: Store,
    call_id: str,
    *,
    request_json: dict | None,
    created_at: str = "2026-06-13T00:00:00+00:00",
    category: str = "tool-result",
    text_chars: int = 120_000,
    status_code: int = 200,
) -> None:
    routing = {
        "category": category,
        "workflow_phase": "tool-execution",
        "reason": "keep requested model for thinking request",
        "text_chars": text_chars,
        "has_tools": category.startswith("tool"),
    }
    store.log_call(
        id=call_id,
        created_at=created_at,
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=1000,
        input_tokens_est=text_chars // 4,
        output_tokens_est=250,
        actual_input_tokens=300,
        actual_output_tokens=250,
        cost_est_usd=0.05,
        cost_baseline_usd=0.50,
        crunch_json=stable_json({"changed": False, "anthropic_thinking_history": {"status": "ready"}}),
        routing_json=stable_json(routing),
        cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
        error=None,
        request_json=stable_json(request_json) if request_json is not None else None,
        response_json=stable_json({"text": "raw response must not leak"}),
        session_id="raw-thinking-session-id-must-not-leak",
        category=category,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=max(0, (text_chars // 4) - 400),
        retry_count=0,
        thinking_output_tokens=120,
        provider="anthropic",
        source_surface="anthropic_messages",
        endpoint="messages",
        requested_model_family="sonnet",
        routed_model_family="sonnet",
    )


class AnthropicThinkingCompactionDryRunTests(unittest.TestCase):
    def test_dry_run_plans_exact_and_near_duplicate_thinking_blocks_privately(self):
        exact_secret = "raw-exact-thinking-secret"
        near_secret = "raw-near-thinking-secret"
        exact = _thinking_text(exact_secret)
        near_one = _thinking_text(near_secret, marker="alpha beta gamma delta epsilon")
        near_two = near_one + " small newer suffix"
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "exact-duplicates",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json=_safe_tool_result_body(exact, exact),
                )
                _log_call(
                    store,
                    "near-duplicates",
                    created_at="2026-06-13T00:01:00+00:00",
                    request_json=_safe_tool_result_body(near_one, near_two),
                )
                payload = build_anthropic_thinking_compaction_dry_run(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "agentflow.anthropic_thinking_compaction_dry_run.v1")
        self.assertEqual(payload["summary"]["planned_candidate_count"], 2)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["projected_saved_usd"], 0)
        staged = payload["policy"]["staged_local_canary"]
        self.assertEqual(staged["schema"], "agentflow.anthropic_thinking_compaction_staged_local_canary.v1")
        self.assertFalse(staged["runtime_mutation_enabled"])
        self.assertEqual(staged["configured_rule_count"], 1)
        self.assertEqual(staged["rules"][0]["rule_id"], "local-repeated-context-thinking-tool-result-canary")
        self.assertEqual(staged["rules"][0]["candidate_id"], "repeated-context-thinking-tool-result-gte-128k")
        self.assertEqual(staged["rules"][0]["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertEqual(staged["rules"][0]["canary"]["fraction"], 0.0)
        self.assertEqual(staged["rules"][0]["canary"]["holdout_fraction"], 1.0)
        self.assertTrue(staged["lifecycle_metadata"]["emits_applied"])
        self.assertTrue(staged["lifecycle_metadata"]["emits_holdout"])
        self.assertEqual(staged["lifecycle_metadata"]["impact_report"], "agentflow.anthropic_thinking_compaction_impact.v1")
        duplicate_kinds = {
            plan["thinking_block"]["duplicate_kind"]
            for plan in payload["plans"]
            if plan["status"] == "planned"
        }
        self.assertEqual(duplicate_kinds, {"exact", "near"})
        for plan in payload["plans"]:
            self.assertFalse(plan["mutation"]["request_body_changed"])
            self.assertFalse(plan["fallback"]["provider_call_made"])
            self.assertFalse(plan["privacy"]["raw_thinking_text_included"])
            self.assertFalse(plan["thinking_block"]["fingerprint_included"])
            self.assertTrue(plan["source_metadata"]["available"])
            self.assertFalse(plan["source_metadata"]["fingerprints_included"])

        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            exact_secret,
            near_secret,
            "private thinking",
            "raw-tool-use-id-must-not-leak",
            "raw tool payload secret",
            "/workspace/private",
            "raw-thinking-session-id-must-not-leak",
            "raw response must not leak",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_dry_run_blocks_metadata_only_and_active_thinking_rows(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "metadata-only",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json=None,
                )
                _log_call(
                    store,
                    "active-thinking",
                    created_at="2026-06-13T00:01:00+00:00",
                    request_json=_unsafe_active_thinking_body("raw-active-thinking-secret"),
                )
                payload = build_anthropic_thinking_compaction_dry_run(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["planned_candidate_count"], 0)
        blockers = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("request-body-unavailable", blockers)
        self.assertIn("metadata-insufficient", blockers)
        self.assertIn("active-top-level-thinking-request", blockers)
        self.assertTrue(all(plan["status"] == "blocked" for plan in payload["plans"]))
        self.assertNotIn("raw-active-thinking-secret", json.dumps(payload, sort_keys=True))

    def test_holdout_eligible_rows_preserve_projected_savings_without_mutation(self):
        secret = "raw-holdout-thinking-secret"
        duplicate = _thinking_text(secret)
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "holdout-eligible",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json=_safe_tool_result_body(duplicate, duplicate),
                )
                payload = build_anthropic_thinking_compaction_dry_run(
                    store,
                    limit=10,
                    canary_fraction=0.0,
                    holdout_fraction=1.0,
                )
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["planned_candidate_count"], 0)
        self.assertEqual(payload["summary"]["holdout_candidate_count"], 1)
        self.assertEqual(payload["summary"]["eligible_candidate_count"], 1)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["holdout_projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["holdout_projected_saved_usd"], 0)
        holdout_plans = [plan for plan in payload["plans"] if plan["status"] == "holdout"]
        self.assertEqual(len(holdout_plans), 1)
        plan = holdout_plans[0]
        self.assertEqual(plan["reason"], "thinking-compaction-holdout")
        self.assertEqual(plan["no_op_reason"], "canary-holdout-forward-original")
        self.assertEqual(plan["blockers"], [])
        self.assertFalse(plan["mutation"]["request_body_changed"])
        self.assertTrue(plan["mutation"]["eligible_for_apply"])
        self.assertTrue(plan["mutation"]["would_change_request_body_if_applied"])
        self.assertEqual(plan["counts"]["saved_tokens_est"], 0)
        self.assertGreater(plan["counts"]["projected_saved_tokens_est"], 0)
        self.assertGreater(plan["counts"]["projected_saved_usd"], 0)
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn(secret, rendered)
        self.assertNotIn("private thinking", rendered)
        self.assertNotIn("raw-tool-use-id-must-not-leak", rendered)
        self.assertNotIn("/workspace/private", rendered)

    def test_dry_run_blocks_unsupported_shapes_and_unresolved_tool_dependencies_privately(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "unsupported-shape",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json={"model": "claude-sonnet-4-6", "messages": "raw prompt string must not leak"},
                )
                _log_call(
                    store,
                    "unresolved-tool-dependency",
                    created_at="2026-06-13T00:01:00+00:00",
                    request_json=_unresolved_tool_dependency_body("raw-unresolved-thinking-secret"),
                )
                payload = build_anthropic_thinking_compaction_dry_run(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["planned_candidate_count"], 0)
        blockers = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("unsupported-content-block-shape", blockers)
        self.assertIn("unresolved-tool-use-dependency", blockers)
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw prompt string must not leak",
            "raw-unresolved-thinking-secret",
            "raw-unresolved-tool-id-must-not-leak",
            "raw delayed tool payload",
            "/workspace/private/unresolved.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_dry_run_leaves_stored_provider_request_body_unchanged(self):
        body = _safe_tool_result_body(
            _thinking_text("raw-unchanged-secret"),
            _thinking_text("raw-unchanged-secret"),
        )
        original = copy.deepcopy(body)
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(store, "unchanged", request_json=body)
                before = store.conn.execute("select request_json from calls where id = ?", ("unchanged",)).fetchone()[0]
                payload = build_anthropic_thinking_compaction_dry_run(store, limit=10)
                after = store.conn.execute("select request_json from calls where id = ?", ("unchanged",)).fetchone()[0]
            finally:
                store.conn.close()

        self.assertEqual(body, original)
        self.assertEqual(after, before)
        self.assertEqual(json.loads(after), original)
        self.assertFalse(payload["summary"]["request_bodies_modified"])
        self.assertTrue(all(not plan["mutation"]["request_body_changed"] for plan in payload["plans"]))

    def test_cli_emits_anthropic_thinking_compaction_dry_run(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                secret = "raw-cli-thinking-secret"
                _log_call(
                    store,
                    "cli-thinking",
                    request_json=_safe_tool_result_body(_thinking_text(secret), _thinking_text(secret)),
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.anthropic_thinking_compaction_dry_run_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.anthropic_thinking_compaction_dry_run.v1")
        self.assertEqual(payload["summary"]["planned_candidate_count"], 1)
        self.assertNotIn("raw-cli-thinking-secret", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
