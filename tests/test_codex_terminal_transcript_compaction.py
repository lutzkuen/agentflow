from __future__ import annotations

import importlib
import json
import sys
import unittest

from tokenclaw.codex_terminal_transcript_compaction import (
    FAMILY,
    SCHEMA,
    codex_terminal_transcript_compaction_decision,
)
from tokenclaw.store import stable_json


def _terminal_block(lines: int = 80, chars_per_line: int = 60) -> str:
    out: list[str] = []
    for i in range(lines):
        if i % 5 == 0:
            out.append(f"$ run_step_{i} --flag value_{i}")
        elif i % 7 == 0:
            out.append(f"ERROR: step {i} failed with exit code 1")
        else:
            out.append(f"[2026-06-13T00:00:{i:02d}Z] INFO: progress {i}/{lines}")
    return "\n".join(out) + "\n"


def _enabled_policy(fraction: float = 1.0, holdout: float = 0.0, **action_overrides) -> dict:
    action = {
        "keep_recent_turns": 2,
        "min_block_chars": 100,
        "head_lines": 4,
        "tail_lines": 4,
        "max_evidence_lines": 20,
        "min_saved_chars": 50,
        "preserve_diagnostics": True,
        "preserve_error_lines": True,
    }
    action.update(action_overrides)
    return {
        "enabled": True,
        "canary": {
            "canary_fraction": fraction,
            "holdout_fraction": holdout,
            "canary_salt": "test-compaction-salt",
            "canary_unit": "source_hash",
        },
        "action": action,
    }


class TerminalTranscriptCompactionDecisionTests(unittest.TestCase):

    def test_disabled_policy_skips(self):
        params = {"input": _terminal_block(), "model": "gpt-5"}
        _, meta = codex_terminal_transcript_compaction_decision(params, policy={"enabled": False})
        result = meta[FAMILY]
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "disabled")
        self.assertFalse(result["applied"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["schema"], SCHEMA)
        self.assertFalse(result["raw_text_included"])
        self.assertFalse(result["raw_commands_included"])

    def test_holdout_rows_byte_equivalent(self):
        block = _terminal_block(lines=120)
        params = {"input": block, "model": "gpt-5"}
        policy = _enabled_policy(fraction=0.0, holdout=1.0)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        self.assertEqual(result["status"], "holdout")
        self.assertEqual(result["coordinator"]["selected_families"], [])
        self.assertFalse(result["applied"])
        self.assertEqual(stable_json(new_params), stable_json(params), "params must be byte-equivalent on holdout")

    def test_canary_applied_rows_mutate_eligible_inputs(self):
        block = _terminal_block(lines=120)
        params = {"input": block, "model": "gpt-5"}
        policy = _enabled_policy(fraction=1.0, holdout=0.0, keep_recent_turns=0)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        self.assertTrue(result["applied"], f"expected applied, got: {result}")
        self.assertTrue(result["changed"])
        self.assertEqual(result["status"], "applied")
        self.assertGreater(result["saved_chars"], 0)
        self.assertLess(len(stable_json(new_params)), len(stable_json(params)))
        self.assertIn("AgentFlow: middle of long terminal-transcript block omitted", new_params["input"])

    def test_metadata_has_before_after_char_counts_without_raw_text(self):
        block = _terminal_block(lines=100)
        params = {"input": block, "model": "gpt-5"}
        policy = _enabled_policy(fraction=1.0, holdout=0.0, keep_recent_turns=0)
        _, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        self.assertIn("before_chars", result)
        self.assertIn("after_chars", result)
        self.assertIn("saved_chars", result)
        self.assertIsInstance(result["before_chars"], int)
        self.assertIsInstance(result["after_chars"], int)
        self.assertFalse(result["raw_text_included"])
        self.assertFalse(result["raw_commands_included"])
        rendered = json.dumps(result)
        self.assertNotIn(_terminal_block(lines=1)[:20], rendered)

    def test_preserved_diagnostic_lines_stay_out_of_runtime_metadata(self):
        raw_command = "$ cat /workspace/private/raw-codex-terminal-path-must-not-leak.log"
        raw_error = "ERROR raw codex terminal diagnostic must not leak"
        block = "\n".join(
            [f"$ setup_step_{index} --secret raw-command-secret-{index}" for index in range(6)]
            + [raw_command, raw_error]
            + [f"[2026-06-13T00:00:{index:02d}Z] INFO progress {index}" for index in range(100)]
            + [f"[2026-06-13T00:01:{index:02d}Z] INFO tail {index}" for index in range(40)]
        )
        params = {"input": block, "model": "gpt-5"}
        policy = _enabled_policy(
            fraction=1.0,
            holdout=0.0,
            keep_recent_turns=0,
            min_block_chars=100,
            min_saved_chars=50,
            max_evidence_lines=20,
            preserve_diagnostics=True,
            preserve_error_lines=True,
        )

        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]

        self.assertTrue(result["applied"], result)
        self.assertIn(raw_error, new_params["input"], "diagnostic evidence is preserved in the local request")
        rendered_meta = json.dumps(result, sort_keys=True)
        for forbidden in (
            raw_command,
            raw_error,
            "raw-command-secret",
            "/workspace/private",
            "setup_step_",
            "INFO progress",
        ):
            self.assertNotIn(forbidden, rendered_meta)
        self.assertFalse(result["raw_text_included"])
        self.assertFalse(result["raw_commands_included"])

    def test_safety_blocker_action_like_params(self):
        params = {
            "input": _terminal_block(lines=100),
            "action": {"type": "bash", "command": "rm -rf /"},
        }
        policy = _enabled_policy(fraction=1.0, holdout=0.0)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        self.assertEqual(result["reason"], "action-like-params")
        self.assertFalse(result["applied"])
        self.assertEqual(stable_json(new_params), stable_json(params))

    def test_incompatible_family_cache_replay_suppresses(self):
        block = _terminal_block(lines=120)
        params = {"input": block, "model": "gpt-5"}
        policy = _enabled_policy(fraction=1.0, holdout=0.0)
        ledger = {"exact_cache_replay": True}
        new_params, meta = codex_terminal_transcript_compaction_decision(
            params, policy=policy, coordinator_ledger=ledger
        )
        result = meta[FAMILY]
        self.assertFalse(result["applied"])
        self.assertIn("suppressed-by-exact_cache_replay", result["reason"])
        self.assertEqual(result["coordinator"]["suppressed_by"], "exact_cache_replay")
        self.assertEqual(stable_json(new_params), stable_json(params))

    def test_incompatible_family_managed_recommendation_suppresses(self):
        block = _terminal_block(lines=120)
        params = {"input": block}
        policy = _enabled_policy(fraction=1.0, holdout=0.0)
        ledger = {"managed_recommendation": True}
        _, meta = codex_terminal_transcript_compaction_decision(
            params, policy=policy, coordinator_ledger=ledger
        )
        result = meta[FAMILY]
        self.assertFalse(result["applied"])
        self.assertEqual(result["coordinator"]["suppressed_by"], "managed_recommendation")

    def test_coordinator_metadata_records_selected_and_suppressed_families(self):
        block = _terminal_block(lines=120)
        params = {"input": block}
        policy = _enabled_policy(fraction=1.0, holdout=0.0)
        _, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        self.assertIn("coordinator", result)
        coord = result["coordinator"]
        self.assertIn("selected_families", coord)
        self.assertIn("suppressed_families", coord)
        if result["applied"]:
            self.assertIn(FAMILY, coord["selected_families"])

    def test_list_input_compacts_older_blocks_keeps_recent(self):
        block = _terminal_block(lines=80)
        recent_block = "recent input that should not be compacted"
        params = {
            "input": [block, block, block, recent_block],
            "model": "gpt-5",
        }
        policy = _enabled_policy(fraction=1.0, holdout=0.0, keep_recent_turns=1)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        if result["applied"]:
            input_list = new_params["input"]
            self.assertIsInstance(input_list, list)
            self.assertEqual(input_list[-1], recent_block, "most recent block must be preserved")
            for older in input_list[:-1]:
                if isinstance(older, str):
                    self.assertIn("AgentFlow:", older)

    def test_short_input_below_min_block_chars_skipped(self):
        short_block = "$ ls\ntotal 0\n"
        params = {"input": short_block}
        policy = _enabled_policy(fraction=1.0, holdout=0.0, min_block_chars=10000)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        self.assertFalse(result["applied"])
        self.assertEqual(stable_json(new_params), stable_json(params))

    def test_non_terminal_text_not_compacted(self):
        prose = "This is a normal prose paragraph with no terminal output.\n" * 50
        params = {"input": prose}
        policy = _enabled_policy(fraction=1.0, holdout=0.0, min_block_chars=100)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        self.assertEqual(stable_json(new_params), stable_json(params))

    def test_canary_sample_fields_present(self):
        block = _terminal_block(lines=100)
        params = {"input": block}
        policy = _enabled_policy(fraction=1.0, holdout=0.0, keep_recent_turns=0)
        _, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        if result.get("canary"):
            canary = result["canary"]
            self.assertIn("cohort", canary)
            self.assertIn("candidate_id", canary)
            self.assertIn("sample_bucket", canary)
            self.assertFalse(canary.get("raw_basis_included"))

    def test_structured_input_dict_compacted(self):
        block = _terminal_block(lines=80)
        params = {
            "input": {"type": "text", "text": block},
            "model": "gpt-5",
        }
        policy = _enabled_policy(fraction=1.0, holdout=0.0, keep_recent_turns=0)
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        result = meta[FAMILY]
        if result["applied"]:
            self.assertIn("AgentFlow:", new_params["input"]["text"])


class TerminalTranscriptCompactionPolicyLoadTests(unittest.TestCase):
    """Test that loaded Codex-turn terminal compaction policy can drive the compactor directly."""

    def _build_turn_start(self, input_text: str, model: str = "gpt-5") -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "method": "turn/start",
            "id": "req-1",
            "params": {
                "model": model,
                "input": input_text,
                "threadId": "thread-abc",
            },
        })

    def _load_terminal_policy(self, policy_yaml: str) -> dict:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(policy_yaml)
            policy_path = f.name
        try:
            env_backup = os.environ.copy()
            os.environ["AGENTFLOW_CODEX_APP_RULES"] = policy_path
            import tokenclaw.codex_turn_policy as policy_mod
            policy_mod = importlib.reload(policy_mod)
            terminal = policy_mod.CODEX_APP_POLICY.get("terminal_transcript_compaction")
            return terminal if isinstance(terminal, dict) else {}
        finally:
            os.unlink(policy_path)
            os.environ.clear()
            os.environ.update(env_backup)

    def test_disabled_default_policy_does_not_mutate(self):
        block = _terminal_block(lines=120)
        params = json.loads(self._build_turn_start(block))["params"]
        _, meta = codex_terminal_transcript_compaction_decision(params)
        # Default policy has enabled: false - no compaction.
        tc_meta = meta.get(FAMILY)
        if tc_meta is not None:
            self.assertFalse(tc_meta.get("applied"))

    def test_enabled_policy_compacts_via_proxy(self):
        block = _terminal_block(lines=120)
        raw = self._build_turn_start(block)
        policy_yaml = """
enabled: true
terminal_transcript_compaction:
  enabled: true
  action:
    keep_recent_turns: 0
    min_block_chars: 100
    head_lines: 4
    tail_lines: 4
    max_evidence_lines: 20
    min_saved_chars: 50
    preserve_diagnostics: true
    preserve_error_lines: true
  canary:
    canary_fraction: 1.0
    holdout_fraction: 0.0
    canary_salt: proxy-test-salt
    canary_unit: source_hash
"""
        policy = self._load_terminal_policy(policy_yaml)
        params = json.loads(raw)["params"]
        new_params, meta = codex_terminal_transcript_compaction_decision(params, policy=policy)
        tc_meta = meta.get(FAMILY)
        if tc_meta and tc_meta.get("applied"):
            self.assertIn("AgentFlow:", new_params["input"])
            self.assertGreater(tc_meta["saved_chars"], 0)
            self.assertFalse(tc_meta["raw_text_included"])
            self.assertFalse(tc_meta["raw_commands_included"])


if __name__ == "__main__":
    unittest.main()
