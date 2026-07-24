"""Tests for the per-pocket downroute dial (CLAUDE.md "Calibrated local dials").

Covers the pure logic in tokenclaw/downroute.py (classifier, pocket map, seeded
coin-flip, Wilson-bounded AIMD controller, tool-target extraction), the store's
per-pocket state round-trip and deferred harm-verdict finalize pass, the proxy
integration seam (_maybe_apply_downroute: armed-off no-op vs forced-f downroute),
and the boundary firewall that keeps a decided downroute off the server-bound
feedback streams.

unittest.TestCase style to match the repo's existing suite; runnable under both
`python -m unittest` and `python -m pytest`.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from tokenclaw import anthropic_proxy, downroute, openai_proxy
from tokenclaw.cli_commands import onboarding
from tokenclaw.downroute import DownrouteConfig
from tokenclaw.provider_context import ProviderContext
from tokenclaw.routing_experiments import _suggest_adjacent_routed_model
from tokenclaw.store import Store


# --- shared helpers ---------------------------------------------------------

def _assistant_tool_use(*names: str) -> dict:
    """One assistant message issuing tool_use blocks with the given names."""
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "name": n, "input": {"file_path": f"/repo/{n}.py"}}
            for n in names
        ],
    }


def _body(*, last_tools: tuple[str, ...], model: str = "claude-opus-4-5") -> dict:
    """A request body whose most-recent assistant turn used `last_tools`."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_use(*last_tools),
            {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
        ],
    }


def _response_reading(path: str) -> str:
    """A stored response_json (string) whose tool_use read `path`."""
    return json.dumps(
        {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": path}}]}
    )


def _dummy_context(store: Store) -> ProviderContext:
    async def _noop(*_a, **_k):
        return None

    return ProviderContext(
        provider="anthropic",
        anthropic_upstream="http://127.0.0.1:1",
        openai_upstream="http://127.0.0.1:2",
        default_upstream="http://127.0.0.1:1",
        openai_auth_mode="client",
        openai_model_list=(),
        store=store,
        limiter=None,
        log_bodies=False,
        http_timeout=30.0,
        anthropic_messages_handler=_noop,
        openai_optimized_handler=_noop,
        openai_passthrough_handler=_noop,
        openai_responses_websocket_handler=_noop,
    )


# --- eligibility classifier -------------------------------------------------

class TestEligibilityClassifier(unittest.TestCase):
    def setUp(self):
        self.cfg = DownrouteConfig()

    def test_recent_tool_use_names_takes_most_recent_assistant_turn(self):
        body = {
            "messages": [
                _assistant_tool_use("Read"),
                {"role": "user", "content": "then"},
                _assistant_tool_use("Grep", "Glob"),
            ]
        }
        self.assertEqual(downroute.recent_tool_use_names(body), ["grep", "glob"])

    def test_recent_tool_use_names_empty_when_no_tool_use(self):
        body = {"messages": [{"role": "assistant", "content": "just text"}]}
        self.assertEqual(downroute.recent_tool_use_names(body), [])

    def test_read_only_tool_heavy_is_eligible(self):
        elig = downroute.classify_eligibility(_body(last_tools=("Read", "Grep")), "tool-heavy", self.cfg)
        self.assertTrue(elig.eligible)
        self.assertEqual(elig.reason, "read-only-tool-heavy")
        self.assertEqual(elig.tool_names, ("read", "grep"))

    def test_mutating_tool_is_ineligible(self):
        elig = downroute.classify_eligibility(_body(last_tools=("Read", "Edit")), "tool-heavy", self.cfg)
        self.assertFalse(elig.eligible)
        self.assertTrue(elig.reason.startswith("mutating-or-unknown:"))
        self.assertIn("edit", elig.reason)

    def test_unknown_mcp_tool_is_ineligible(self):
        elig = downroute.classify_eligibility(_body(last_tools=("SomeMcpTool",)), "tool-heavy", self.cfg)
        self.assertFalse(elig.eligible)
        self.assertTrue(elig.reason.startswith("mutating-or-unknown:"))

    def test_wrong_category_is_ineligible(self):
        elig = downroute.classify_eligibility(_body(last_tools=("Read",)), "chat", self.cfg)
        self.assertFalse(elig.eligible)
        self.assertEqual(elig.reason, "category:chat")

    def test_no_recent_tool_use_is_ineligible(self):
        body = {"model": "claude-opus-4-5", "messages": [{"role": "user", "content": "hi"}]}
        elig = downroute.classify_eligibility(body, "tool-heavy", self.cfg)
        self.assertFalse(elig.eligible)
        self.assertEqual(elig.reason, "no-recent-tool-use")


# --- pocket resolution ------------------------------------------------------

class TestPocketResolution(unittest.TestCase):
    def test_pocket_for_known_families(self):
        self.assertEqual(downroute.pocket_for("claude-opus-4-5"), ("opus", "sonnet"))
        self.assertEqual(downroute.pocket_for("claude-sonnet-5"), ("sonnet", "haiku"))

    def test_pocket_for_bottom_tier_is_none(self):
        self.assertIsNone(downroute.pocket_for("claude-haiku-4-5-20251001"))

    def test_pocket_for_unknown_model_is_none(self):
        self.assertIsNone(downroute.pocket_for("gpt-5-codex"))
        self.assertIsNone(downroute.pocket_for(None))

    def test_pocket_key_format(self):
        self.assertEqual(downroute.pocket_key("opus", "sonnet"), "opus->sonnet")

    def test_resolve_target_model_via_injected_tier_map(self):
        tier_map = {"sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5-20251001"}
        self.assertEqual(downroute.resolve_target_model("sonnet", tier_map), "claude-sonnet-5")
        self.assertIsNone(downroute.resolve_target_model("opus", tier_map))

    def test_cli_resolve_pocket_from_bare_family(self):
        self.assertEqual(onboarding._resolve_downroute_pocket("opus"), ("opus->sonnet", "opus", "sonnet"))

    def test_cli_resolve_pocket_from_full_key(self):
        self.assertEqual(
            onboarding._resolve_downroute_pocket("sonnet->haiku"), ("sonnet->haiku", "sonnet", "haiku")
        )

    def test_cli_resolve_pocket_rejects_noncanonical(self):
        # opus->haiku is not a canonical single-step pocket
        self.assertIsNone(onboarding._resolve_downroute_pocket("opus->haiku"))
        self.assertIsNone(onboarding._resolve_downroute_pocket("haiku"))
        self.assertIsNone(onboarding._resolve_downroute_pocket(""))


# --- seeded coin-flip -------------------------------------------------------

class TestCoinFlip(unittest.TestCase):
    def test_f_zero_never_downroutes(self):
        self.assertFalse(downroute.decide_downroute(session_id="s", call_id="c", f=0.0))

    def test_f_one_always_downroutes(self):
        self.assertTrue(downroute.decide_downroute(session_id="s", call_id="c", f=1.0))

    def test_deterministic_for_same_ids(self):
        a = downroute.decide_downroute(session_id="s1", call_id="c1", f=0.5)
        b = downroute.decide_downroute(session_id="s1", call_id="c1", f=0.5)
        self.assertEqual(a, b)

    def test_aggregate_rate_matches_f(self):
        n, f = 5000, 0.2
        hits = sum(
            downroute.decide_downroute(session_id="sess", call_id=f"call-{i}", f=f)
            for i in range(n)
        )
        rate = hits / n
        self.assertGreater(rate, f - 0.03)
        self.assertLess(rate, f + 0.03)


# --- controller law ---------------------------------------------------------

class TestController(unittest.TestCase):
    def test_wilson_bounds_zero_n_is_maximal_uncertainty(self):
        self.assertEqual(downroute.wilson_bounds(0, 0, 1.96), (0.0, 1.0))

    def test_disabled_controller_always_holds(self):
        cfg = DownrouteConfig(controller_enabled=False)
        d = downroute.controller_step(f=0.05, window_applied=100, window_harm=0, cfg=cfg)
        self.assertEqual(d.action, "hold")
        self.assertFalse(d.reset_window)
        self.assertEqual(d.new_f, 0.05)

    def test_advance_on_clean_window(self):
        cfg = DownrouteConfig(controller_enabled=True)
        d = downroute.controller_step(f=0.05, window_applied=100, window_harm=0, cfg=cfg)
        self.assertEqual(d.action, "advance")
        self.assertTrue(d.reset_window)
        self.assertAlmostEqual(d.new_f, 0.07, places=4)

    def test_retreat_on_harmful_window(self):
        cfg = DownrouteConfig(controller_enabled=True)
        d = downroute.controller_step(f=0.10, window_applied=20, window_harm=10, cfg=cfg)
        self.assertEqual(d.action, "retreat")
        self.assertTrue(d.reset_window)
        self.assertAlmostEqual(d.new_f, 0.05, places=4)

    def test_hold_below_sample_thresholds(self):
        cfg = DownrouteConfig(controller_enabled=True)
        d = downroute.controller_step(f=0.05, window_applied=5, window_harm=0, cfg=cfg)
        self.assertEqual(d.action, "hold")
        self.assertFalse(d.reset_window)

    def test_advance_clamped_at_f_max(self):
        cfg = DownrouteConfig(controller_enabled=True, f_max=0.06)
        d = downroute.controller_step(f=0.05, window_applied=100, window_harm=0, cfg=cfg)
        self.assertEqual(d.action, "advance")
        self.assertAlmostEqual(d.new_f, 0.06, places=4)

    def test_retreat_clamped_at_f_min(self):
        cfg = DownrouteConfig(controller_enabled=True, f_min=0.04)
        d = downroute.controller_step(f=0.05, window_applied=20, window_harm=20, cfg=cfg)
        self.assertEqual(d.action, "retreat")
        self.assertAlmostEqual(d.new_f, 0.04, places=4)


# --- tool-target extraction -------------------------------------------------

class TestToolTargets(unittest.TestCase):
    def test_extract_targets_from_full_message(self):
        resp = {
            "content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": "/a.py"}},
                {"type": "tool_use", "name": "Grep", "input": {"pattern": "foo"}},
                {"type": "text", "text": "ignored"},
            ]
        }
        self.assertEqual(
            downroute.extract_tool_targets(resp), {"file_path=/a.py", "pattern=foo"}
        )

    def test_extract_targets_from_bare_list(self):
        resp = [{"type": "tool_use", "name": "Read", "input": {"path": "/b"}}]
        self.assertEqual(downroute.extract_tool_targets(resp), {"path=/b"})

    def test_extract_targets_empty_when_no_tool_use(self):
        self.assertEqual(downroute.extract_tool_targets({"content": [{"type": "text", "text": "x"}]}), set())

    def test_error_tool_result_detected(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "tool_result", "is_error": True, "content": "boom"}]}
            ]
        }
        self.assertTrue(downroute.response_has_error_tool_result(body))

    def test_no_error_tool_result(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "tool_result", "content": "fine"}]}
            ]
        }
        self.assertFalse(downroute.response_has_error_tool_result(body))


# --- store round-trip -------------------------------------------------------

class TestStoreRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_unseen_pocket_returns_default(self):
        self.assertEqual(self.store.get_downroute_pocket_f("opus->sonnet", default=0.0), 0.0)
        self.assertEqual(self.store.get_downroute_pocket_f("opus->sonnet", default=0.33), 0.33)

    def test_arm_sets_f_and_armed_at(self):
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=0.05, requested_family="opus",
            target_family="sonnet", action="arm", reset_window=True,
        )
        self.assertAlmostEqual(self.store.get_downroute_pocket_f("opus->sonnet"), 0.05)
        row = self.store._downroute_pocket_row("opus->sonnet")
        self.assertTrue(row.get("armed_at"))
        self.assertEqual(row.get("last_action"), "arm")

    def test_disarm_sets_f_zero(self):
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=0.05, requested_family="opus", target_family="sonnet", action="arm",
        )
        self.store.set_downroute_pocket_f(pocket="opus->sonnet", f=0.0, action="disarm", reset_window=True)
        self.assertEqual(self.store.get_downroute_pocket_f("opus->sonnet"), 0.0)

    def test_note_eligible_creates_then_increments(self):
        self.store.note_downroute_pocket_eligible(
            pocket="opus->sonnet", requested_family="opus", target_family="sonnet", default_f=0.0,
        )
        self.store.note_downroute_pocket_eligible(
            pocket="opus->sonnet", requested_family="opus", target_family="sonnet", default_f=0.0,
        )
        row = self.store._downroute_pocket_row("opus->sonnet")
        self.assertEqual(int(row["eligible_count"]), 2)
        # note_eligible must not arm the pocket
        self.assertEqual(float(row["f"]), 0.0)

    def test_list_pockets_ordered(self):
        self.store.set_downroute_pocket_f(
            pocket="sonnet->haiku", f=0.05, requested_family="sonnet", target_family="haiku", action="arm",
        )
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=0.05, requested_family="opus", target_family="sonnet", action="arm",
        )
        keys = [r["pocket"] for r in self.store.list_downroute_pockets()]
        self.assertEqual(keys, sorted(keys))
        self.assertIn("opus->sonnet", keys)
        self.assertIn("sonnet->haiku", keys)


# --- deferred harm-verdict finalize pass ------------------------------------

class TestFinalizeOutcomes(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))
        self.now = datetime(2026, 7, 1, 0, 10, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self._tmp.cleanup()

    def _log_pending_downroute(self, *, call_id, session_id, created_at, target_read):
        self.store.log_call(
            id=call_id,
            path="/v1/messages",
            session_id=session_id,
            requested_model="claude-opus-4-5",
            routed_model="claude-sonnet-5",
            stream=False,
            status_code=200,
            retry_count=0,
            created_at=created_at.isoformat(),
            downroute_verdict="pending",
            response_json=_response_reading(target_read),
        )

    def _log_frontier_follower(self, *, call_id, session_id, created_at, target_read):
        self.store.log_call(
            id=call_id,
            path="/v1/messages",
            session_id=session_id,
            requested_model="claude-opus-4-5",
            routed_model="claude-opus-4-5",
            stream=False,
            status_code=200,
            retry_count=0,
            created_at=created_at.isoformat(),
            response_json=_response_reading(target_read),
        )

    def test_repair_harm_and_clean_verdicts(self):
        a_at = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        b_at = datetime(2026, 7, 1, 0, 0, 10, tzinfo=timezone.utc)
        # Session s1: downrouted read of /repo/x.py, then a frontier turn re-reads it -> repair harm.
        self._log_pending_downroute(call_id="a", session_id="s1", created_at=a_at, target_read="/repo/x.py")
        self._log_frontier_follower(call_id="b", session_id="s1", created_at=b_at, target_read="/repo/x.py")
        # Session s2: downrouted read of /repo/y.py, no follower -> clean.
        self._log_pending_downroute(call_id="c", session_id="s2", created_at=a_at, target_read="/repo/y.py")

        result = self.store.finalize_downroute_outcomes(now=self.now.isoformat(), config=DownrouteConfig())

        self.assertEqual(result["schema"], "tokenclaw.downroute_outcome_finalization.v1")
        self.assertEqual(result["harm_count"], 1)
        self.assertEqual(result["clean_count"], 1)

        verdicts = {
            r["id"]: r["downroute_verdict"]
            for r in self.store.conn.execute(
                "select id, downroute_verdict from calls where id in ('a','b','c')"
            ).fetchall()
        }
        self.assertEqual(verdicts["a"], "harm")
        self.assertEqual(verdicts["c"], "clean")
        self.assertIsNone(verdicts["b"])  # frontier follower was never a downroute candidate

        pocket = self.store._downroute_pocket_row("opus->sonnet")
        self.assertEqual(int(pocket["applied_count"]), 2)
        self.assertEqual(int(pocket["clean_count"]), 1)
        self.assertEqual(int(pocket["harm_count"]), 1)
        self.assertEqual(int(pocket["harm_repair_count"]), 1)
        self.assertEqual(int(pocket["harm_error_count"]), 0)

    def test_error_harm_from_retry(self):
        a_at = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.store.log_call(
            id="e", path="/v1/messages", session_id="s3",
            requested_model="claude-opus-4-5", routed_model="claude-sonnet-5",
            stream=False, status_code=200, retry_count=2,
            created_at=a_at.isoformat(), downroute_verdict="pending",
            response_json=_response_reading("/repo/z.py"),
        )
        result = self.store.finalize_downroute_outcomes(now=self.now.isoformat(), config=DownrouteConfig())
        self.assertEqual(result["harm_count"], 1)
        pocket = self.store._downroute_pocket_row("opus->sonnet")
        self.assertEqual(int(pocket["harm_error_count"]), 1)
        self.assertEqual(int(pocket["harm_repair_count"]), 0)

    def test_pending_within_ttl_is_not_picked_up(self):
        # created_at is inside the TTL window relative to now -> not yet judged.
        recent = datetime(2026, 7, 1, 0, 9, 30, tzinfo=timezone.utc)
        self._log_pending_downroute(call_id="fresh", session_id="s4", created_at=recent, target_read="/repo/w.py")
        result = self.store.finalize_downroute_outcomes(now=self.now.isoformat(), config=DownrouteConfig())
        self.assertEqual(result["candidate_count"], 0)
        row = self.store.conn.execute(
            "select downroute_verdict from calls where id = 'fresh'"
        ).fetchone()
        self.assertEqual(row["downroute_verdict"], "pending")


# --- proxy integration seam -------------------------------------------------

class TestProxyIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))
        anthropic_proxy._DOWNROUTE_F_CACHE.clear()
        self.cfg = DownrouteConfig()

    def tearDown(self):
        anthropic_proxy._DOWNROUTE_F_CACHE.clear()
        self._tmp.cleanup()

    def _apply(self, *, crunched, routing_meta, sampled_shadow=False, category="tool-heavy"):
        anthropic_proxy._maybe_apply_downroute(
            store_obj=self.store,
            raw_body=_body(last_tools=("Read", "Grep")),
            crunched=crunched,
            routing_meta=routing_meta,
            category=category,
            session_id="sess",
            call_id="call-1",
            resolved_requested_model="claude-opus-4-5",
            sampled_shadow_pass_through=sampled_shadow,
            input_tokens=1000,
            cfg=self.cfg,
        )

    def test_armed_off_is_noop(self):
        crunched = {"model": "claude-opus-4-5"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta)
        # model untouched; recorded as eligible-but-unarmed, not decided
        self.assertEqual(crunched["model"], "claude-opus-4-5")
        self.assertEqual(routing_meta["downroute"]["decided"], False)
        self.assertEqual(routing_meta["downroute"]["reason"], "pocket-unarmed")

    def test_forced_f_downroutes_to_target(self):
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=1.0, requested_family="opus",
            target_family="sonnet", action="arm", reset_window=True,
        )
        anthropic_proxy._DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "claude-opus-4-5"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta)
        self.assertEqual(crunched["model"], "claude-sonnet-5")
        self.assertEqual(routing_meta["routed_model"], "claude-sonnet-5")
        self.assertEqual(routing_meta["downroute"]["decided"], True)
        self.assertEqual(routing_meta["downroute"]["target_model"], "claude-sonnet-5")
        self.assertEqual(routing_meta["downroute"]["input_tokens"], 1000)

    def test_sampled_shadow_is_skipped(self):
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=1.0, requested_family="opus", target_family="sonnet", action="arm",
        )
        anthropic_proxy._DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "claude-opus-4-5"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta, sampled_shadow=True)
        self.assertEqual(crunched["model"], "claude-opus-4-5")
        self.assertNotIn("downroute", routing_meta)

    def test_non_passthrough_is_skipped(self):
        # Something upstream already moved the model off the requested one.
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=1.0, requested_family="opus", target_family="sonnet", action="arm",
        )
        anthropic_proxy._DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "claude-sonnet-5"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta)
        self.assertEqual(crunched["model"], "claude-sonnet-5")
        self.assertNotIn("downroute", routing_meta)

    def test_ineligible_turn_is_skipped(self):
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=1.0, requested_family="opus", target_family="sonnet", action="arm",
        )
        anthropic_proxy._DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "claude-opus-4-5"}
        routing_meta = {}
        # category not eligible -> classifier rejects before the dial
        self._apply(crunched=crunched, routing_meta=routing_meta, category="chat")
        self.assertEqual(crunched["model"], "claude-opus-4-5")
        self.assertNotIn("downroute", routing_meta)


# --- boundary firewall ------------------------------------------------------

class TestFirewall(unittest.TestCase):
    """A decided downroute must suppress every server-bound feedback stream and
    only persist the local routing_meta (CLAUDE.md "local applies / server learns")."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))
        self.ctx = _dummy_context(self.store)
        at = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.store.log_call(
            id="dr-fw", path="/v1/messages", session_id="s1",
            requested_model="claude-opus-4-5", routed_model="claude-sonnet-5",
            stream=False, status_code=200, retry_count=0, created_at=at.isoformat(),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, routing_meta):
        asyncio.run(
            anthropic_proxy._record_managed_outcome_feedback(
                context=self.ctx, call_id="dr-fw", path="/v1/messages",
                requested_model="claude-opus-4-5", routed_model="claude-sonnet-5",
                status_code=200, latency_ms=10, retry_count=0,
                input_tokens_est=100, output_tokens_est=10,
                actual_input_tokens=None, actual_output_tokens=None,
                cache_creation_input_tokens=None, cache_read_input_tokens=None,
                thinking_output_tokens=None, cost_est_usd=None, cost_baseline_usd=None,
                cache_meta={}, crunch_meta={}, routing_meta=routing_meta,
                category="tool-heavy", stream=False, session_id="s1", error=None,
            )
        )

    def test_decided_downroute_short_circuits_before_server_streams(self):
        routing_meta = {
            "routed_model": "claude-sonnet-5",
            "downroute": {"pocket": "opus->sonnet", "decided": True, "target_model": "claude-sonnet-5"},
        }
        # If the guard is NOT taken, the first server-bound builder runs and raises.
        with patch.object(
            anthropic_proxy, "build_cache_replay_lifecycle_feedback",
            side_effect=AssertionError("server-bound path must not run for a decided downroute"),
        ):
            self._run(routing_meta)  # must not raise
        # The local routing_meta was persisted, stamping the call 'pending' for finalize.
        row = self.store.conn.execute(
            "select downroute_verdict from calls where id = 'dr-fw'"
        ).fetchone()
        self.assertEqual(row["downroute_verdict"], "pending")

    def test_non_decided_call_reaches_server_path(self):
        routing_meta = {"routed_model": "claude-opus-4-5"}  # no decided downroute
        sentinel = RuntimeError("reached server path")
        with patch.object(
            anthropic_proxy, "build_cache_replay_lifecycle_feedback", side_effect=sentinel
        ):
            with self.assertRaises(RuntimeError):
                self._run(routing_meta)


# --- OpenAI: tier / family / pockets ----------------------------------------

class TestOpenAITierFamily(unittest.TestCase):
    """cache._model_family collapses every gpt-5.x to a single "gpt-5"; the
    downroute path uses _downroute_family (= _openai_tier or _model_family) so
    5.6 tiers become DISTINCT pockets while the Anthropic path stays identical."""

    def test_openai_tier_resolution(self):
        self.assertEqual(downroute._openai_tier("gpt-5.6-sol"), "sol")
        self.assertEqual(downroute._openai_tier("gpt-5.6-terra"), "terra")
        self.assertEqual(downroute._openai_tier("gpt-5.6-luna"), "luna")
        self.assertEqual(downroute._openai_tier("gpt-5-mini"), "mini")
        self.assertEqual(downroute._openai_tier("gpt-5-nano"), "mini")
        # bare current-gen flagship maps to sol (adjacent-down is terra)
        self.assertEqual(downroute._openai_tier("gpt-5.6"), "sol")

    def test_openai_tier_none_for_non_openai_and_codex(self):
        self.assertIsNone(downroute._openai_tier("claude-opus-4-5"))
        self.assertIsNone(downroute._openai_tier("gpt-5-codex"))
        self.assertIsNone(downroute._openai_tier("gpt-5.3-codex"))
        self.assertIsNone(downroute._openai_tier(None))

    def test_downroute_family_identical_to_model_family_for_anthropic(self):
        # The committed Anthropic path must be byte-identical: for every Anthropic
        # model _openai_tier returns None so _downroute_family falls through.
        from tokenclaw.cache import _model_family

        for m in ("claude-opus-4-5", "claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-fable-5"):
            self.assertEqual(downroute._downroute_family(m), _model_family(m))

    def test_downroute_family_distinguishes_openai_tiers(self):
        self.assertEqual(downroute._downroute_family("gpt-5.6-terra"), "terra")
        self.assertEqual(downroute._downroute_family("gpt-5.6-luna"), "luna")
        self.assertNotEqual(
            downroute._downroute_family("gpt-5.6-terra"),
            downroute._downroute_family("gpt-5.6-luna"),
        )


class TestOpenAIPockets(unittest.TestCase):
    def test_openai_pocket_chain(self):
        self.assertEqual(downroute.pocket_for("gpt-5.6-sol"), ("sol", "terra"))
        self.assertEqual(downroute.pocket_for("gpt-5.6-terra"), ("terra", "luna"))
        self.assertEqual(downroute.pocket_for("gpt-5.6-luna"), ("luna", "mini"))

    def test_openai_floor_and_codex_have_no_pocket(self):
        # gpt-5-mini is the deliberate rock bottom; Codex is an action lane.
        self.assertIsNone(downroute.pocket_for("gpt-5-mini"))
        self.assertIsNone(downroute.pocket_for("gpt-5-codex"))

    def test_openai_tier_map_agrees_with_ladder_source_of_truth(self):
        """The seam's tier->model map must land each pocket on exactly the model
        routing_experiments._suggest_adjacent_routed_model would pick. That
        function is the single source of truth for the OpenAI ladder; a silent
        divergence here would downroute to a rung the rest of the system doesn't
        recognize."""
        tier_map = openai_proxy._OPENAI_DOWNROUTE_TIER_MAP
        for requested in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            req_family, target_family = downroute.pocket_for(requested)
            seam_target = downroute.resolve_target_model(target_family, tier_map)
            self.assertEqual(
                seam_target, _suggest_adjacent_routed_model(requested),
                f"pocket {req_family}->{target_family} diverged from ladder",
            )


# --- OpenAI: tool-name extraction across request shapes ---------------------

class TestOpenAIToolNames(unittest.TestCase):
    def test_chat_completions_tool_calls(self):
        body = {
            "model": "gpt-5.6-terra",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "web_search", "arguments": "{}"}},
                        {"type": "function", "function": {"name": "file_search", "arguments": "{}"}},
                    ],
                },
            ],
        }
        self.assertEqual(downroute.recent_tool_use_names(body), ["web_search", "file_search"])

    def test_responses_function_call_and_builtin_normalization(self):
        # /v1/responses flat input list: function_call names + built-in *_call
        # item types normalized (web_search_call -> web_search).
        body = {
            "model": "gpt-5.6-terra",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "web_search_call", "id": "ws1"},
                {"type": "function_call", "name": "read_file", "arguments": "{}"},
            ],
        }
        self.assertEqual(downroute.recent_tool_use_names(body), ["web_search", "read_file"])

    def test_responses_stops_at_most_recent_user_message(self):
        body = {
            "model": "gpt-5.6-terra",
            "input": [
                {"type": "function_call", "name": "old_call", "arguments": "{}"},
                {"type": "message", "role": "user", "content": "new turn"},
                {"type": "file_search_call", "id": "fs1"},
            ],
        }
        # only this turn's activity after the last user message counts
        self.assertEqual(downroute.recent_tool_use_names(body), ["file_search"])

    def test_responses_builtin_readonly_turn_is_eligible(self):
        body = {
            "model": "gpt-5.6-terra",
            "input": [
                {"type": "message", "role": "user", "content": "search the docs"},
                {"type": "web_search_call", "id": "ws1"},
                {"type": "file_search_call", "id": "fs1"},
            ],
        }
        elig = downroute.classify_eligibility(body, "tool-heavy", DownrouteConfig())
        self.assertTrue(elig.eligible)
        self.assertEqual(elig.tool_names, ("web_search", "file_search"))

    def test_responses_custom_function_tool_fails_closed(self):
        # The user's real generic_openai traffic issues custom function tools whose
        # names are not on the read-only allow-list. Classification must fail closed.
        body = {
            "model": "gpt-5.6-terra",
            "input": [
                {"type": "message", "role": "user", "content": "read the file"},
                {"type": "function_call", "name": "read_file", "arguments": "{}"},
            ],
        }
        elig = downroute.classify_eligibility(body, "tool-heavy", DownrouteConfig())
        self.assertFalse(elig.eligible)
        self.assertTrue(elig.reason.startswith("mutating-or-unknown:"))
        self.assertIn("read_file", elig.reason)


# --- OpenAI: tool-target extraction (harm stitching) ------------------------

class TestOpenAIToolTargets(unittest.TestCase):
    def test_targets_from_responses_output_items(self):
        resp = {
            "output": [
                {"type": "function_call", "name": "read_file", "arguments": '{"file_path": "/a.py"}'},
                {"type": "web_search_call", "id": "ws1"},  # no target args -> contributes nothing
            ]
        }
        self.assertEqual(downroute.extract_tool_targets(resp), {"file_path=/a.py"})

    def test_targets_from_chat_tool_calls(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"type": "function", "function": {"name": "grep", "arguments": '{"pattern": "foo"}'}},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(downroute.extract_tool_targets(resp), {"pattern=foo"})

    def test_malformed_arguments_yield_no_targets(self):
        resp = {"output": [{"type": "function_call", "name": "read_file", "arguments": "not-json"}]}
        self.assertEqual(downroute.extract_tool_targets(resp), set())


# --- OpenAI: proxy integration seam -----------------------------------------

class TestOpenAIProxySeam(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        self.cfg = DownrouteConfig()

    def tearDown(self):
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        self._tmp.cleanup()

    def _body_responses(self, *tool_names: str) -> dict:
        """A /v1/responses body whose current turn used built-in read-only tools."""
        items = [{"type": "message", "role": "user", "content": "go"}]
        items += [{"type": f"{n}_call", "id": f"{n}1"} for n in tool_names]
        return {"model": "gpt-5.6-terra", "input": items}

    def _apply(self, *, crunched, routing_meta, body=None, sampled_shadow=False, category="tool-heavy"):
        openai_proxy._maybe_apply_openai_downroute(
            store_obj=self.store,
            raw_body=body if body is not None else self._body_responses("web_search", "file_search"),
            crunched=crunched,
            routing_meta=routing_meta,
            category=category,
            session_id="sess",
            call_id="call-1",
            resolved_requested_model="gpt-5.6-terra",
            sampled_shadow_pass_through=sampled_shadow,
            input_tokens=1000,
            cfg=self.cfg,
        )

    def test_armed_off_is_noop(self):
        crunched = {"model": "gpt-5.6-terra"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta)
        self.assertEqual(crunched["model"], "gpt-5.6-terra")
        self.assertEqual(routing_meta["downroute"]["decided"], False)
        self.assertEqual(routing_meta["downroute"]["reason"], "pocket-unarmed")
        self.assertEqual(routing_meta["downroute"]["pocket"], "terra->luna")

    def test_forced_f_downroutes_terra_to_luna(self):
        self.store.set_downroute_pocket_f(
            pocket="terra->luna", f=1.0, requested_family="terra",
            target_family="luna", action="arm", reset_window=True,
        )
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "gpt-5.6-terra"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta)
        self.assertEqual(crunched["model"], "gpt-5.6-luna")
        self.assertEqual(routing_meta["routed_model"], "gpt-5.6-luna")
        self.assertEqual(routing_meta["downroute"]["decided"], True)
        self.assertEqual(routing_meta["downroute"]["target_model"], "gpt-5.6-luna")
        self.assertEqual(routing_meta["downroute"]["input_tokens"], 1000)

    def test_sampled_shadow_is_skipped(self):
        self.store.set_downroute_pocket_f(
            pocket="terra->luna", f=1.0, requested_family="terra", target_family="luna", action="arm",
        )
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "gpt-5.6-terra"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta, sampled_shadow=True)
        self.assertEqual(crunched["model"], "gpt-5.6-terra")
        self.assertNotIn("downroute", routing_meta)

    def test_non_passthrough_is_skipped(self):
        self.store.set_downroute_pocket_f(
            pocket="terra->luna", f=1.0, requested_family="terra", target_family="luna", action="arm",
        )
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        # something upstream already moved the model off the requested one
        crunched = {"model": "gpt-5.6-luna"}
        routing_meta = {}
        self._apply(crunched=crunched, routing_meta=routing_meta)
        self.assertEqual(crunched["model"], "gpt-5.6-luna")
        self.assertNotIn("downroute", routing_meta)

    def test_custom_function_tool_fails_closed_no_stamp(self):
        # The user's real generic_openai traffic: a custom function tool named
        # read_file is not on the allow-list -> ineligible -> seam does nothing.
        self.store.set_downroute_pocket_f(
            pocket="terra->luna", f=1.0, requested_family="terra", target_family="luna", action="arm",
        )
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "gpt-5.6-terra"}
        routing_meta = {}
        custom = {
            "model": "gpt-5.6-terra",
            "input": [
                {"type": "message", "role": "user", "content": "read it"},
                {"type": "function_call", "name": "read_file", "arguments": "{}"},
            ],
        }
        self._apply(crunched=crunched, routing_meta=routing_meta, body=custom)
        self.assertEqual(crunched["model"], "gpt-5.6-terra")
        self.assertNotIn("downroute", routing_meta)

    def test_codex_model_has_no_pocket(self):
        self.store.set_downroute_pocket_f(
            pocket="terra->luna", f=1.0, requested_family="terra", target_family="luna", action="arm",
        )
        openai_proxy._OPENAI_DOWNROUTE_F_CACHE.clear()
        crunched = {"model": "gpt-5-codex"}
        routing_meta = {}
        openai_proxy._maybe_apply_openai_downroute(
            store_obj=self.store,
            raw_body=self._body_responses("web_search"),
            crunched=crunched, routing_meta=routing_meta, category="tool-heavy",
            session_id="sess", call_id="call-x", resolved_requested_model="gpt-5-codex",
            sampled_shadow_pass_through=False, input_tokens=1000, cfg=self.cfg,
        )
        self.assertEqual(crunched["model"], "gpt-5-codex")
        self.assertNotIn("downroute", routing_meta)


# --- OpenAI: boundary firewall ----------------------------------------------

class TestOpenAIFirewall(unittest.TestCase):
    """A decided OpenAI downroute must suppress the server-bound managed feedback
    and persist only the local routing_meta (CLAUDE.md "local applies / server
    learns"). Mirrors the Anthropic firewall test."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))
        at = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.store.log_call(
            id="dr-openai", path="/v1/responses", session_id="s1",
            requested_model="gpt-5.6-terra", routed_model="gpt-5.6-luna",
            stream=False, status_code=200, retry_count=0, created_at=at.isoformat(),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, routing_meta):
        from tokenclaw.optimization.openai_outcomes import record_managed_outcome_feedback

        asyncio.run(
            record_managed_outcome_feedback(
                store=self.store, call_id="dr-openai", path="/v1/responses",
                requested_model="gpt-5.6-terra", routed_model="gpt-5.6-luna",
                status_code=200, latency_ms=10, retry_count=0,
                input_tokens_est=100, output_tokens_est=10,
                actual_input_tokens=None, actual_output_tokens=None,
                cache_creation_input_tokens=None, cache_read_input_tokens=None,
                thinking_output_tokens=None, cost_est_usd=None, cost_baseline_usd=None,
                cache_meta={}, crunch_meta={}, routing_meta=routing_meta,
                category="tool-heavy", session_id="s1", error=None,
            )
        )

    def test_decided_downroute_short_circuits_before_server_streams(self):
        routing_meta = {
            "routed_model": "gpt-5.6-luna",
            "managed_recommendation": {"enabled": True},
            "downroute": {"pocket": "terra->luna", "decided": True, "target_model": "gpt-5.6-luna"},
        }
        # If the guard is NOT taken, the first server-bound builder runs and raises.
        with patch(
            "tokenclaw.optimization.openai_outcomes.build_openai_optimization_lifecycle_event",
            side_effect=AssertionError("server-bound path must not run for a decided downroute"),
        ):
            self._run(routing_meta)  # must not raise
        # local routing_meta persisted, stamping the call 'pending' for finalize
        row = self.store.conn.execute(
            "select downroute_verdict from calls where id = 'dr-openai'"
        ).fetchone()
        self.assertEqual(row["downroute_verdict"], "pending")

    def test_non_decided_call_reaches_server_path(self):
        routing_meta = {"routed_model": "gpt-5.6-terra"}  # no decided downroute
        with patch(
            "tokenclaw.optimization.openai_outcomes.build_openai_optimization_lifecycle_event",
            side_effect=RuntimeError("reached server path"),
        ):
            with self.assertRaises(RuntimeError):
                self._run(routing_meta)


# --- shipped-vanilla seeding ------------------------------------------------

class TestSeedDefaultPockets(unittest.TestCase):
    """seed_default_pockets is the shipped-vanilla default: it arms the full
    same-provider cascade at f_start, but idempotently so operator intent
    (disarm, controller-stepped f) survives a restart re-seed."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = Store(str(Path(self._tmp.name) / "tokenclaw.sqlite3"))
        self.cfg = DownrouteConfig()

    def tearDown(self):
        self._tmp.cleanup()

    def test_fresh_seed_arms_full_cascade(self):
        seeded = downroute.seed_default_pockets(self.store, self.cfg)
        self.assertEqual(
            set(seeded),
            {"opus->sonnet", "sonnet->haiku", "sol->terra", "terra->luna", "luna->mini"},
        )
        for key in seeded:
            self.assertAlmostEqual(self.store.get_downroute_pocket_f(key), self.cfg.f_start)
            row = self.store._downroute_pocket_row(key)
            self.assertIsNotNone(row["armed_at"])

    def test_seed_is_idempotent(self):
        first = downroute.seed_default_pockets(self.store, self.cfg)
        self.assertEqual(len(first), 5)
        second = downroute.seed_default_pockets(self.store, self.cfg)
        self.assertEqual(second, [])

    def test_operator_disarm_survives_reseed(self):
        downroute.seed_default_pockets(self.store, self.cfg)
        self.store.set_downroute_pocket_f(
            pocket="terra->luna", f=0.0, requested_family="terra",
            target_family="luna", action="disarm",
        )
        seeded = downroute.seed_default_pockets(self.store, self.cfg)
        self.assertEqual(seeded, [])
        self.assertEqual(self.store.get_downroute_pocket_f("terra->luna"), 0.0)

    def test_controller_stepped_f_survives_reseed(self):
        downroute.seed_default_pockets(self.store, self.cfg)
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=0.20, requested_family="opus",
            target_family="sonnet", action="step",
        )
        downroute.seed_default_pockets(self.store, self.cfg)
        self.assertAlmostEqual(self.store.get_downroute_pocket_f("opus->sonnet"), 0.20)

    def test_partial_seed_fills_only_missing_pockets(self):
        self.store.set_downroute_pocket_f(
            pocket="opus->sonnet", f=0.11, requested_family="opus",
            target_family="sonnet", action="arm",
        )
        seeded = downroute.seed_default_pockets(self.store, self.cfg)
        self.assertNotIn("opus->sonnet", seeded)
        self.assertEqual(len(seeded), 4)
        self.assertAlmostEqual(self.store.get_downroute_pocket_f("opus->sonnet"), 0.11)


if __name__ == "__main__":
    unittest.main()
