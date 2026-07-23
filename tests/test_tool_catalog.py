"""Tests for the tool catalog + operator-settable read-only overrides that back
the dashboard "Tool calls" tab and feed downroute eligibility.

Store and downroute tests need only the local package (like test_store.py /
test_downroute.py); the endpoint tests are gated on the web stack.
"""

from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenclaw import downroute
from tokenclaw.downroute import DownrouteConfig, READ_ONLY_TOOLS
from tokenclaw.store import Store


def _assistant_tool_use(*names: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "name": n, "input": {"file_path": f"/repo/{n}.py"}}
            for n in names
        ],
    }


def _body(*, last_tools: tuple[str, ...], model: str = "claude-opus-4-5") -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "go"},
            _assistant_tool_use(*last_tools),
            {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
        ],
    }


class ToolCatalogStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "tokenclaw.sqlite3"))

    def tearDown(self):
        self.store.conn.close()
        self.tmp.cleanup()

    def test_record_tool_sightings_counts_turns_and_dedupes_within_turn(self):
        # A single turn using the same tool twice counts as one turn-seen.
        self.store.record_tool_sightings(["Read", "Read", "Grep"])
        by_name = {r["name"]: r for r in self.store.list_tool_catalog()}
        self.assertEqual(by_name["read"]["call_count"], 1)
        self.assertEqual(by_name["grep"]["call_count"], 1)

        # A second turn bumps the count and advances last_seen.
        first_last_seen = by_name["read"]["last_seen"]
        self.store.record_tool_sightings(["read"])
        read_row = {r["name"]: r for r in self.store.list_tool_catalog()}["read"]
        self.assertEqual(read_row["call_count"], 2)
        self.assertGreaterEqual(read_row["last_seen"], first_last_seen)
        self.assertEqual(read_row["readonly_override"], None)

    def test_record_tool_sightings_lowercases_and_ignores_blanks(self):
        self.store.record_tool_sightings(["MyCustomTool", "  ", "", None])  # type: ignore[list-item]
        names = [r["name"] for r in self.store.list_tool_catalog()]
        self.assertEqual(names, ["mycustomtool"])

    def test_list_tool_catalog_orders_by_call_count_desc_then_name(self):
        self.store.record_tool_sightings(["alpha"])
        self.store.record_tool_sightings(["beta"])
        self.store.record_tool_sightings(["beta"])
        self.store.record_tool_sightings(["gamma"])
        names = [r["name"] for r in self.store.list_tool_catalog()]
        self.assertEqual(names[0], "beta")  # highest count first
        self.assertEqual(names[1:], ["alpha", "gamma"])  # ties sorted by name

    def test_set_tool_readonly_override_set_true_false_and_clear(self):
        row = self.store.set_tool_readonly_override("CustomTool", True)
        self.assertEqual(row["name"], "customtool")
        self.assertEqual(row["readonly_override"], 1)
        self.assertEqual(self.store.read_only_override_map(), {"customtool": True})

        self.store.set_tool_readonly_override("customtool", False)
        self.assertEqual(self.store.read_only_override_map(), {"customtool": False})

        cleared = self.store.set_tool_readonly_override("customtool", None)
        self.assertEqual(cleared["readonly_override"], None)
        self.assertEqual(self.store.read_only_override_map(), {})

    def test_set_tool_readonly_override_upserts_unseen_tool(self):
        # A tool never observed is still settable (call_count stays 0).
        row = self.store.set_tool_readonly_override("never_seen", True)
        self.assertEqual(row["call_count"], 0)
        self.assertEqual(row["readonly_override"], 1)

    def test_set_tool_readonly_override_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            self.store.set_tool_readonly_override("   ", True)

    def test_read_only_override_map_only_includes_rows_with_override(self):
        self.store.record_tool_sightings(["seen_no_override"])
        self.store.set_tool_readonly_override("on", True)
        self.store.set_tool_readonly_override("off", False)
        self.assertEqual(
            self.store.read_only_override_map(), {"on": True, "off": False}
        )


class EffectiveReadOnlyNamesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "tokenclaw.sqlite3"))
        self._reset_cache()

    def tearDown(self):
        self._reset_cache()
        self.store.conn.close()
        self.tmp.cleanup()

    def _reset_cache(self):
        downroute._EFFECTIVE_READ_ONLY_CACHE["names"] = None
        downroute._EFFECTIVE_READ_ONLY_CACHE["expires"] = 0.0

    def test_merges_both_directions(self):
        self.store.set_tool_readonly_override("customtool", True)
        self.store.set_tool_readonly_override("read", False)  # untick a default
        names = downroute.effective_read_only_names(self.store)
        self.assertIn("customtool", names)  # added
        self.assertNotIn("read", names)  # removed
        self.assertIn("grep", names)  # untouched default survives

    def test_defaults_to_code_allowlist_when_no_overrides(self):
        names = downroute.effective_read_only_names(self.store)
        self.assertEqual(names, READ_ONLY_TOOLS)

    def test_store_failure_falls_back_to_code_default(self):
        class Boom:
            def read_only_override_map(self):
                raise RuntimeError("db down")

        names = downroute.effective_read_only_names(Boom())
        self.assertEqual(names, READ_ONLY_TOOLS)

    def test_ttl_caches_between_calls_and_reresolves_after_expiry(self):
        class Counting:
            def __init__(self):
                self.calls = 0

            def read_only_override_map(self):
                self.calls += 1
                return {"customtool": True}

        counting = Counting()
        first = downroute.effective_read_only_names(counting)
        second = downroute.effective_read_only_names(counting)
        self.assertEqual(counting.calls, 1)  # second call served from cache
        self.assertEqual(first, second)
        self.assertIn("customtool", first)

        downroute._EFFECTIVE_READ_ONLY_CACHE["expires"] = 0.0  # force expiry
        downroute.effective_read_only_names(counting)
        self.assertEqual(counting.calls, 2)


class ClassifyEligibilityOverrideTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DownrouteConfig()

    def test_override_flips_custom_tool_turn_to_eligible(self):
        body = _body(last_tools=("CustomTool",))

        # Baseline: the code allow-list does not know the tool -> ineligible.
        before = downroute.classify_eligibility(body, "tool-heavy", self.cfg)
        self.assertFalse(before.eligible)
        self.assertTrue(before.reason.startswith("mutating-or-unknown:"))

        # rule 6: eligibility is a classification, not a rewrite -> token count of
        # the turn is invariant across the flip; only the decision changes.
        tokens_before = len(json.dumps(body))
        allow = frozenset(READ_ONLY_TOOLS | {"customtool"})
        after = downroute.classify_eligibility(
            body, "tool-heavy", self.cfg, read_only_names=allow
        )
        tokens_after = len(json.dumps(body))
        self.assertEqual(tokens_before, tokens_after)
        self.assertTrue(after.eligible)
        self.assertEqual(after.reason, "read-only-tool-heavy")

    def test_untick_default_makes_read_only_turn_ineligible(self):
        body = _body(last_tools=("Read",))
        allow = frozenset(READ_ONLY_TOOLS - {"read"})
        elig = downroute.classify_eligibility(
            body, "tool-heavy", self.cfg, read_only_names=allow
        )
        self.assertFalse(elig.eligible)
        self.assertIn("read", elig.reason)

    def test_full_loop_store_override_makes_turn_eligible(self):
        tmp = TemporaryDirectory()
        try:
            store = Store(str(Path(tmp.name) / "tokenclaw.sqlite3"))
            downroute._EFFECTIVE_READ_ONLY_CACHE["names"] = None
            downroute._EFFECTIVE_READ_ONLY_CACHE["expires"] = 0.0
            store.set_tool_readonly_override("customtool", True)
            allow = downroute.effective_read_only_names(store)
            elig = downroute.classify_eligibility(
                _body(last_tools=("customtool",)), "tool-heavy", self.cfg,
                read_only_names=allow,
            )
            self.assertTrue(elig.eligible)
            store.conn.close()
        finally:
            downroute._EFFECTIVE_READ_ONLY_CACHE["names"] = None
            downroute._EFFECTIVE_READ_ONLY_CACHE["expires"] = 0.0
            tmp.cleanup()


class ScheduleToolSightingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "tokenclaw.sqlite3"))
        self._old = os.environ.get("TOKENCLAW_TOOL_CATALOG")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("TOKENCLAW_TOOL_CATALOG", None)
        else:
            os.environ["TOKENCLAW_TOOL_CATALOG"] = self._old
        self.store.conn.close()
        self.tmp.cleanup()

    def test_sync_fallback_records_when_no_event_loop(self):
        # Called outside an event loop (as in this test), it writes synchronously.
        os.environ.pop("TOKENCLAW_TOOL_CATALOG", None)
        downroute.schedule_tool_sightings(self.store, _body(last_tools=("Read", "Edit")))
        names = sorted(r["name"] for r in self.store.list_tool_catalog())
        self.assertEqual(names, ["edit", "read"])

    def test_kill_switch_disables_recording(self):
        os.environ["TOKENCLAW_TOOL_CATALOG"] = "0"
        downroute.schedule_tool_sightings(self.store, _body(last_tools=("Read",)))
        self.assertEqual(self.store.list_tool_catalog(), [])


HAS_WEB = importlib.util.find_spec("fastapi") is not None

if HAS_WEB:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tokenclaw.admin import create_admin_router
    from tokenclaw.dashboard_app import create_dashboard_app


@unittest.skipUnless(HAS_WEB, "runtime web dependencies are not installed")
class AdminToolReadonlyEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "tokenclaw.sqlite3"))
        self._old_event_log = os.environ.get("TOKENCLAW_POLICY_EVENTS_LOG")
        os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = str(
            Path(self.tmp.name) / "policy_events.jsonl"
        )
        app = FastAPI()
        app.include_router(create_admin_router(store_obj=lambda: self.store))
        self.app = app

    def tearDown(self):
        if self._old_event_log is None:
            os.environ.pop("TOKENCLAW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = self._old_event_log
        self.store.conn.close()
        self.tmp.cleanup()

    def test_loopback_post_sets_override(self):
        local = TestClient(self.app, client=("127.0.0.1", 50000))
        resp = local.post(
            "/tokenclaw/admin/tool-readonly",
            json={"name": "CustomTool", "read_only": True},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["tool"]["name"], "customtool")
        self.assertEqual(data["tool"]["readonly_override"], 1)
        self.assertFalse(data["provider_calls_made"])
        self.assertFalse(data["managed_server_calls_made"])
        self.assertEqual(self.store.read_only_override_map(), {"customtool": True})

    def test_loopback_post_clears_override_with_null(self):
        self.store.set_tool_readonly_override("customtool", True)
        local = TestClient(self.app, client=("127.0.0.1", 50000))
        resp = local.post(
            "/tokenclaw/admin/tool-readonly",
            json={"name": "customtool", "read_only": None},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.store.read_only_override_map(), {})

    def test_remote_post_is_forbidden(self):
        remote = TestClient(self.app, client=("192.168.1.50", 50000))
        resp = remote.post(
            "/tokenclaw/admin/tool-readonly",
            json={"name": "customtool", "read_only": True},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertFalse(resp.json()["ok"])
        self.assertEqual(resp.json()["error"]["type"], "forbidden")
        # The forbidden request never touched the store.
        self.assertEqual(self.store.read_only_override_map(), {})

    def test_remote_options_preflight_is_forbidden(self):
        remote = TestClient(self.app, client=("192.168.1.50", 50000))
        resp = remote.options("/tokenclaw/admin/tool-readonly")
        self.assertEqual(resp.status_code, 403)

    def test_loopback_options_preflight_ok(self):
        local = TestClient(self.app, client=("127.0.0.1", 50000))
        resp = local.options("/tokenclaw/admin/tool-readonly")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    def test_missing_name_is_rejected(self):
        local = TestClient(self.app, client=("127.0.0.1", 50000))
        resp = local.post(
            "/tokenclaw/admin/tool-readonly", json={"read_only": True}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["type"], "invalid_payload")

    def test_non_boolean_read_only_is_rejected(self):
        local = TestClient(self.app, client=("127.0.0.1", 50000))
        resp = local.post(
            "/tokenclaw/admin/tool-readonly",
            json={"name": "customtool", "read_only": "yes"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["type"], "invalid_payload")


@unittest.skipUnless(HAS_WEB, "runtime web dependencies are not installed")
class DashboardToolEndpointTests(unittest.TestCase):
    def _make_app(self, *, admin_forwarder=None):
        return create_dashboard_app(
            store_obj=lambda: self.store,
            default_db=self.db_path,
            upstream="https://anthropic.test",
            limiter_status=lambda: [],
            limiter_config={
                "min_request_interval_ms": 0,
                "max_tier_backoff_wait_s": 30,
                "max_concurrent_per_tier": 2,
            },
            admin_forwarder=admin_forwarder,
        )

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tokenclaw.sqlite3")
        self.store = Store(self.db_path)
        self._old_event_log = os.environ.get("TOKENCLAW_POLICY_EVENTS_LOG")
        self._old_allow = os.environ.get("TOKENCLAW_DASHBOARD_ALLOW_WRITES")
        os.environ["TOKENCLAW_POLICY_EVENTS_LOG"] = str(
            Path(self.tmp.name) / "policy_events.jsonl"
        )

    def tearDown(self):
        for key, saved in (
            ("TOKENCLAW_POLICY_EVENTS_LOG", self._old_event_log),
            ("TOKENCLAW_DASHBOARD_ALLOW_WRITES", self._old_allow),
        ):
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved
        self.store.conn.close()
        self.tmp.cleanup()

    def test_stats_tools_reports_defaults_and_overrides(self):
        self.store.record_tool_sightings(["Read"])  # a built-in read-only default
        self.store.record_tool_sightings(["CustomTool"])  # unknown -> not read-only
        self.store.set_tool_readonly_override("customtool", True)  # operator ticks it

        client = TestClient(self._make_app())
        resp = client.get("/tokenclaw/stats/tools")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["schema"], "tokenclaw.stats_tools.v1")
        by_name = {t["name"]: t for t in payload["tools"]}

        self.assertTrue(by_name["read"]["default_read_only"])
        self.assertTrue(by_name["read"]["effective_read_only"])
        self.assertFalse(by_name["read"]["has_override"])

        self.assertFalse(by_name["customtool"]["default_read_only"])
        self.assertTrue(by_name["customtool"]["effective_read_only"])
        self.assertTrue(by_name["customtool"]["has_override"])
        self.assertEqual(by_name["read"]["call_count"], 1)

    def test_forwarder_writes_through_admin_when_enabled(self):
        os.environ["TOKENCLAW_DASHBOARD_ALLOW_WRITES"] = "1"
        calls = []

        async def fake_forwarder(path, payload, headers):
            calls.append((path, payload, headers))
            return 200, {
                "schema": "tokenclaw.tool_readonly_override.v1",
                "ok": True,
                "tool": {"name": "customtool", "readonly_override": 1},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
            }

        client = TestClient(
            self._make_app(admin_forwarder=fake_forwarder),
            client=("192.168.178.25", 50000),
        )
        resp = client.post(
            "/tokenclaw/dashboard/admin/tool-readonly",
            json={"name": "customtool", "read_only": True},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        self.assertEqual(len(calls), 1)
        path, payload, headers = calls[0]
        self.assertEqual(path, "/tokenclaw/admin/tool-readonly")
        self.assertEqual(payload, {"name": "customtool", "read_only": True})
        self.assertEqual(headers["x-tokenclaw-admin-source"], "dashboard_lan_forwarder")
        self.assertEqual(headers["x-tokenclaw-forwarded-client-host"], "192.168.178.25")

    def test_forwarder_is_disabled_when_writes_off(self):
        os.environ["TOKENCLAW_DASHBOARD_ALLOW_WRITES"] = "0"
        calls = []

        async def fake_forwarder(path, payload, headers):
            calls.append((path, payload, headers))
            return 200, {"ok": True}

        client = TestClient(
            self._make_app(admin_forwarder=fake_forwarder),
            client=("192.168.178.25", 50000),
        )
        resp = client.post(
            "/tokenclaw/dashboard/admin/tool-readonly",
            json={"name": "customtool", "read_only": True},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertFalse(resp.json()["ok"])
        self.assertEqual(resp.json()["error"]["type"], "dashboard_writes_disabled")
        self.assertEqual(
            resp.json()["schema"],
            "tokenclaw.dashboard_tool_readonly_override_forward.v1",
        )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
