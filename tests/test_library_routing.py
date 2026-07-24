import os
import subprocess
import sys
import unittest

import tokenclaw
from tokenclaw.library import RouteResult, route_openai, route_request


def _anthropic_body(*names: str, model: str = "claude-opus-4-5") -> dict:
    """An /v1/messages body whose most recent assistant turn issued tool_use blocks."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "do the thing"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": n, "input": {"file_path": f"/repo/{n}.py"}}
                    for n in names
                ],
            },
        ],
    }


def _chat_messages(*names: str) -> list:
    """OpenAI chat-completions messages whose last assistant turn called these tools."""
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": n, "arguments": "{}"}}
                for n in names
            ],
        },
    ]


def _responses_input(*names: str) -> list:
    """OpenAI /v1/responses flat input list with a user turn then function_call items."""
    return [
        {"type": "message", "role": "user", "content": "search the docs"},
        *[{"type": "function_call", "name": n, "arguments": "{}"} for n in names],
    ]


class RouteEligibilityTests(unittest.TestCase):
    def test_custom_tool_on_allow_list_downroutes_terra_to_luna(self):
        # The terra->luna dead-end: a custom function tool the frozenset doesn't know,
        # made eligible purely by the caller-supplied allow-list — no dashboard/server.
        route = route_openai(
            model="gpt-5.6-terra",
            input=_responses_input("read_file"),
            read_only_tools=["read_file"],
            f=1.0,
            call_id="c1",
        )
        self.assertIsInstance(route, RouteResult)
        self.assertTrue(route.eligible)
        self.assertTrue(route.downrouted)
        self.assertEqual(route.pocket, "terra->luna")
        self.assertEqual(route.routed_model, "gpt-5.6-luna")
        self.assertEqual(route.reason, "read-only-tool-heavy")
        self.assertEqual(route.tool_names, ("read_file",))

    def test_chat_shape_downroutes_at_full_rate(self):
        route = route_openai(
            model="gpt-5.6-terra",
            messages=_chat_messages("search_docs", "get_file"),
            read_only_tools=["search_docs", "get_file"],
            f=1.0,
            call_id="c2",
        )
        self.assertTrue(route.downrouted)
        self.assertEqual(route.routed_model, "gpt-5.6-luna")

    def test_anthropic_opus_downroutes_to_sonnet(self):
        # Library uses replace semantics: the caller's array is the full read-only
        # source of truth (no inherited Claude-Code built-ins), so it must vouch here.
        route = route_request(
            _anthropic_body("Read", "Grep", model="claude-opus-4-5"),
            read_only_tools=["Read", "Grep"],
            f=1.0,
            call_id="c3",
        )
        from tokenclaw import router

        self.assertTrue(route.downrouted)
        self.assertEqual(route.pocket, "opus->sonnet")
        self.assertEqual(route.routed_model, router.SONNET_DEFAULT)

    def test_mutating_tool_fails_closed(self):
        route = route_openai(
            model="gpt-5.6-terra",
            input=_responses_input("read_file", "write_file"),
            read_only_tools=["read_file"],  # write_file NOT vouched
            f=1.0,
            call_id="c4",
        )
        self.assertFalse(route.eligible)
        self.assertFalse(route.downrouted)
        self.assertEqual(route.routed_model, "gpt-5.6-terra")
        self.assertTrue(route.reason.startswith("mutating-or-unknown:"))
        self.assertIn("write_file", route.reason)

    def test_empty_allow_list_fails_closed_on_custom_tool(self):
        route = route_openai(
            model="gpt-5.6-terra",
            input=_responses_input("read_file"),
            read_only_tools=[],
            f=1.0,
            call_id="c5",
        )
        self.assertFalse(route.eligible)
        self.assertTrue(route.reason.startswith("mutating-or-unknown:"))

    def test_no_recent_tool_use_is_ineligible(self):
        route = route_openai(
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "just chat"}],
            read_only_tools=["read_file"],
            f=1.0,
            call_id="c6",
        )
        self.assertFalse(route.eligible)
        self.assertEqual(route.reason, "no-recent-tool-use")


class RouteArmingTests(unittest.TestCase):
    def test_f_zero_is_eligible_but_unarmed(self):
        route = route_openai(
            model="gpt-5.6-terra",
            input=_responses_input("read_file"),
            read_only_tools=["read_file"],
            f=0.0,
            call_id="c7",
        )
        self.assertTrue(route.eligible)
        self.assertFalse(route.downrouted)
        self.assertEqual(route.reason, "pocket-unarmed")
        self.assertEqual(route.routed_model, "gpt-5.6-terra")

    def test_no_pocket_floor_tier(self):
        route = route_openai(
            model="gpt-5-mini",
            input=_responses_input("read_file"),
            read_only_tools=["read_file"],
            f=1.0,
            call_id="c8",
        )
        self.assertFalse(route.eligible)
        self.assertEqual(route.reason, "no-pocket")
        self.assertEqual(route.routed_model, "gpt-5-mini")

    def test_unknown_model_has_no_pocket(self):
        route = route_request(
            {"model": "gpt-5-codex", "input": _responses_input("read_file")},
            read_only_tools=["read_file"],
            f=1.0,
        )
        self.assertEqual(route.reason, "no-pocket")
        self.assertFalse(route.downrouted)


class RouteCoinFlipTests(unittest.TestCase):
    def test_rate_approximates_f(self):
        fired = 0
        n = 400
        for i in range(n):
            route = route_openai(
                model="gpt-5.6-terra",
                input=_responses_input("read_file"),
                read_only_tools=["read_file"],
                f=0.5,
                call_id=f"call-{i}",
            )
            fired += int(route.downrouted)
        self.assertAlmostEqual(fired / n, 0.5, delta=0.1)

    def test_same_call_id_is_deterministic(self):
        kwargs = dict(
            model="gpt-5.6-terra",
            input=_responses_input("read_file"),
            read_only_tools=["read_file"],
            f=0.5,
            session_id="s1",
            call_id="stable",
        )
        first = route_openai(**kwargs)
        for _ in range(5):
            self.assertEqual(route_openai(**kwargs).downrouted, first.downrouted)


class RouteHygieneTests(unittest.TestCase):
    def test_body_not_mutated(self):
        body = _anthropic_body("Read", model="claude-opus-4-5")
        before = str(body)
        route_request(body, read_only_tools=[], f=1.0, call_id="x")
        self.assertEqual(str(body), before)

    def test_requires_exactly_one_payload(self):
        with self.assertRaises(ValueError):
            route_openai(model="gpt-5.6-terra", read_only_tools=["read_file"])
        with self.assertRaises(ValueError):
            route_openai(
                model="gpt-5.6-terra", messages=[], input=[], read_only_tools=["read_file"]
            )

    def test_custom_tier_map_overrides_default(self):
        route = route_openai(
            model="gpt-5.6-terra",
            input=_responses_input("read_file"),
            read_only_tools=["read_file"],
            f=1.0,
            call_id="tm",
            tier_map={"luna": "my-luna-alias"},
        )
        self.assertEqual(route.routed_model, "my-luna-alias")


class RoutePublicSurfaceTests(unittest.TestCase):
    def test_exports_present(self):
        for name in ("route_request", "route_openai", "RouteResult"):
            self.assertTrue(hasattr(tokenclaw, name), name)

    def test_import_is_server_free_with_routing_symbols(self):
        code = (
            "import sys, tokenclaw;"
            "tokenclaw.route_request; tokenclaw.route_openai; tokenclaw.RouteResult;"
            "heavy=[m for m in sys.modules if m.split('.')[0] in {'fastapi','uvicorn','httpx','starlette'}];"
            "print('HEAVY' if heavy else 'CLEAN')"
        )
        env = dict(os.environ)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("CLEAN", out.stdout)


if __name__ == "__main__":
    unittest.main()
