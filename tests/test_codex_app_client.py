from __future__ import annotations

import argparse
import unittest


from agentflow_proxy.codex_app_client import thread_sandbox_mode, turn_sandbox_policy_type


class CodexAppClientTests(unittest.TestCase):
    def test_sandbox_modes_are_serialized_for_each_protocol_shape(self) -> None:
        self.assertEqual(thread_sandbox_mode("danger-full-access"), "danger-full-access")
        self.assertEqual(turn_sandbox_policy_type("danger-full-access"), "dangerFullAccess")
        self.assertEqual(thread_sandbox_mode("workspaceWrite"), "workspace-write")
        self.assertEqual(turn_sandbox_policy_type("workspace-write"), "workspaceWrite")

    def test_unknown_sandbox_mode_is_rejected(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            thread_sandbox_mode("unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            turn_sandbox_policy_type("unknown")


if __name__ == "__main__":
    unittest.main()
