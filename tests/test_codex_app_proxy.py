import io
import json
import unittest
from unittest.mock import patch

from agentflow_proxy import codex_app_proxy


class FailingStore:
    def log_codex_app_event(self, **kwargs):
        raise RuntimeError("database is locked")


class CodexAppProxyTelemetryTest(unittest.TestCase):
    def test_locked_telemetry_store_does_not_interrupt_relay_recording(self):
        request_started = {}
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"input": "hello"},
        }

        with patch.object(codex_app_proxy, "store", FailingStore()):
            with patch.object(codex_app_proxy.sys, "stderr", io.StringIO()) as stderr:
                codex_app_proxy._record_message(
                    json.dumps(message),
                    direction="client_to_server",
                    session_id="session-a",
                    request_started=request_started,
                )

        self.assertIn("1", request_started)
        self.assertIn("database is locked", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
