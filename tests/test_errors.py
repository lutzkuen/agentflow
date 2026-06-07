import unittest

from agentflow_proxy.errors import upstream_error_text


class UpstreamErrorTextTest(unittest.TestCase):
    def test_empty_body_falls_back_to_status(self):
        self.assertEqual(upstream_error_text("", 400), "upstream_error: status=400")
        self.assertEqual(upstream_error_text(None, 529), "upstream_error: status=529")

    def test_json_body_is_stable_and_nonblank(self):
        self.assertEqual(
            upstream_error_text({"error": {"message": "bad request", "type": "invalid"}}, 400),
            '{"error":{"message":"bad request","type":"invalid"}}',
        )

    def test_bytes_body_is_decoded_and_trimmed(self):
        self.assertEqual(upstream_error_text(b"  upstream failed  ", 500), "upstream failed")

    def test_error_text_is_limited(self):
        self.assertEqual(upstream_error_text("x" * 20, 500, limit=7), "xxxxxxx")


if __name__ == "__main__":
    unittest.main()
