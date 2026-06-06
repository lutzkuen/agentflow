import unittest

from agentflow_proxy.cache import cache_decision_meta, cache_lookup_meta


class CacheDecisionMetaTest(unittest.TestCase):
    def test_cache_hit_metadata_has_explicit_status_and_source(self):
        meta = cache_decision_meta(
            "hit",
            "exact-match",
            hit_type="exact",
            exact_enabled=True,
            semantic_enabled=False,
        )

        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["status"], "hit")
        self.assertEqual(meta["reason"], "exact-match")
        self.assertEqual(meta["hit_type"], "exact")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_tool_requests_are_skipped_when_tool_cache_disabled(self):
        can_exact, can_semantic, meta = cache_lookup_meta(has_tool_blocks=True)

        self.assertFalse(can_exact)
        self.assertFalse(can_semantic)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "tools-disabled")
        self.assertTrue(meta["enabled"])
        self.assertFalse(meta["tool_cache_enabled"])

    def test_non_tool_requests_report_exact_miss_by_default(self):
        can_exact, can_semantic, meta = cache_lookup_meta(has_tool_blocks=False)

        self.assertTrue(can_exact)
        self.assertFalse(can_semantic)
        self.assertEqual(meta["status"], "miss")
        self.assertEqual(meta["reason"], "exact-miss")


if __name__ == "__main__":
    unittest.main()
