import os
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory

import tokenclaw
from tokenclaw.library import CrunchResult, LocalCache, crunch_openai, crunch_request


def _bloated_messages():
    dup = "PROJECT RULES:\n" + ("- a rule line with trailing spaces        \n" * 30)
    return [
        {"role": "system", "content": dup},
        {"role": "user", "content": "code:\n" + ("y = 1\n\n\n\n" * 40)},
        {"role": "user", "content": dup},  # exact duplicate of the system block
    ]


class CrunchRequestTests(unittest.TestCase):
    def test_reduces_and_reports_rules(self):
        body = {"model": "gpt-5", "messages": _bloated_messages()}
        result = crunch_request(body, provider="openai")
        self.assertIsInstance(result, CrunchResult)
        self.assertTrue(result.changed)
        self.assertGreater(result.chars_saved, 0)
        self.assertEqual(result.chars_saved, result.chars_before - result.chars_after)
        self.assertGreaterEqual(result.input_tokens_saved_est, 0)
        self.assertIn("whitespace_normalization", result.applied_rules)
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.endpoint, "/v1/chat/completions")

    def test_input_body_not_mutated(self):
        messages = _bloated_messages()
        original_len = len(messages[0]["content"])
        crunch_request({"model": "gpt-5", "messages": messages}, provider="openai")
        self.assertEqual(len(messages[0]["content"]), original_len)

    def test_rejects_non_dict(self):
        with self.assertRaises(TypeError):
            crunch_request("not a dict")  # type: ignore[arg-type]


class CrunchOpenAITests(unittest.TestCase):
    def test_chat_kwargs_are_splat_ready(self):
        kwargs, result = crunch_openai(model="gpt-5", messages=_bloated_messages(), temperature=0)
        self.assertEqual(set(kwargs), {"model", "messages", "temperature"})
        self.assertEqual(kwargs["model"], "gpt-5")
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["messages"], result.body["messages"])
        self.assertEqual(result.endpoint, "/v1/chat/completions")

    def test_responses_variant(self):
        dup = "RULES:\n" + ("- x        \n" * 30)
        kwargs, result = crunch_openai(model="gpt-5", input=[{"role": "user", "content": dup}])
        self.assertEqual(set(kwargs), {"model", "input"})
        self.assertEqual(result.endpoint, "/v1/responses")

    def test_requires_exactly_one_payload(self):
        with self.assertRaises(ValueError):
            crunch_openai(model="gpt-5")
        with self.assertRaises(ValueError):
            crunch_openai(model="gpt-5", messages=[], input=[])


class LocalCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.cache = LocalCache(path=os.path.join(self._tmp.name, "cache.sqlite3"))

    def tearDown(self):
        self.cache.close()
        self._tmp.cleanup()

    def test_roundtrip_with_result(self):
        _, result = crunch_openai(model="gpt-5", messages=_bloated_messages())
        response = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        self.assertIsNone(self.cache.get(result))
        self.cache.put(result, response)
        self.assertEqual(self.cache.get(result), response)

    def test_result_and_body_resolve_same_key(self):
        _, result = crunch_openai(model="gpt-5", messages=_bloated_messages())
        key_from_result = self.cache.key(result)
        key_from_body = self.cache.key(
            result.body, endpoint=result.endpoint, provider=result.provider
        )
        self.assertEqual(key_from_result, key_from_body)

    def test_distinct_requests_do_not_collide(self):
        _, a = crunch_openai(model="gpt-5", messages=[{"role": "user", "content": "one"}])
        _, b = crunch_openai(model="gpt-5", messages=[{"role": "user", "content": "two"}])
        self.cache.put(a, {"id": "a"})
        self.assertEqual(self.cache.get(a), {"id": "a"})
        self.assertIsNone(self.cache.get(b))


class PublicSurfaceTests(unittest.TestCase):
    def test_exports_present(self):
        for name in ("crunch_request", "crunch_openai", "CrunchResult", "LocalCache", "estimate_tokens_from_text"):
            self.assertTrue(hasattr(tokenclaw, name), name)

    def test_import_is_server_free(self):
        # In a fresh interpreter, `import tokenclaw` must not pull the proxy web stack.
        code = (
            "import sys, tokenclaw;"
            "heavy=[m for m in sys.modules if m.split('.')[0] in {'fastapi','uvicorn','httpx','starlette'}];"
            "print('HEAVY' if heavy else 'CLEAN')"
        )
        env = dict(os.environ)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("CLEAN", out.stdout)


if __name__ == "__main__":
    unittest.main()
