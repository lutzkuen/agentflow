# Using TokenClaw as a Python library (no server)

TokenClaw's local capabilities can run directly inside your own Python
application — no proxy server, no background process. This is the supported path
when you have a self-built OpenAI (or Anthropic) app and just want TokenClaw to
**crunch** your requests and/or serve a **local exact-match cache**.

Routing is deliberately *not* part of the library: it is "backed or off" (it
needs the managed server or manual hard rules), so a stateless in-process call
cannot honor it. Use the proxy for routing.

## Install

The base install is server-free and has a single dependency (PyYAML):

```bash
pip install tokenclaw
```

You do **not** need the `[server]` extra (fastapi/uvicorn/httpx) for library use.

## Quickstart — OpenAI, crunch + local cache

TokenClaw never calls the provider for you and does not depend on the `openai`
SDK. You bring your own client; TokenClaw shrinks the request and (optionally)
caches the response.

```python
from openai import OpenAI
from tokenclaw import crunch_openai, LocalCache

client = OpenAI()
cache = LocalCache()  # SQLite at ~/.tokenclaw/library_cache.sqlite3 by default

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_text},
]

# 1) Crunch: returns splat-ready kwargs + a savings report.
kwargs, report = crunch_openai(model="gpt-5", messages=messages, temperature=0)
print(f"crunched {report.chars_before}->{report.chars_after} chars "
      f"(~{report.input_tokens_saved_est} input tokens) via {report.applied_rules}")

# 2) Optional exact-match cache, keyed on the *crunched* request.
cached = cache.get(report)
if cached is not None:
    response = cached
else:
    response = client.chat.completions.create(**kwargs).model_dump()
    cache.put(report, response)

# `response` is a plain dict — use it as you normally would.
```

### Responses API

Pass `input=` instead of `messages=`; TokenClaw targets `/v1/responses`
automatically:

```python
kwargs, report = crunch_openai(model="gpt-5", input=my_input)
resp = client.responses.create(**kwargs)
```

## Anthropic

Use the provider-agnostic core for Anthropic message bodies:

```python
from tokenclaw import crunch_request, LocalCache

body = {"model": "claude-opus-4-8", "max_tokens": 1024, "messages": messages}
result = crunch_request(body, provider="anthropic")   # endpoint defaults to /v1/messages
# send result.body with the anthropic SDK; cache with LocalCache(...).get/put(result)
```

## What crunching does

Crunching is conservative and lossless-first — it never calls another model to
summarize. Rules currently include:

- **`whitespace_normalization`** — collapse runs of whitespace in large text blocks.
- **`exact_duplicate_block_omission`** — drop exact repeats of a text block within
  the same request (common with re-sent system prompts / rules).
- Bounded compaction of oversized *older* non-tool blocks once a request exceeds a
  size threshold (tune with `threshold_chars=`).

`report.applied_rules` lists what fired; `report.meta` has the full per-rule
detail if you need it.

## API reference

### `crunch_request(body, *, provider="openai", endpoint=None, threshold_chars=None) -> CrunchResult`
Crunch a raw provider request dict. The input `body` is **not** mutated.
`provider` is `"openai"` or `"anthropic"`; `endpoint` defaults to the provider's
primary route and selects the crunch surface.

### `crunch_openai(*, model=None, messages=None, input=None, endpoint=None, threshold_chars=None, **passthrough) -> (kwargs, CrunchResult)`
Convenience wrapper for OpenAI. Provide exactly one of `messages=` (Chat
Completions) or `input=` (Responses). Extra keyword args (`temperature`, `tools`,
…) pass through onto `kwargs`, which is splat-ready for the OpenAI SDK.

### `CrunchResult`
Dataclass with: `body` (the crunched payload to send), `changed`, `chars_before`,
`chars_after`, `chars_saved`, `crunch_ratio`, `input_tokens_saved_est`,
`applied_rules` (list of rule ids), `provider`, `endpoint`, and `meta` (raw crunch
metadata).

### `LocalCache(path="~/.tokenclaw/library_cache.sqlite3")`
Local, server-free exact-match response cache (SQLite).

- `get(body_or_result, *, endpoint=None, provider=None, namespace=None) -> dict | None`
- `put(body_or_result, response, *, endpoint=None, provider=None, model=None, namespace=None, ttl_seconds=None) -> str`
- `key(body_or_result, ...) -> str` — the exact-match key (advanced use)
- `close()`

Pass a `CrunchResult` (recommended — it carries provider/endpoint and the crunched
body) or a raw request dict plus `endpoint`/`provider`. Cache the **crunched**
request so keys match what you actually send. Entries expire after a default TTL
unless you pass `ttl_seconds`. Use `namespace=` to isolate cache scopes.

## Notes & limits

- **Exact-match only.** Semantic (embedding) caching needs an embedding provider
  and is not exposed in v1.
- **Keys ignore auth**, and include provider + endpoint + body. Different providers
  or endpoints never share cache entries.
- **Concurrency:** a `LocalCache` instance is safe to share across threads (the
  underlying SQLite store is lock-guarded). For multiple processes, point them at
  the same `path`.
- **Config via env still applies** to crunching (e.g. `TOKENCLAW_CRUNCH=0` disables
  it, `TOKENCLAW_CACHE_NAMESPACE` sets a default cache namespace), so the library
  honors the same local knobs as the proxy.
```
