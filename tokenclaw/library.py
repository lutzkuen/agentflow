"""Public, server-free library API for TokenClaw.

This module is the supported way to use TokenClaw's local capabilities directly
inside another Python application — for example a self-built OpenAI or Anthropic
app — without running the proxy server.

It exposes two things:

- **Crunching** (stateless): :func:`crunch_request` / :func:`crunch_openai` shrink a
  request payload with the same manual, lossless-first rules the proxy applies
  locally (whitespace normalization, exact-duplicate block omission, bounded
  compaction of oversized older blocks). No model is called to summarize.
- **Local exact-match cache** (stateful, SQLite): :class:`LocalCache` stores and
  replays provider responses keyed by the (post-crunch) request, endpoint, and
  provider — the same key discipline the proxy uses.

Routing is intentionally NOT exposed here: it is "backed or off" (it needs the
managed server or manual hard rules) and cannot be honored by a stateless call.

Everything here imports cleanly without the proxy's web stack — see the ``[server]``
extra in ``pyproject.toml``. The only hard runtime dependency is PyYAML.

Example (OpenAI chat, crunch + local cache)::

    from openai import OpenAI
    from tokenclaw import crunch_openai, LocalCache

    client = OpenAI()
    cache = LocalCache()

    kwargs, report = crunch_openai(model="gpt-5", messages=messages)
    print("saved", report.chars_saved, "chars via", report.applied_rules)

    hit = cache.get(report)                       # keys on the crunched request
    if hit is not None:
        response = hit
    else:
        response = client.chat.completions.create(**kwargs).model_dump()
        cache.put(report, response)
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenclaw.cache import cache_key_for
from tokenclaw.crunch import crunch_body, estimate_tokens_from_text
from tokenclaw.paths import safe_expanduser
from tokenclaw.store import SQLiteStore, stable_json

__all__ = [
    "CrunchResult",
    "crunch_request",
    "crunch_openai",
    "LocalCache",
]

_OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
_OPENAI_RESPONSES_ENDPOINT = "/v1/responses"
_ANTHROPIC_ENDPOINT = "/v1/messages"

_SURFACE_BY_ENDPOINT = {
    _OPENAI_CHAT_ENDPOINT: "openai_chat_completions",
    _OPENAI_RESPONSES_ENDPOINT: "openai_responses",
    _ANTHROPIC_ENDPOINT: "anthropic_messages",
}


def _default_endpoint(provider: str) -> str:
    return _ANTHROPIC_ENDPOINT if provider == "anthropic" else _OPENAI_CHAT_ENDPOINT


def _source_surface(provider: str, endpoint: str) -> str:
    surface = _SURFACE_BY_ENDPOINT.get(endpoint)
    if surface:
        return surface
    return "anthropic_messages" if provider == "anthropic" else "openai_chat_completions"


def _managed_profile(threshold_chars: int | None) -> dict[str, Any] | None:
    if threshold_chars is None:
        return None
    return {"threshold_chars": int(threshold_chars), "policy_source": "library-override"}


@dataclass
class CrunchResult:
    """Outcome of crunching a single request.

    ``body`` is the crunched payload to send to the provider. The remaining fields
    summarize the savings; ``meta`` carries the full internal crunch metadata for
    power users who want per-rule detail.
    """

    body: dict[str, Any]
    changed: bool
    chars_before: int
    chars_after: int
    chars_saved: int
    crunch_ratio: float
    input_tokens_saved_est: int
    applied_rules: list[str]
    provider: str
    endpoint: str
    meta: dict[str, Any] = field(repr=False)


def crunch_request(
    body: dict[str, Any],
    *,
    provider: str = "openai",
    endpoint: str | None = None,
    threshold_chars: int | None = None,
) -> CrunchResult:
    """Crunch a raw provider request ``body`` and return a :class:`CrunchResult`.

    The input ``body`` is not mutated. ``provider`` is ``"openai"`` or
    ``"anthropic"``; ``endpoint`` defaults to the provider's primary route and also
    selects the crunch surface. ``threshold_chars`` overrides the minimum request
    size before compaction rules engage (whitespace/dedup always run).
    """
    if not isinstance(body, dict):
        raise TypeError("body must be a dict (a provider request payload)")
    provider = (provider or "openai").lower()
    endpoint = endpoint or _default_endpoint(provider)
    original = copy.deepcopy(body)
    crunched, meta = crunch_body(
        copy.deepcopy(body),
        provider=provider,
        source_surface=_source_surface(provider, endpoint),
        endpoint=endpoint,
        managed_profile=_managed_profile(threshold_chars),
    )
    chars_before = int(meta.get("before_chars") or len(stable_json(original)))
    chars_after = int(meta.get("after_chars") or len(stable_json(crunched)))
    chars_saved = max(0, chars_before - chars_after)
    ratio = meta.get("crunch_ratio")
    if ratio is None:
        ratio = (chars_saved / chars_before) if chars_before else 0.0
    tokens_saved = max(
        0,
        estimate_tokens_from_text(json.dumps(original))
        - estimate_tokens_from_text(json.dumps(crunched)),
    )
    applied_rules = [
        str(rule.get("rule_id"))
        for rule in meta.get("applied_rules", [])
        if isinstance(rule, dict) and rule.get("rule_id")
    ]
    return CrunchResult(
        body=crunched,
        changed=bool(meta.get("changed")),
        chars_before=chars_before,
        chars_after=chars_after,
        chars_saved=chars_saved,
        crunch_ratio=float(ratio),
        input_tokens_saved_est=tokens_saved,
        applied_rules=applied_rules,
        provider=provider,
        endpoint=endpoint,
        meta=meta,
    )


def crunch_openai(
    *,
    model: str | None = None,
    messages: list[Any] | None = None,
    input: Any | None = None,
    endpoint: str | None = None,
    threshold_chars: int | None = None,
    **passthrough: Any,
) -> tuple[dict[str, Any], CrunchResult]:
    """Crunch an OpenAI request and return ``(kwargs, result)``.

    Provide exactly one of ``messages=`` (Chat Completions) or ``input=``
    (Responses). ``kwargs`` is splat-ready for the OpenAI SDK::

        kwargs, report = crunch_openai(model="gpt-5", messages=messages, temperature=0)
        resp = client.chat.completions.create(**kwargs)

    Any extra keyword arguments (``temperature``, ``tools``, ...) pass through
    unchanged onto ``kwargs``.
    """
    if (messages is None) == (input is None):
        raise ValueError("provide exactly one of messages= (chat) or input= (responses)")
    if messages is not None:
        endpoint = endpoint or _OPENAI_CHAT_ENDPOINT
        payload_key = "messages"
        body: dict[str, Any] = {"messages": messages}
    else:
        endpoint = endpoint or _OPENAI_RESPONSES_ENDPOINT
        payload_key = "input"
        body = {"input": input}
    if model is not None:
        body["model"] = model
    result = crunch_request(body, provider="openai", endpoint=endpoint, threshold_chars=threshold_chars)
    kwargs: dict[str, Any] = dict(passthrough)
    if model is not None:
        kwargs["model"] = result.body.get("model", model)
    kwargs[payload_key] = result.body.get(payload_key)
    return kwargs, result


class LocalCache:
    """A local, server-free exact-match response cache backed by SQLite.

    Keys are derived from the (post-crunch) request body, endpoint, and provider —
    so cache the crunched request (pass a :class:`CrunchResult` or its ``.body``).
    Entries expire after a default TTL unless ``ttl_seconds`` is given.
    """

    def __init__(self, path: str = "~/.tokenclaw/library_cache.sqlite3") -> None:
        self.path = str(safe_expanduser(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteStore(self.path)

    def _resolve(
        self,
        body_or_result: "dict[str, Any] | CrunchResult",
        endpoint: str | None,
        provider: str | None,
    ) -> tuple[dict[str, Any], str, str]:
        if isinstance(body_or_result, CrunchResult):
            body = body_or_result.body
            endpoint = endpoint or body_or_result.endpoint
            provider = provider or body_or_result.provider
        else:
            body = body_or_result
            provider = provider or "openai"
            endpoint = endpoint or _default_endpoint(provider.lower())
        if not isinstance(body, dict):
            raise TypeError("expected a request dict or a CrunchResult")
        return body, endpoint, provider.lower()

    def key(
        self,
        body_or_result: "dict[str, Any] | CrunchResult",
        *,
        endpoint: str | None = None,
        provider: str | None = None,
        namespace: str | None = None,
    ) -> str:
        """Return the exact-match cache key for a request (advanced use)."""
        body, endpoint, provider = self._resolve(body_or_result, endpoint, provider)
        return cache_key_for(body, endpoint, provider=provider, namespace=namespace)

    def get(
        self,
        body_or_result: "dict[str, Any] | CrunchResult",
        *,
        endpoint: str | None = None,
        provider: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the cached response dict for a request, or ``None`` on a miss."""
        return self.store.get_cache(
            self.key(body_or_result, endpoint=endpoint, provider=provider, namespace=namespace)
        )

    def put(
        self,
        body_or_result: "dict[str, Any] | CrunchResult",
        response: dict[str, Any],
        *,
        endpoint: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        namespace: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """Store ``response`` for a request and return the cache key used."""
        body, endpoint, provider = self._resolve(body_or_result, endpoint, provider)
        cache_key = cache_key_for(body, endpoint, provider=provider, namespace=namespace)
        resolved_model = model or (body.get("model") if isinstance(body, dict) else None) or "unknown"
        self.store.set_cache(
            cache_key,
            str(resolved_model),
            len(stable_json(body)),
            response,
            ttl_seconds=ttl_seconds,
        )
        return cache_key

    def close(self) -> None:
        """Close the underlying SQLite connection (optional)."""
        conn = getattr(self.store, "conn", None)
        closer = getattr(conn, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
