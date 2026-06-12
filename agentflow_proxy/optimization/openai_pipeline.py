from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agentflow_proxy.cache import cache_decision_meta
from agentflow_proxy.crunch import crunch_body, estimate_tokens_from_text
from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.optimization.managed_actions import (
    cache_profile_from_decision,
    crunch_profile_from_decision,
)
from agentflow_proxy.optimization.openai_features import (
    build_openai_outcome_feature_unit,
    build_openai_preflight_feature_unit,
    build_openai_request_feature_unit,
    openai_call_store_fields,
    summarize_openai_outcome_feature_unit,
    summarize_openai_request_feature_unit,
)
from agentflow_proxy.optimization.openai_recommendations import (
    apply_openai_recommendation_decision,
    fetch_openai_recommendation_decision,
)
from agentflow_proxy.recommendations import pattern_feature_diagnostics
from agentflow_proxy.router import categorize_request, extract_text, has_tools, route_openai_model


FetchPolicyDecision = Callable[..., Awaitable[dict[str, Any]]]
Cruncher = Callable[..., tuple[dict[str, Any], dict[str, Any]]]
Router = Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]
RecommendationApplier = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class OpenAIParsedRequest:
    stage: str
    body: dict[str, Any]
    stream: bool
    requested_model: str
    category: str | None
    text_chars: int
    input_tokens_est: int


@dataclass(frozen=True)
class OpenAIPreflightStage:
    stage: str
    parsed: OpenAIParsedRequest
    path: str
    routing_meta: dict[str, Any]
    feature_unit: dict[str, Any]
    feature_summary: dict[str, Any]
    pattern_features: dict[str, Any]


@dataclass(frozen=True)
class OpenAILocalPolicyStage:
    stage: str
    provider_body: dict[str, Any]
    resolved_requested_model: str
    local_routed_model: str
    routed_model: str
    input_tokens_est: int
    crunch_meta: dict[str, Any]
    routing_meta: dict[str, Any]
    managed_crunch_profile: dict[str, Any] | None
    managed_cache_profile: dict[str, Any] | None
    recommendation_meta: dict[str, Any]


def parse_openai_request_body(raw_body: dict[str, Any], default_models: list[str]) -> OpenAIParsedRequest:
    """Normalize local request metadata without building any server-bound payload."""
    requested_model = str(raw_body.get("model") or default_models[0])
    raw_body.setdefault("model", requested_model)
    text = extract_text(raw_body)
    return OpenAIParsedRequest(
        stage="parse_request",
        body=raw_body,
        stream=bool(raw_body.get("stream")),
        requested_model=requested_model,
        category=categorize_request(raw_body),
        text_chars=len(text),
        input_tokens_est=estimate_tokens_from_text(text),
    )


def extract_openai_preflight_features(parsed: OpenAIParsedRequest, *, path: str) -> OpenAIPreflightStage:
    """Create feature-only preflight metadata before local crunching or routing mutates the request."""
    routing_meta = {
        "enabled": False,
        "requested_model": parsed.requested_model,
        "routed_model": None,
        "reason": "preflight feature extraction before local mutation",
        "text_chars": parsed.text_chars,
        "has_tools": has_tools(parsed.body),
        "category": parsed.category,
        "policy_source": "preflight",
        "provider": "openai",
    }
    feature_unit = build_openai_preflight_feature_unit(
        body=parsed.body,
        path=path,
        requested_model=parsed.requested_model,
        routing_meta=routing_meta,
        category=parsed.category,
        stream=parsed.stream,
        input_tokens_est=parsed.input_tokens_est,
    )
    assert_managed_egress_safe(feature_unit)
    feature_summary = summarize_openai_request_feature_unit(feature_unit)
    pattern_features = pattern_feature_diagnostics(feature_unit)
    return OpenAIPreflightStage(
        stage="extract_preflight_features",
        parsed=parsed,
        path=path,
        routing_meta=routing_meta,
        feature_unit=feature_unit,
        feature_summary=feature_summary,
        pattern_features=pattern_features,
    )


async def fetch_openai_policy_decision(
    preflight: OpenAIPreflightStage,
    *,
    fetcher: FetchPolicyDecision = fetch_openai_recommendation_decision,
) -> dict[str, Any]:
    """Fetch a managed policy decision using only the guarded preflight feature unit."""
    assert_managed_egress_safe(preflight.feature_unit)
    return await fetcher(
        recommendation_unit=preflight.feature_unit,
        current_model=preflight.parsed.requested_model,
        input_tokens_est=preflight.parsed.input_tokens_est,
    )


def execute_openai_local_policy(
    *,
    raw_body: dict[str, Any],
    path: str,
    requested_model: str,
    category: str | None,
    stream: bool,
    session_id: str | None,
    preflight: OpenAIPreflightStage,
    policy_decision: dict[str, Any],
    store_obj: Any,
    cruncher: Cruncher = crunch_body,
    router: Router = route_openai_model,
    applier: RecommendationApplier = apply_openai_recommendation_decision,
) -> OpenAILocalPolicyStage:
    """Apply local crunch, route, cache-profile, and safe managed actions without provider I/O."""
    managed_crunch_profile = crunch_profile_from_decision(policy_decision)
    managed_cache_profile = cache_profile_from_decision(policy_decision)
    if managed_crunch_profile:
        provider_body, crunch_meta = cruncher(
            raw_body,
            store_obj=store_obj,
            managed_profile=managed_crunch_profile,
        )
    else:
        provider_body, crunch_meta = cruncher(raw_body, store_obj=store_obj)

    routed_model, routing_meta = router(provider_body)
    resolved_requested_model = str(provider_body.get("model") or requested_model)
    local_routed_model = str(routed_model)
    provider_body["model"] = routed_model
    input_tokens = estimate_tokens_from_text(extract_text(provider_body))
    cache_meta = cache_decision_meta("skipped", "not-evaluated")
    local_feature_unit = build_openai_request_feature_unit(
        body=provider_body,
        path=path,
        requested_model=resolved_requested_model,
        routed_model=str(provider_body.get("model") or routed_model),
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
        category=category,
        stream=stream,
        input_tokens_est=input_tokens,
        session_id=session_id,
    )
    local_pattern_features = pattern_feature_diagnostics(local_feature_unit)
    routing_meta["openai_feature_unit"] = preflight.feature_summary
    routing_meta["openai_preflight_unit"] = preflight.feature_summary
    routing_meta["openai_local_feature_unit"] = summarize_openai_request_feature_unit(local_feature_unit)
    routing_meta.update(openai_call_store_fields(
        path,
        resolved_requested_model,
        str(provider_body.get("model") or routed_model),
    ))
    routing_meta["openai_preflight_pattern_features"] = preflight.pattern_features
    routing_meta["managed_pattern_features"] = local_pattern_features
    if isinstance(policy_decision.get("local_actions"), dict):
        routing_meta["managed_local_actions"] = policy_decision["local_actions"]
    recommendation_meta = applier(
        body=provider_body,
        routing_meta=routing_meta,
        decision=policy_decision,
    )
    routing_meta["managed_recommendation"] = recommendation_meta
    return OpenAILocalPolicyStage(
        stage="execute_local_policy",
        provider_body=provider_body,
        resolved_requested_model=resolved_requested_model,
        local_routed_model=local_routed_model,
        routed_model=str(provider_body.get("model") or routed_model),
        input_tokens_est=input_tokens,
        crunch_meta=crunch_meta,
        routing_meta=routing_meta,
        managed_crunch_profile=managed_crunch_profile,
        managed_cache_profile=managed_cache_profile,
        recommendation_meta=recommendation_meta,
    )


def serialize_openai_outcome_summary(**kwargs: Any) -> dict[str, Any]:
    """Build guarded feature-only local outcome metadata for routing logs and managed feedback."""
    outcome_unit = build_openai_outcome_feature_unit(**kwargs)
    assert_managed_egress_safe(outcome_unit)
    return summarize_openai_outcome_feature_unit(outcome_unit)
