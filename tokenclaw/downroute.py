"""Per-pocket probabilistic downrouting of read-only tool-heavy turns.

This is the lowest-priority *local* routing dial. It sits below manual hard
rules, managed experiments, and managed session tiers: it only ever fires when
none of those moved the model, and only on turns whose recent tool activity is
purely read-only (Read/Grep/Glob-class). A "pocket" is a requested-family ->
target-family swap (opus->sonnet, sonnet->haiku); each pocket carries its own
downroute fraction ``f``.

The module is intentionally pure: no store, no proxy, no network. It exposes the
classifier, the seeded coin-flip, the pocket map, the Wilson-bounded AIMD
controller law, and the tool-target extractors the harm-verdict pass needs to
stitch a trajectory. State (per-pocket ``f`` and evidence counters) lives in the
store; the proxy and store call into these functions.

Boundary note (CLAUDE.md "local applies / server learns"): a per-pocket scalar
tuned by a transparent local counter over local-only signals is a *calibrated
local dial* (a thermostat), not a learned feature policy. It is a single
inspectable number, operator-armable and operator-overridable, and defaults to
off. See CLAUDE.md "Calibrated local dials".
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .cache import _model_family


# --- Eligibility: read-only tool classification -----------------------------

# Names are matched case-insensitively. Only tools that cannot mutate state or
# cause side effects belong here. Anything not on this list (including unknown
# MCP tool names) is treated as ineligible -> the frontier model is kept. That
# conservative default is deliberate: we would rather never downroute a turn we
# cannot vouch for than guess.
READ_ONLY_TOOLS = frozenset(
    {
        "read",
        "grep",
        "glob",
        "ls",
        "notebookread",
        "webfetch",
        "websearch",
        "todoread",
        # OpenAI Responses built-in read-only tools, normalized from their
        # "*_call" item types (web_search_call -> web_search) so they match here.
        "web_search",
        "web_search_preview",
        "file_search",
    }
)

# Not consulted by the decision (unknown already fails the read-only test); kept
# for telemetry so an operator can see *why* a turn was rejected.
MUTATING_TOOLS = frozenset(
    {
        "edit",
        "write",
        "multiedit",
        "notebookedit",
        "bash",
        "str_replace_editor",
        "todowrite",
        "task",
    }
)


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str
    tool_names: tuple[str, ...] = ()


def _iter_content_blocks(content: Any) -> Iterable[dict]:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block


def _assistant_tool_names(msg: dict) -> list[str]:
    """Lowercased tool names emitted by one assistant message, across both the
    Anthropic ``content[].tool_use`` shape and the OpenAI chat/completions
    ``tool_calls[].function.name`` shape."""
    names: list[str] = []
    for block in _iter_content_blocks(msg.get("content")):
        if block.get("type") == "tool_use" and block.get("name"):
            names.append(str(block.get("name")).lower())
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
            if name:
                names.append(str(name).lower())
    return names


def _openai_responses_tool_names(input_items: list) -> list[str]:
    """Tool names from the current turn's activity in an OpenAI Responses
    ``input`` list. Scans in reverse, collecting ``function_call`` names and
    normalized built-in ``*_call`` item types (``web_search_call`` ->
    ``web_search``), and stops at the most recent user message so only this
    turn's tool activity counts. Unknown ``*_call`` types (e.g. ``computer_call``)
    are kept verbatim so the read-only test fails closed on them."""
    names: list[str] = []
    for item in reversed(input_items):
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message" and item.get("role") == "user":
            break
        if itype == "function_call":
            name = item.get("name")
            if name:
                names.append(str(name).lower())
        elif isinstance(itype, str) and itype.endswith("_call"):
            names.append(itype[: -len("_call")].lower())
    names.reverse()
    return names


def recent_tool_use_names(body: Any) -> list[str]:
    """Lowercased tool_use names from the most recent assistant tool activity.
    Returns [] when no assistant turn in the transcript issued a tool call.
    Handles all three provider request shapes: Anthropic ``messages`` (tool_use
    content blocks), OpenAI chat/completions ``messages`` (assistant
    ``tool_calls``), and the OpenAI Responses flat ``input`` list.

    We read *actual* tool calls, not the declared ``tools`` array. The declared
    array is a surface/config fingerprint (it clusters at 0 / 28 / ~100 by
    MCP-server count and is constant per surface); it says nothing about what
    this turn is doing. The names the model actually emitted do.
    """
    if not isinstance(body, dict):
        return []
    input_items = body.get("input")
    if isinstance(input_items, list):
        return _openai_responses_tool_names(input_items)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        names = _assistant_tool_names(msg)
        if names:
            return names
    return []


def classify_eligibility(
    body: Any,
    category: Optional[str],
    cfg: "DownrouteConfig",
    *,
    read_only_names: frozenset[str] | None = None,
) -> Eligibility:
    """Decide whether this turn is a read-only tool-heavy turn we may downroute.

    Gate: the proxy-computed ``category`` must be in the operator-configured
    eligible set (default {"tool-heavy"}), AND the recent assistant tool_use
    names must be non-empty and all read-only. Requiring >=1 recent read-only
    tool_use tightens the population to genuine mechanical read trajectories and
    excludes first/planning turns (the interpretation/judgment turns that must
    stay on the frontier model).

    ``read_only_names`` is the effective read-only allow-list; when omitted it is
    the code default ``READ_ONLY_TOOLS`` (keeps this function pure/store-free for
    tests). Seams pass ``effective_read_only_names(store)`` so an operator's
    dashboard override extends the set without a code change.
    """
    allow = READ_ONLY_TOOLS if read_only_names is None else read_only_names
    if category not in cfg.eligible_categories:
        return Eligibility(False, f"category:{category}")
    names = recent_tool_use_names(body)
    if not names:
        return Eligibility(False, "no-recent-tool-use")
    non_read = sorted({n for n in names if n not in allow})
    if non_read:
        return Eligibility(False, "mutating-or-unknown:" + ",".join(non_read)[:120], tuple(names))
    return Eligibility(True, "read-only-tool-heavy", tuple(names))


# pocket-independent cache of the operator-effective read-only allow-list; a few-
# second TTL keeps both downroute seams off a per-request DB read while an
# operator's dashboard toggle still takes effect within the TTL. Mirrors
# openai_proxy._openai_cached_pocket_f.
_EFFECTIVE_READ_ONLY_CACHE: dict[str, Any] = {"names": None, "expires": 0.0}
_EFFECTIVE_READ_ONLY_TTL_SECONDS = 5.0


def effective_read_only_names(store: Any) -> frozenset[str]:
    """The read-only allow-list after operator overrides: the code default plus
    tools the operator ticked read-only, minus defaults the operator un-ticked.
    Both directions honored. Store failure falls back to the code default."""
    now = time.monotonic()
    cached = _EFFECTIVE_READ_ONLY_CACHE
    if cached["names"] is not None and cached["expires"] > now:
        return cached["names"]
    try:
        override_map = store.read_only_override_map()
    except Exception:
        override_map = {}
    added = {n for n, v in override_map.items() if v}
    removed = {n for n, v in override_map.items() if not v}
    names = frozenset((READ_ONLY_TOOLS | added) - removed)
    cached["names"] = names
    cached["expires"] = now + _EFFECTIVE_READ_ONLY_TTL_SECONDS
    return names


def schedule_tool_sightings(store: Any, body: Any) -> None:
    """Fire-and-forget: persist the turn's observed tool-use names to the local
    tool catalog that backs the dashboard "Tool calls" tab. Called from the
    downroute seams for every turn (independent of downroute eligibility) so
    unknown and mutating tools surface for the operator too. Off the request hot
    path — the DB write runs in a worker thread — and never raises. Names stay
    local (never forwarded to the server), same privacy design as the dial.
    ``TOKENCLAW_TOOL_CATALOG=0`` is the kill-switch."""
    if not _env_bool("TOKENCLAW_TOOL_CATALOG", True):
        return
    try:
        names = recent_tool_use_names(body)
    except Exception:
        return
    if not names:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        try:
            store.record_tool_sightings(names)
        except Exception:
            pass
        return
    task = loop.create_task(asyncio.to_thread(store.record_tool_sightings, names))
    task.add_done_callback(_consume_sighting_task_exception)


def _consume_sighting_task_exception(task: "asyncio.Task[Any]") -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        return


# --- Pockets: requested-family -> target-family -----------------------------

# cache._model_family collapses every gpt-5.x tier to a single "gpt-5", which
# would make an OpenAI pocket a no-op self-swap (gpt-5 -> gpt-5). Downrouting
# needs the fine-grained ladder tier instead. These tokens map an OpenAI model
# string to its ladder tier; Codex is deliberately excluded (it is an action
# lane and fails closed on tool names anyway, and its family stays "codex").
_OPENAI_TIER_TOKENS = ("sol", "terra", "luna")


def _openai_tier(model: str | None) -> Optional[str]:
    """Fine-grained OpenAI ladder tier (sol/terra/luna/mini) for a model string,
    else None for non-OpenAI or Codex models. Bare current-gen ``gpt-5.6`` (no
    tier suffix) is the flagship, treated as ``sol`` — its adjacent-down is
    terra, matching routing_experiments._suggest_adjacent_routed_model."""
    if not model:
        return None
    m = str(model).lower()
    if "gpt-5" not in m:
        return None
    if "codex" in m:
        return None
    if "mini" in m or "nano" in m:
        return "mini"
    for tok in _OPENAI_TIER_TOKENS:
        if tok in m:
            return tok
    if "gpt-5.6" in m:
        return "sol"
    return None


def _downroute_family(model: str | None) -> Optional[str]:
    """Tier-aware family for downrouting. For OpenAI models this returns the
    ladder tier so gpt-5.6-terra and gpt-5.6-luna are DISTINCT pockets. For
    Anthropic models ``_openai_tier`` returns None, so this is byte-identical to
    ``_model_family`` and the committed Anthropic path is unchanged."""
    return _openai_tier(model) or _model_family(model)


# Cross-provider routing is deliberately out of scope; each pocket stays within
# one provider's ladder. OpenAI rungs mirror routing_experiments.
# _suggest_adjacent_routed_model: sol -> terra -> luna -> mini (rock bottom).
# "mini"/"haiku" have no entry: they are the floor and never downroute further.
POCKET_TARGET_FAMILY = {
    "opus": "sonnet",
    "sonnet": "haiku",
    "sol": "terra",
    "terra": "luna",
    "luna": "mini",
}


def pocket_for(requested_model: str | None) -> Optional[tuple[str, str]]:
    """(requested_family, target_family) for a downroute-eligible model, else None."""
    fam = _downroute_family(requested_model)
    if not fam:
        return None
    target = POCKET_TARGET_FAMILY.get(fam)
    if not target:
        return None
    return (fam, target)


def pocket_key(requested_family: str, target_family: str) -> str:
    return f"{requested_family}->{target_family}"


def resolve_target_model(target_family: str, tier_map: dict[str, str]) -> Optional[str]:
    """Concrete target model for a family, via an injected tier->model map
    (the proxy passes router._TIER_MAP). Injected rather than imported so this
    module stays pure and unit-testable without the router."""
    model = tier_map.get(target_family)
    return str(model) if model else None


# --- Seeded coin-flip -------------------------------------------------------

def decide_downroute(*, session_id: str | None, call_id: str | None, f: float) -> bool:
    """Deterministic per-call coin flip: reproducible from (session_id, call_id),
    uniform in aggregate at rate ``f``. Same call re-scored gives the same
    verdict, so a retried request is not re-diced onto a different model."""
    try:
        frac = float(f)
    except (TypeError, ValueError):
        return False
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    seed = f"{session_id or ''}:{call_id or ''}"
    return random.Random(seed).random() < frac


# --- AIMD controller law (Wilson-bounded) -----------------------------------

def wilson_bounds(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a proportion. Returns (lower, upper). For n==0
    returns (0.0, 1.0) — maximal uncertainty, which makes both controller gates
    (advance needs upper<=target, retreat needs lower>=target) decline."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class ControllerDecision:
    new_f: float
    action: str  # "advance" | "retreat" | "hold"
    reason: str
    reset_window: bool


def controller_step(
    *,
    f: float,
    window_applied: int,
    window_harm: int,
    cfg: "DownrouteConfig",
) -> ControllerDecision:
    """One AIMD step over a closed evidence window.

    Eager retreat, patient advance:
      - retreat (multiplicative, halve) as soon as the Wilson *lower* bound on
        the window harm-rate is at/above target with >= min_harm_sample applied.
        Being eager here means acting on the first statistically-real harm.
      - advance (additive, +step_up) only when the Wilson *upper* bound is
        at/below target with >= min_advance_sample applied. Being patient here
        means we do not creep up on a small, lucky-clean window.
      - otherwise hold and keep accumulating (window not reset).

    f is clamped to [f_min, f_max]. Disabled controller always holds.
    """
    if not cfg.controller_enabled:
        return ControllerDecision(f, "hold", "controller-disabled", False)
    lo, hi = wilson_bounds(window_harm, window_applied, cfg.z)
    if window_applied >= cfg.min_harm_sample and lo >= cfg.harm_target:
        new_f = max(cfg.f_min, round(f * cfg.decay, 4))
        return ControllerDecision(new_f, "retreat", f"harm-lower={lo:.3f}>=target", True)
    if window_applied >= cfg.min_advance_sample and hi <= cfg.harm_target:
        new_f = min(cfg.f_max, round(f + cfg.step_up, 4))
        return ControllerDecision(new_f, "advance", f"harm-upper={hi:.3f}<=target", True)
    return ControllerDecision(f, "hold", "accumulating", False)


# --- Tool-target extraction for harm stitching ------------------------------

# Argument keys whose string value names the resource a read-only tool touched.
# Two turns hitting the same target within the repair window is the workhorse
# harm signal: the frontier model re-doing a read the cheap model already did.
_TARGET_ARG_KEYS = (
    "file_path",
    "path",
    "pattern",
    "notebook_path",
    "url",
    "query",
    "glob",
)


def _tool_use_blocks(content: Any) -> Iterable[dict]:
    for block in _iter_content_blocks(content):
        if block.get("type") == "tool_use":
            yield block


def _response_content(response_json: Any) -> Any:
    """Pull the assistant content list out of a stored response, whether it was
    stored as the full Anthropic message ({"content": [...]}) or as a bare list."""
    if isinstance(response_json, dict):
        return response_json.get("content")
    if isinstance(response_json, list):
        return response_json
    return None


def _collect_target_args(inp: dict, targets: set[str]) -> None:
    for key in _TARGET_ARG_KEYS:
        value = inp.get(key)
        if isinstance(value, str) and value.strip():
            targets.add(f"{key}={value.strip()}")


def _loads_arguments(arguments: Any) -> Optional[dict]:
    """OpenAI tool-call arguments arrive as a JSON *string*. Parse defensively;
    a non-string or malformed payload yields no targets (fail closed)."""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return None
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _openai_targets_from_output_items(items: Any, targets: set[str]) -> None:
    """OpenAI Responses: output[] ``*_call`` items whose ``arguments`` is a JSON
    string. Built-in calls (web_search_call, ...) carry no _TARGET_ARG_KEYS and
    simply contribute nothing."""
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if not (isinstance(itype, str) and itype.endswith("_call")):
            continue
        inp = _loads_arguments(item.get("arguments"))
        if inp:
            _collect_target_args(inp, targets)


def _openai_targets_from_tool_calls(tool_calls: Any, targets: set[str]) -> None:
    """OpenAI chat/completions: choices[].message.tool_calls[].function.arguments
    (a JSON string)."""
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        arguments = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments")
        inp = _loads_arguments(arguments)
        if inp:
            _collect_target_args(inp, targets)


def extract_tool_targets(response_json: Any) -> set[str]:
    """Normalized ``key=value`` target strings from the tool calls a response
    emitted, across all provider response shapes (Anthropic ``content[].tool_use``,
    OpenAI Responses ``output[]``, OpenAI chat ``choices[].message.tool_calls``).
    Empty set when the response issued no tool call. Two turns hitting the same
    target within the repair window is the harm signal the finalize pass reads."""
    targets: set[str] = set()
    for block in _tool_use_blocks(_response_content(response_json)):
        inp = block.get("input")
        if isinstance(inp, dict):
            _collect_target_args(inp, targets)
    if isinstance(response_json, dict):
        _openai_targets_from_output_items(response_json.get("output"), targets)
        choices = response_json.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                    _openai_targets_from_tool_calls(choice["message"].get("tool_calls"), targets)
    return targets


def response_has_error_tool_result(request_body: Any) -> bool:
    """True when the last user message carries an is_error tool_result — a
    direct signal that a prior tool call failed."""
    if not isinstance(request_body, dict):
        return False
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for block in _iter_content_blocks(msg.get("content")):
            if block.get("type") == "tool_result" and block.get("is_error") is True:
                return True
        return False
    return False


# --- Configuration ----------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_categories(name: str, default: frozenset[str]) -> frozenset[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return frozenset(parts) if parts else default


@dataclass(frozen=True)
class DownrouteConfig:
    # f defaults to 0.0: an unseen or unarmed pocket downroutes nothing. Arming a
    # pocket sets its f to f_start. The controller (off by default) may step an
    # armed pocket's f within [f_min, f_max].
    f_default: float = 0.0
    f_start: float = 0.05
    f_min: float = 0.0
    f_max: float = 0.5
    controller_enabled: bool = False
    # The repair signal (a frontier turn redoing a downrouted read) is the
    # quality half of harm detection. On by default so verdicts and the operator
    # harm_repair_count are meaningful even before the controller is enabled;
    # settable off to fall back to error-only harm while validating precision.
    repair_signal_enabled: bool = True
    harm_target: float = 0.05
    step_up: float = 0.02
    decay: float = 0.5
    z: float = 1.96
    min_advance_sample: int = 50
    min_harm_sample: int = 20
    window_turns: int = 3
    verdict_ttl_seconds: int = 120
    eligible_categories: frozenset[str] = frozenset({"tool-heavy"})

    @classmethod
    def from_env(cls) -> "DownrouteConfig":
        return cls(
            f_default=_env_float("TOKENCLAW_DOWNROUTE_F_DEFAULT", 0.0),
            f_start=_env_float("TOKENCLAW_DOWNROUTE_F_START", 0.05),
            f_min=_env_float("TOKENCLAW_DOWNROUTE_F_MIN", 0.0),
            f_max=_env_float("TOKENCLAW_DOWNROUTE_F_MAX", 0.5),
            controller_enabled=_env_bool("TOKENCLAW_DOWNROUTE_CONTROLLER", False),
            repair_signal_enabled=_env_bool("TOKENCLAW_DOWNROUTE_REPAIR_SIGNAL", True),
            harm_target=_env_float("TOKENCLAW_DOWNROUTE_HARM_TARGET", 0.05),
            step_up=_env_float("TOKENCLAW_DOWNROUTE_STEP_UP", 0.02),
            decay=_env_float("TOKENCLAW_DOWNROUTE_DECAY", 0.5),
            z=_env_float("TOKENCLAW_DOWNROUTE_Z", 1.96),
            min_advance_sample=_env_int("TOKENCLAW_DOWNROUTE_MIN_ADVANCE_SAMPLE", 50),
            min_harm_sample=_env_int("TOKENCLAW_DOWNROUTE_MIN_HARM_SAMPLE", 20),
            window_turns=_env_int("TOKENCLAW_DOWNROUTE_WINDOW_TURNS", 3),
            verdict_ttl_seconds=_env_int("TOKENCLAW_DOWNROUTE_VERDICT_TTL", 120),
            eligible_categories=_env_categories(
                "TOKENCLAW_DOWNROUTE_CATEGORIES", frozenset({"tool-heavy"})
            ),
        )
