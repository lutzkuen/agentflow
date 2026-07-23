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

import math
import os
import random
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


def recent_tool_use_names(body: Any) -> list[str]:
    """Lowercased tool_use names from the most recent assistant message that has
    any. Returns [] when no assistant turn in the transcript issued a tool call.

    We read *actual* tool_use blocks, not the declared ``tools`` array. The
    declared array is a surface/config fingerprint (it clusters at 0 / 28 / ~100
    by MCP-server count and is constant per surface); it says nothing about what
    this turn is doing. The names the model actually emitted do.
    """
    if not isinstance(body, dict):
        return []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        names = [
            str(block.get("name")).lower()
            for block in _iter_content_blocks(msg.get("content"))
            if block.get("type") == "tool_use" and block.get("name")
        ]
        if names:
            return names
    return []


def classify_eligibility(body: Any, category: Optional[str], cfg: "DownrouteConfig") -> Eligibility:
    """Decide whether this turn is a read-only tool-heavy turn we may downroute.

    Gate: the proxy-computed ``category`` must be in the operator-configured
    eligible set (default {"tool-heavy"}), AND the recent assistant tool_use
    names must be non-empty and all read-only. Requiring >=1 recent read-only
    tool_use tightens the population to genuine mechanical read trajectories and
    excludes first/planning turns (the interpretation/judgment turns that must
    stay on the frontier model).
    """
    if category not in cfg.eligible_categories:
        return Eligibility(False, f"category:{category}")
    names = recent_tool_use_names(body)
    if not names:
        return Eligibility(False, "no-recent-tool-use")
    non_read = sorted({n for n in names if n not in READ_ONLY_TOOLS})
    if non_read:
        return Eligibility(False, "mutating-or-unknown:" + ",".join(non_read)[:120], tuple(names))
    return Eligibility(True, "read-only-tool-heavy", tuple(names))


# --- Pockets: requested-family -> target-family -----------------------------

# Cross-provider routing is deliberately out of scope; these are Anthropic tiers
# only. OpenAI terra->luna is a deferred candidate, intentionally absent.
POCKET_TARGET_FAMILY = {
    "opus": "sonnet",
    "sonnet": "haiku",
}


def pocket_for(requested_model: str | None) -> Optional[tuple[str, str]]:
    """(requested_family, target_family) for a downroute-eligible model, else None."""
    fam = _model_family(requested_model)
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


def extract_tool_targets(response_json: Any) -> set[str]:
    """Normalized ``key=value`` target strings from the tool_use blocks a
    response emitted. Empty set when the response issued no tool call."""
    targets: set[str] = set()
    for block in _tool_use_blocks(_response_content(response_json)):
        inp = block.get("input")
        if not isinstance(inp, dict):
            continue
        for key in _TARGET_ARG_KEYS:
            value = inp.get(key)
            if isinstance(value, str) and value.strip():
                targets.add(f"{key}={value.strip()}")
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
