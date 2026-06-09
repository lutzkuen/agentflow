from __future__ import annotations

import re
from typing import Any


TERMINAL_LOG_FEATURE_SCHEMA = "agentflow.terminal_log_features.v1"

_TIMESTAMP_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?"
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r")"
)
_LOG_LEVEL_RE = re.compile(r"\b(?:TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|ERR|FATAL|CRITICAL)\b")
_PID_PREFIX_RE = re.compile(r"\b(?:pid|process|thread|tid)[=:]?\d+\b|\[\d{2,6}\]", re.IGNORECASE)
_SHELL_PROMPT_RE = re.compile(
    r"^\s*(?:[$#>]\s+\S|(?:[\w.\-]+@[\w.\-]+:.*?[$#]\s+\S)|PS\s+[^>]+>\s+\S|[A-Za-z]:\\[^>]+>\s+\S|\+\s+\S)"
)
_STD_STREAM_RE = re.compile(r"^\s*(?:stdout|stderr|STDOUT|STDERR)\s*[:|]\s+")
_STACK_TRACE_RE = re.compile(
    r"^\s*(?:Traceback \(most recent call last\):|File \".+\", line \d+|at\s+[\w.$<>]+\(.+:\d+(?::\d+)?\)|Caused by:|panic:|Exception in thread)"
)
_TEST_OUTPUT_RE = re.compile(
    r"^\s*(?:=+\s*(?:FAILURES|short test summary info|test session starts)\s*=+|FAILED\s+\S+|FAIL:\s+|ERROR:\s+|Ran\s+\d+\s+tests?|"
    r"(?:\d+\s+)?(?:passed|failed|skipped|xfailed|xpassed|error)s?\b|AssertionError\b|FAIL\s+\S+)",
    re.IGNORECASE,
)
_BUILD_OUTPUT_RE = re.compile(
    r"^\s*(?:(?:error|warning)\s+TS\d+:|(?:cc|gcc|g\+\+|clang|rustc|go|cargo|make|cmake|mvn|gradle)\b|"
    r"\S+\.(?:c|cc|cpp|h|hpp|go|rs|java|js|jsx|ts|tsx|py):\d+(?::\d+)?:\s*(?:error|warning)|"
    r"(?:Build failed|Compilation failed|SyntaxError:|TypeError:|ReferenceError:))",
    re.IGNORECASE,
)
_PACKAGE_OUTPUT_RE = re.compile(
    r"^\s*(?:npm\s+(?:ERR!|WARN|notice)|pnpm\s+(?:ERR!|WARN)|yarn\s+(?:error|warning)|pip\s+install\b|"
    r"Collecting\s+\S+|Requirement already satisfied|Successfully installed|added\s+\d+\s+packages?)",
    re.IGNORECASE,
)
_SERVER_LOG_RE = re.compile(
    r"^\s*(?:INFO:\s+\d{1,3}(?:\.\d{1,3}){3}:\d+\s+-|(?:GET|POST|PUT|PATCH|DELETE)\s+\S+\s+HTTP/\d|"
    r"(?:uvicorn|gunicorn|docker|containerd|kubectl|nginx|apache)\b)",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(r"\b(?:ERROR|ERR|FATAL|CRITICAL|Exception|Traceback|AssertionError|failed|failure|panic:)\b", re.IGNORECASE)
_CODE_CONTAINER_RE = re.compile(
    r"^\s*(?:(?:const|let|var|if|for|while|return|def|class|print|console\.log|logger\.)\b|"
    r"[\w.]+\s*=\s*[rfbu]?[\"']|.*[\"']\s*[);]?\s*$)"
)

_CODE_FENCE_RE = re.compile(r"^\s*```\s*([A-Za-z0-9_+.-]*)")
_TERMINAL_FENCE_LANGS = {"", "text", "txt", "log", "logs", "console", "terminal", "shell", "bash", "sh", "zsh", "powershell", "ps1"}


def _fraction_bucket(numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator <= 0:
        return "none"
    ratio = numerator / denominator
    if ratio < 0.10:
        return "lt_10pct"
    if ratio < 0.25:
        return "10_25pct"
    if ratio < 0.50:
        return "25_50pct"
    if ratio < 0.75:
        return "50_75pct"
    return "gte_75pct"


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "two_three"
    if value <= 10:
        return "four_ten"
    return "gte_11"


def _looks_like_code_container(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _SHELL_PROMPT_RE.search(stripped):
        return False
    if _TIMESTAMP_PREFIX_RE.search(stripped):
        return False
    if _STD_STREAM_RE.search(stripped):
        return False
    if _CODE_CONTAINER_RE.search(stripped) and ("\"" in stripped or "'" in stripped):
        return True
    return False


def _line_classes(line: str) -> set[str]:
    if not line.strip() or _looks_like_code_container(line):
        return set()
    classes: set[str] = set()
    if _SHELL_PROMPT_RE.search(line):
        classes.add("command_transcript")
    if _STD_STREAM_RE.search(line):
        classes.add("stdio_stream")
    if _TIMESTAMP_PREFIX_RE.search(line):
        classes.add("timestamp_prefix")
    if _LOG_LEVEL_RE.search(line) and (_TIMESTAMP_PREFIX_RE.search(line) or _PID_PREFIX_RE.search(line) or ":" in line):
        classes.add("log_line")
    if _PID_PREFIX_RE.search(line) and (_LOG_LEVEL_RE.search(line) or _TIMESTAMP_PREFIX_RE.search(line)):
        classes.add("pid_prefix")
    if _STACK_TRACE_RE.search(line):
        classes.add("stack_trace")
    if _TEST_OUTPUT_RE.search(line):
        classes.add("test_output")
    if _BUILD_OUTPUT_RE.search(line):
        classes.add("build_output")
    if _PACKAGE_OUTPUT_RE.search(line):
        classes.add("package_output")
    if _SERVER_LOG_RE.search(line):
        classes.add("server_runtime_log")
    if _ERROR_RE.search(line) and (classes or _STACK_TRACE_RE.search(line)):
        classes.add("error_line")
    if "timestamp_prefix" in classes:
        classes.add("log_line")
    if classes:
        classes.add("terminal_output")
    return classes


def _prefix_key(classes: set[str]) -> str | None:
    keys = [
        key
        for key in (
            "timestamp_prefix",
            "log_line",
            "pid_prefix",
            "stdio_stream",
            "server_runtime_log",
            "package_output",
            "build_output",
            "test_output",
            "command_transcript",
        )
        if key in classes
    ]
    return "+".join(keys) if keys else None


def terminal_log_features_from_text(text: str | None) -> dict[str, Any]:
    """Return metadata-only terminal/log composition features for managed policy inputs."""
    safe_text = text if isinstance(text, str) else ""
    total_chars = len(safe_text)
    lines = safe_text.splitlines()
    total_lines = len([line for line in lines if line.strip()])
    terminal_chars = 0
    log_lines = 0
    timestamp_lines = 0
    error_lines = 0
    prefix_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    unique_error_shapes: set[str] = set()
    stack_trace_present = False
    test_output_present = False
    command_transcript_present = False

    in_fence = False
    fence_terminal_like = True
    for line in lines:
        fence = _CODE_FENCE_RE.match(line)
        if fence:
            if not in_fence:
                lang = (fence.group(1) or "").strip().lower()
                in_fence = True
                fence_terminal_like = lang in _TERMINAL_FENCE_LANGS
            else:
                in_fence = False
                fence_terminal_like = True
            continue
        if in_fence and not fence_terminal_like:
            continue

        classes = _line_classes(line)
        if not classes:
            continue
        for class_name in classes:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        terminal_chars += len(line) + 1
        log_lines += 1 if "log_line" in classes else 0
        timestamp_lines += 1 if "timestamp_prefix" in classes else 0
        if "error_line" in classes:
            error_lines += 1
            shape = re.sub(r"\d+", "N", line.strip())
            shape = re.sub(r"0x[0-9a-fA-F]+", "0xN", shape)
            shape = re.sub(r"['\"].*?['\"]", "Q", shape)
            unique_error_shapes.add(shape[:120])
        stack_trace_present = stack_trace_present or "stack_trace" in classes
        test_output_present = test_output_present or "test_output" in classes
        command_transcript_present = command_transcript_present or "command_transcript" in classes
        key = _prefix_key(classes)
        if key:
            prefix_counts[key] = prefix_counts.get(key, 0) + 1

    repeated_prefix_patterns = sum(1 for count in prefix_counts.values() if count >= 2)
    return {
        "schema": TERMINAL_LOG_FEATURE_SCHEMA,
        "detector_version": "2026-06-09.1",
        "terminal_output_char_fraction_bucket": _fraction_bucket(terminal_chars, total_chars),
        "log_line_fraction_bucket": _fraction_bucket(log_lines, total_lines),
        "stack_trace_present": stack_trace_present,
        "error_line_count_bucket": _count_bucket(error_lines),
        "timestamp_prefix_line_fraction_bucket": _fraction_bucket(timestamp_lines, total_lines),
        "repeated_log_prefix_pattern_count_bucket": _count_bucket(repeated_prefix_patterns),
        "test_output_present": test_output_present,
        "command_transcript_present": command_transcript_present,
        "unique_error_signature_count_bucket": _count_bucket(len(unique_error_shapes)),
        "class_count_buckets": {
            key: _count_bucket(class_counts.get(key, 0))
            for key in (
                "command_transcript",
                "stdio_stream",
                "log_line",
                "stack_trace",
                "test_output",
                "build_output",
                "package_output",
                "server_runtime_log",
            )
        },
        "privacy": {
            "metadata_only": True,
            "raw_terminal_text_included": False,
            "raw_log_text_included": False,
            "raw_command_text_included": False,
            "raw_stack_trace_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
        },
    }
