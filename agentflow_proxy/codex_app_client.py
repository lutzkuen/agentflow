from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, NoReturn

import websockets
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = os.getenv("AGENTFLOW_CODEX_APP_URL", "ws://127.0.0.1:4013")
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("AGENTFLOW_CODEX_APP_TIMEOUT_SECONDS", "7200"))
DEFAULT_EFFORT = os.getenv("AGENTFLOW_CODEX_APP_EFFORT", "high")

_THREAD_SANDBOX = {
    "danger-full-access": "danger-full-access",
    "dangerfullaccess": "danger-full-access",
    "dangerFullAccess": "danger-full-access",
    "read-only": "read-only",
    "readonly": "read-only",
    "readOnly": "read-only",
    "workspace-write": "workspace-write",
    "workspacewrite": "workspace-write",
    "workspaceWrite": "workspace-write",
}
_TURN_SANDBOX_POLICY = {
    "danger-full-access": "dangerFullAccess",
    "dangerfullaccess": "dangerFullAccess",
    "dangerFullAccess": "dangerFullAccess",
    "read-only": "readOnly",
    "readonly": "readOnly",
    "readOnly": "readOnly",
    "workspace-write": "workspaceWrite",
    "workspacewrite": "workspaceWrite",
    "workspaceWrite": "workspaceWrite",
}
_APPROVAL_REQUEST_METHODS = {
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
}


class CodexAppError(RuntimeError):
    pass


class CodexAppTurnFailed(CodexAppError):
    pass


def _abs_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def thread_sandbox_mode(value: str) -> str:
    try:
        return _THREAD_SANDBOX[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported sandbox mode: {value}") from exc


def turn_sandbox_policy_type(value: str) -> str:
    try:
        return _TURN_SANDBOX_POLICY[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported sandbox mode: {value}") from exc


def _json_rpc_request(method: str, params: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    request_id = str(uuid.uuid4())
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return request_id, payload


def _json_rpc_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    return payload


def _format_error(error: Any) -> str:
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        data = error.get("data")
        parts = []
        if code is not None:
            parts.append(f"code={code}")
        if message:
            parts.append(str(message))
        if data:
            parts.append(json.dumps(data, ensure_ascii=False))
        return " ".join(parts) if parts else json.dumps(error, ensure_ascii=False)
    return str(error)


def _extract_agent_delta(params: Any) -> str:
    if not isinstance(params, dict):
        return ""
    for key in ("delta", "text", "content"):
        value = params.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_completed_agent_message(params: Any) -> str:
    if not isinstance(params, dict):
        return ""
    item = params.get("item")
    if not isinstance(item, dict):
        return ""
    if item.get("type") not in {"agentMessage", "assistantMessage"}:
        return ""
    for key in ("text", "message", "content"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _token_usage_line(params: Any) -> str | None:
    if not isinstance(params, dict):
        return None
    info = params.get("usage") if isinstance(params.get("usage"), dict) else params
    fields = {
        "input": info.get("inputTokens") or info.get("input_tokens"),
        "cached": info.get("cachedInputTokens") or info.get("cached_input_tokens"),
        "output": info.get("outputTokens") or info.get("output_tokens"),
        "reasoning": info.get("reasoningOutputTokens") or info.get("reasoning_output_tokens"),
    }
    present = [f"{key}={value}" for key, value in fields.items() if value is not None]
    if not present:
        return None
    return "token_usage " + " ".join(present)


def _rate_limit_line(params: Any) -> str | None:
    if not isinstance(params, dict):
        return None
    limits = params.get("rateLimits") or params.get("rate_limits") or params.get("limits")
    if not isinstance(limits, dict):
        return None
    plan = limits.get("planType") or limits.get("plan_type")
    primary = limits.get("primary") if isinstance(limits.get("primary"), dict) else {}
    secondary = limits.get("secondary") if isinstance(limits.get("secondary"), dict) else {}
    parts = []
    if plan:
        parts.append(f"plan={plan}")
    if primary.get("usedPercent") is not None:
        parts.append(f"primary_used={primary['usedPercent']}%")
    if secondary.get("usedPercent") is not None:
        parts.append(f"secondary_used={secondary['usedPercent']}%")
    return "rate_limits " + " ".join(parts) if parts else None


class CodexAppClient:
    def __init__(
        self,
        *,
        url: str,
        cwd: str,
        roots: list[str],
        model: str | None,
        effort: str,
        approval_policy: str,
        sandbox: str,
        auto_approve: bool,
        output_last_message: str | None,
    ) -> None:
        self.url = url
        self.cwd = _abs_path(cwd)
        self.roots = sorted(set([self.cwd, *[_abs_path(root) for root in roots]]))
        self.model = model
        self.effort = effort
        self.approval_policy = approval_policy
        self.thread_sandbox = thread_sandbox_mode(sandbox)
        self.turn_sandbox = turn_sandbox_policy_type(sandbox)
        self.auto_approve = auto_approve
        self.output_last_message = output_last_message
        self.agent_message_parts: list[str] = []
        self.completed_agent_message = ""

    async def run(self, prompt: str) -> int:
        started = time.time()
        async with websockets.connect(self.url, max_size=64 * 1024 * 1024) as websocket:
            await self._send_initialize(websocket)
            thread_id = await self._start_thread(websocket)
            turn_id = await self._start_turn(websocket, thread_id, prompt)
            status = await self._read_until_turn_completed(websocket, thread_id, turn_id)
        elapsed_ms = int((time.time() - started) * 1000)
        self._print(f"turn_completed status={status} elapsed_ms={elapsed_ms}")
        final_message = self.final_message
        if final_message:
            self._write_last_message(final_message)
            self._print("")
            self._print("final_message:")
            self._print(final_message)
        if status == "completed":
            return 0
        raise CodexAppTurnFailed(f"turn completed with status={status}")

    @property
    def final_message(self) -> str:
        if self.completed_agent_message:
            return self.completed_agent_message.strip()
        return "".join(self.agent_message_parts).strip()

    async def _send_initialize(self, websocket: Any) -> None:
        request_id, payload = _json_rpc_request(
            "initialize",
            {
                "clientInfo": {
                    "name": "agentflow-codex-app-client",
                    "title": "AgentFlow Codex App Client",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                    "requestAttestation": False,
                    "optOutNotificationMethods": [],
                },
            },
        )
        await websocket.send(json.dumps(payload))
        result = await self._wait_for_response(websocket, request_id, "initialize")
        server_info = result.get("serverInfo") if isinstance(result, dict) else None
        server_name = server_info.get("name") if isinstance(server_info, dict) else None
        self._print(f"initialized server={server_name or 'codex-app-server'}")
        await websocket.send(json.dumps(_json_rpc_notification("initialized")))

    async def _start_thread(self, websocket: Any) -> str:
        params: dict[str, Any] = {
            "cwd": self.cwd,
            "runtimeWorkspaceRoots": self.roots,
            "approvalPolicy": self.approval_policy,
            "sandbox": self.thread_sandbox,
        }
        if self.model:
            params["model"] = self.model
        request_id, payload = _json_rpc_request("thread/start", params)
        await websocket.send(json.dumps(payload))
        result = await self._wait_for_response(websocket, request_id, "thread/start")
        thread_id = None
        if isinstance(result, dict):
            thread_id = result.get("threadId")
            thread = result.get("thread")
            if thread_id is None and isinstance(thread, dict):
                thread_id = thread.get("id")
        if thread_id is None:
            raise CodexAppError(f"thread/start returned no threadId: {result!r}")
        thread_id = str(thread_id)
        model = result.get("model") or self.model or "default"
        self._print(f"thread_started thread_id={thread_id} model={model} cwd={self.cwd}")
        return thread_id

    async def _start_turn(self, websocket: Any, thread_id: str, prompt: str) -> str:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "clientUserMessageId": None,
            "input": [{"type": "text", "text": prompt, "textElements": []}],
            "approvalPolicy": self.approval_policy,
            "sandboxPolicy": {"type": self.turn_sandbox},
            "cwd": self.cwd,
            "runtimeWorkspaceRoots": self.roots,
            "effort": self.effort,
        }
        if self.model:
            params["model"] = self.model
        request_id, payload = _json_rpc_request("turn/start", params)
        await websocket.send(json.dumps(payload))
        result = await self._wait_for_response(websocket, request_id, "turn/start")
        turn_id = result.get("turnId") if isinstance(result, dict) else None
        self._print(f"turn_started turn_id={turn_id or 'unknown'}")
        return str(turn_id) if turn_id else ""

    async def _wait_for_response(self, websocket: Any, request_id: str, label: str) -> Any:
        while True:
            msg = await self._recv_message(websocket)
            if msg.get("id") == request_id:
                if msg.get("error") is not None:
                    raise CodexAppError(f"{label} failed: {_format_error(msg['error'])}")
                return msg.get("result")
            await self._handle_server_message(websocket, msg)

    async def _read_until_turn_completed(self, websocket: Any, thread_id: str, turn_id: str) -> str:
        del thread_id, turn_id
        while True:
            msg = await self._recv_message(websocket)
            status = await self._handle_server_message(websocket, msg)
            if status is not None:
                return status

    async def _recv_message(self, websocket: Any) -> dict[str, Any]:
        raw = await websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CodexAppError(f"received non-JSON app-server message: {raw[:200]!r}") from exc
        if not isinstance(msg, dict):
            raise CodexAppError(f"received non-object app-server message: {msg!r}")
        return msg

    async def _handle_server_message(self, websocket: Any, msg: dict[str, Any]) -> str | None:
        method = msg.get("method")
        params = msg.get("params")
        if method is None:
            return None
        if msg.get("id") is not None:
            await self._handle_server_request(websocket, msg, str(method))
            return None
        if method == "item/agentMessage/delta":
            delta = _extract_agent_delta(params)
            if delta:
                self.agent_message_parts.append(delta)
            return None
        if method == "item/completed":
            completed = _extract_completed_agent_message(params)
            if completed:
                self.completed_agent_message = completed
            return None
        if method in {"turn/completed", "turn/failed"}:
            status = "completed"
            if isinstance(params, dict) and params.get("status"):
                status = str(params["status"])
            elif method == "turn/failed":
                status = "failed"
            return status
        if method in {"notification/configureSession", "notification/warning"}:
            message = params.get("message") if isinstance(params, dict) else None
            if message:
                self._print(f"warning: {message}")
            return None
        usage = _token_usage_line(params) if "token" in str(method).lower() or "usage" in str(method).lower() else None
        if usage:
            self._print(usage)
            return None
        rate_limits = _rate_limit_line(params) if "rate" in str(method).lower() or "limit" in str(method).lower() else None
        if rate_limits:
            self._print(rate_limits)
        return None

    async def _handle_server_request(self, websocket: Any, msg: dict[str, Any], method: str) -> None:
        request_id = msg["id"]
        if method in _APPROVAL_REQUEST_METHODS:
            decision = "accept" if self.auto_approve else "cancel"
            self._print(f"approval_request method={method} decision={decision}")
            response = {"jsonrpc": "2.0", "id": request_id, "result": {"decision": decision}}
        else:
            self._print(f"server_request method={method} decision=method_not_found")
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unsupported app-server request: {method}"},
            }
        await websocket.send(json.dumps(response))

    def _write_last_message(self, message: str) -> None:
        if not self.output_last_message:
            return
        path = Path(self.output_last_message)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message + "\n", encoding="utf-8")

    def _print(self, message: str) -> None:
        print(message, flush=True)


async def _run(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt.strip():
        raise CodexAppError("empty prompt")
    client = CodexAppClient(
        url=args.url,
        cwd=args.cd,
        roots=args.add_dir,
        model=args.model,
        effort=args.effort,
        approval_policy=args.approval_policy,
        sandbox=args.sandbox,
        auto_approve=args.auto_approve_server_requests,
        output_last_message=args.output_last_message,
    )
    print(f"app_server_url={args.url}", flush=True)
    return await asyncio.wait_for(client.run(prompt), timeout=args.timeout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Codex turn through the Codex app-server JSON-RPC protocol")
    parser.add_argument("prompt", nargs="?", help="Prompt text. Reads stdin when omitted.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Codex app-server websocket URL (default: {DEFAULT_URL})")
    parser.add_argument("--cd", default=os.getcwd(), help="Working directory for the Codex turn")
    parser.add_argument("--add-dir", action="append", default=[], help="Additional runtime workspace root")
    parser.add_argument("--model", default=os.getenv("AGENTFLOW_CODEX_APP_MODEL") or None, help="Optional explicit model. Omit to use Codex app-server default.")
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high"], help="Reasoning effort")
    parser.add_argument("--approval-policy", default=os.getenv("AGENTFLOW_CODEX_APPROVAL", "never"))
    parser.add_argument("--sandbox", default=os.getenv("AGENTFLOW_CODEX_SANDBOX", "danger-full-access"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-last-message", help="Write the final agent message to this file")
    parser.add_argument(
        "--auto-approve-server-requests",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AGENTFLOW_CODEX_APP_AUTO_APPROVE", "1") != "0",
        help="Accept app-server command/file approval requests. Disable with AGENTFLOW_CODEX_APP_AUTO_APPROVE=0.",
    )
    return parser


def _exit(code: int) -> NoReturn:
    raise SystemExit(code)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _exit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        _exit(130)
    except asyncio.TimeoutError:
        print(f"Codex app-server turn timed out after {args.timeout}s", file=sys.stderr)
        _exit(124)
    except CodexAppTurnFailed as exc:
        print(str(exc), file=sys.stderr)
        _exit(1)
    except CodexAppError as exc:
        print(f"Codex app-server client error: {exc}", file=sys.stderr)
        _exit(1)


if __name__ == "__main__":
    main()
