# AgentFlow

AgentFlow is a **local proxy for token savings and telemtry on LLM traffic**.

Run it on localhost, point OpenAI-compatible or Anthropic-compatible clients at it, and keep using your normal provider credentials. AgentFlow forwards the real provider call, records local usage metadata, and shows cost and traffic behavior in a read-only dashboard.
It will apply crunching, caching and routing to decrease your token spend while preserving quality.

By default, AgentFlow does **not** store raw prompts or responses.

## How does it work?

AgentFlow sits between tools such as Codex, Claude Code, VS Code extensions, or your own API app and the provider API.

```text
your app / IDE plugin -> AgentFlow on localhost -> OpenAI or Anthropic
```

It currently supports:

| Provider mode | Routes |
| --- | --- |
| Anthropic / Claude-compatible | `POST /v1/messages` |
| OpenAI-compatible | `POST /v1/responses`, `POST /v1/chat/completions`, WebSocket `/v1/responses`, plus files/uploads passthrough |
| Codex app-server | Experimental proxy/telemetry path |

The OpenAI and Anthropic proxies run as separate provider modes. Run two AgentFlow processes if you want both at the same time.

## How does it help me?

Agentflow reduces your LLM spend.

Beyond that AgentFlow gives you local visibility into coding-agent traffic:

- estimated tokens, spend, savings, and recent activity
- which app, session, provider, and model generated calls
- routing, prompt-crunching, cache, retry, backoff, and error decisions
- policy state and whether local policy files need reload
- metadata-only Codex app-server telemetry

It is meant for answering "where did the tokens and cost go?" without sending prompts or responses to another service.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/lutzkuen/agentflow.git
cd agentflow

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Smoke-test the install:

```bash
agentflow --help
```

The public onboarding command is `agentflow`. The Python distribution is still
published from this repository as `agentflow-proxy`; reserving or migrating the
PyPI package name to `agentflow` is a release task. Existing specialist scripts
such as `agentflow-proxy` remain available for compatibility.

Run tests:

```bash
python -m unittest discover -s tests
```

## Start AgentFlow

### Claude / Anthropic-compatible proxy

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
agentflow-proxy --provider anthropic --host 127.0.0.1 --port 4000
```

This serves:

```text
http://127.0.0.1:4000/v1/messages
http://127.0.0.1:4000/agentflow/dashboard
```

### OpenAI-compatible proxy

```bash
export OPENAI_API_KEY="sk-..."
agentflow-proxy --provider openai --openai-auth-mode proxy --host 127.0.0.1 --port 4003
```

This serves:

```text
http://127.0.0.1:4003/v1/responses
http://127.0.0.1:4003/v1/chat/completions
http://127.0.0.1:4003/agentflow/dashboard
```

`--openai-auth-mode proxy` means AgentFlow uses `AGENTFLOW_OPENAI_API_KEY` or `OPENAI_API_KEY` from the proxy environment when forwarding upstream. Use `--openai-auth-mode client` if you want each client request's `Authorization` header to be forwarded.

For a custom OpenAI-compatible provider, keep the client base URL pointed at
AgentFlow and configure the upstream provider URL separately:

```bash
agentflow activate openai \
  --openai-base-url 'https://resource.openai.azure.com/openai/deployments/my-deployment?api-version=2024-10-21' \
  --openai-auth-mode proxy

agentflow run openai
```

In that example:

- `http://127.0.0.1:4003/v1` is the local AgentFlow base URL your OpenAI SDK or tool uses.
- `https://resource.openai.azure.com/openai/deployments/my-deployment?...` is the upstream provider base URL AgentFlow forwards to.
- `--openai-auth-mode proxy` uses credentials from the AgentFlow process environment. `--openai-auth-mode client` forwards each client's `Authorization` header instead.

AgentFlow preserves upstream path and query components such as Azure `api-version`,
avoids duplicate `/v1` route segments, and redacts userinfo or sensitive query
values in CLI and health output.

## Point an existing app at AgentFlow

### Existing OpenAI API app

Change the OpenAI base URL to:

```text
http://127.0.0.1:4003/v1
```

Keep the same models and API key. Examples:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="http://127.0.0.1:4003/v1",
)
```

```js
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://127.0.0.1:4003/v1",
});
```

```bash
curl http://127.0.0.1:4003/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.5","input":"Reply with ok"}'
```

### Existing Claude / Anthropic API app

Change the Anthropic base URL to:

```text
http://127.0.0.1:4000
```

Keep the same `x-api-key` / `Authorization` and `anthropic-version` headers.

```python
import os
from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="http://127.0.0.1:4000",
)
```

```bash
curl http://127.0.0.1:4000/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4.5","max_tokens":32,"messages":[{"role":"user","content":"Reply with ok"}]}'
```

## Run VS Code so Codex and Claude use AgentFlow

VS Code extensions usually read environment variables from the VS Code process. If you open VS Code from the desktop launcher, it may not know about variables you exported in a terminal.

Use this pattern:

1. Start AgentFlow in one terminal and leave it running.
2. Open a second terminal.
3. In the second terminal, export the variables the extension needs.
4. Launch VS Code from that same second terminal with `code .`.

`code .` means "open the current folder in VS Code." Because VS Code was started from that shell, its extensions can inherit that shell's environment.

### Codex VS Code extension

Put the OpenAI proxy base URL in your **user-level** Codex config:

```toml
# ~/.codex/config.toml
openai_base_url = "http://127.0.0.1:4003/v1"
```

Then launch VS Code from a shell that has your OpenAI key:

```bash
export OPENAI_API_KEY="sk-..."
code .
```

For a one-off CLI check:

```bash
codex exec --config 'openai_base_url="http://127.0.0.1:4003/v1"' "Reply with ok"
```

Do not put this provider setting in a project `.codex/config.toml`; Codex treats provider/auth settings as machine-local.

### Claude / Claude Code VS Code integration

Run the Anthropic proxy, then launch VS Code from a shell with the Claude proxy environment:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_API_KEY"
code .
```

If your extension does not inherit shell environment variables, configure the same variables in that extension's environment/wrapper. Keep secrets in your user environment, not in repository files.

## Dashboard

When a proxy is running, open the dashboard from that proxy:

```text
http://127.0.0.1:4000/agentflow/dashboard
http://127.0.0.1:4003/agentflow/dashboard
```

Or run a dashboard-only process that reads the same local database:

```bash
agentflow-dashboard --host 127.0.0.1 --port 4002
```

Then open:

```text
http://127.0.0.1:4002/agentflow/dashboard
```

The dashboard tells you:

- recent calls and sessions
- estimated tokens, cost, and savings
- usage by provider, model, app, and session when labels are available
- cache hits/misses/skips
- routing and prompt-crunch decisions
- retries, errors, rate-limit/backoff state
- active local policies and reload status
- Codex app-server telemetry when used

## Codex app-server telemetry

For Codex OAuth/subscription flows, the OpenAI-compatible base URL path may not be the right fit. AgentFlow also includes an experimental app-server relay:

```bash
codex app-server --listen ws://127.0.0.1:4014
agentflow-codex-app-proxy --host 127.0.0.1 --port 4013 --upstream ws://127.0.0.1:4014
printf 'Reply with exactly: ok\n' | agentflow-codex-app-client --url ws://127.0.0.1:4013 --cd "$PWD"
```

This path focuses on telemetry. It records redacted JSON-RPC method names and size-derived metadata, not raw prompts by default.

Inspect recent Codex telemetry:

```bash
agentflow-codex-diagnose --db ~/.agentflow/agentflow.sqlite3 --pretty
```

## Defaults and privacy

- Local database: `~/.agentflow/agentflow.sqlite3`
- Prompt/response body logging: off by default
- Enable raw body logging only for local debugging:

```bash
export AGENTFLOW_LOG_BODIES=1
```

- Keep proxy ports on `127.0.0.1` unless you add your own network/auth boundary.
- AgentFlow is not an authentication gateway.
- Cost and token numbers are estimates unless the upstream provider returns exact usage.
- If AgentFlow is unsure about routing, crunching, or caching, it forwards the request unchanged and records why.

## Common configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENTFLOW_DB` | `~/.agentflow/agentflow.sqlite3` | Local SQLite metadata DB |
| `AGENTFLOW_DATABASE_URL` | unset | Use Postgres instead of SQLite |
| `AGENTFLOW_CACHE` | `1` | Enable exact local cache where safe |
| `AGENTFLOW_CACHE_TOOL_CALLS` | `0` | Cache tool-call requests only if you accept the risk |
| `AGENTFLOW_ROUTING` | `1` | Enable local model routing |
| `AGENTFLOW_LOG_BODIES` | `0` | Store raw request/response bodies |
| `AGENTFLOW_HOST` | `0.0.0.0` | Proxy host default |
| `AGENTFLOW_PORT` | `4000` | Proxy port default |
| `AGENTFLOW_DASHBOARD_HOST` | `0.0.0.0` | Standalone dashboard host |
| `AGENTFLOW_DASHBOARD_PORT` | `4002` | Standalone dashboard port |

## Advanced policy and diagnostics

The README keeps the happy path short. Advanced local policy review/apply/rollback, managed recommendation bridge, replayability reports, routing experiments, and promotion diagnostics are available as CLI entry points installed by `pip install -e .`.

List them with:

```bash
python - <<'PY'
import importlib.metadata as md
for ep in md.entry_points(group="console_scripts"):
    if ep.name.startswith("agentflow-"):
        print(ep.name)
PY
```
