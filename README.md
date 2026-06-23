# TokenClaw

TokenClaw is a **local proxy for token savings and telemetry on LLM traffic**.

Run it on localhost, point OpenAI-compatible or Anthropic-compatible clients at it, and keep using your normal provider credentials. TokenClaw forwards the real provider call, records local usage metadata, and shows cost and traffic behavior in a read-only dashboard.
It will apply crunching, caching and routing to decrease your token spend while preserving quality.

By default, TokenClaw does **not** store raw prompts or responses.

## Quick start

Requires Python 3.10+.

The target packaged install is:

```bash
pip install tokenclaw
```

The default install is intentionally focused on the local proxy, dashboard,
local SQLite telemetry, and file-backed YAML rules. Advanced dependencies stay
behind optional extras:

```bash
pip install 'tokenclaw[managed]'          # Postgres-backed managed/workbench storage
pip install 'tokenclaw[compression]'      # zstd request-body decoding
pip install 'tokenclaw[openai-realtime]'  # OpenAI realtime WebSocket proxy bridge
pip install 'tokenclaw[all]'              # all optional runtime extras
```

Until the public package name is migrated, install from this repository:

```bash
git clone https://github.com/lutzkuen/tokenclaw.git
cd tokenclaw
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then start the local savings stack from one terminal:

```bash
tokenclaw --help
tokenclaw start
```

`tokenclaw start` creates default local OpenAI-compatible and Anthropic-compatible
activation profiles when they are missing, starts the local provider proxies when
their ports are available, and starts the read-only dashboard:

```text
OpenAI-compatible proxy: http://127.0.0.1:4003/v1
Anthropic-compatible proxy: http://127.0.0.1:4000
Dashboard: http://127.0.0.1:4002/tokenclaw/dashboard
```

Provider credentials stay in your existing shell, OS secret manager, or client
configuration. TokenClaw does not store or print API keys.

Use the public onboarding commands when you want to wire specific clients to those
local URLs:

```bash
tokenclaw activate openai
tokenclaw activate claude
tokenclaw activate codex
tokenclaw activate claude-vscode
tokenclaw activate claude-desktop
tokenclaw deactivate
tokenclaw stats
tokenclaw doctor
```

Activation writes local routing/config files only. It does not store or print API keys.
Keep provider credentials in your shell, OS secret manager, or the client that already
uses them.

Undo TokenClaw-managed activation profiles and local client routing hooks with:

```bash
tokenclaw deactivate
tokenclaw deactivate codex
tokenclaw deactivate claude-vscode
```

Start the configured proxies in separate terminals:

```bash
tokenclaw run openai
tokenclaw run claude
```

Use `--dry-run` to inspect the command without starting a server:

```bash
tokenclaw run openai --dry-run
tokenclaw run claude --dry-run
```

Open the read-only dashboard when a proxy or dashboard process is running:

```text
http://127.0.0.1:4002/tokenclaw/dashboard
```

## What activation changes

| Target | Command | What it changes | Local client URL | Runtime |
| --- | --- | --- | --- | --- |
| OpenAI API apps | `tokenclaw activate openai` | TokenClaw activation profile for OpenAI-compatible traffic | `http://127.0.0.1:4003/v1` | `tokenclaw run openai` |
| Claude API apps | `tokenclaw activate claude` | TokenClaw activation profile for Anthropic-compatible traffic | `http://127.0.0.1:4000` | `tokenclaw run claude` |
| Codex VS Code / CLI | `tokenclaw activate codex` | User-level `~/.codex/config.toml` `openai_base_url`; creates the OpenAI profile if needed | `http://127.0.0.1:4003/v1` | `tokenclaw run openai` |
| Claude VS Code / Claude Code | `tokenclaw activate claude-vscode` | TokenClaw-managed non-secret env file with `ANTHROPIC_BASE_URL`; creates the Claude profile if needed | `http://127.0.0.1:4000` | `tokenclaw run claude` |
| Claude Desktop on Linux | `tokenclaw activate claude-desktop` | User-level Claude Desktop `.desktop` launcher `Exec=` line with `ANTHROPIC_BASE_URL`; creates the Claude profile if needed | `http://127.0.0.1:4000` | `tokenclaw run claude` |
| GitHub Copilot | Unsupported | No base-url activation target | Not applicable | Not applicable |

The OpenAI and Anthropic proxies run as separate provider modes. Run two TokenClaw
processes if you want both at the same time.

## Interpret stats and doctor

```bash
tokenclaw stats
```

`tokenclaw stats` reads the local activation config and reports which targets are
configured, the local base URL each client should use, and the redacted upstream
provider URL TokenClaw will forward to. It does not contact provider APIs.

```bash
tokenclaw doctor
```

`tokenclaw doctor` checks configured targets without printing secrets. For provider
targets it checks the loopback health endpoint and detects common stale-config
states such as a running proxy pointed at a different upstream. For VS Code targets
it checks the local config files and reports when runtime environment inheritance
cannot be proven from the current shell.

Use JSON output for scripts:

```bash
tokenclaw stats --json
tokenclaw doctor --json
```

## OpenAI API apps

Configure the local OpenAI target:

```bash
tokenclaw activate openai
tokenclaw run openai
```

Change the OpenAI base URL in your app to:

```text
http://127.0.0.1:4003/v1
```

Keep the same models and API key handling your app already uses.

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

For a custom OpenAI-compatible provider, keep the client base URL pointed at
TokenClaw and configure the upstream provider URL separately:

```bash
tokenclaw activate openai \
  --openai-base-url 'https://resource.openai.azure.com/openai/deployments/my-deployment?api-version=2024-10-21' \
  --openai-auth-mode proxy

tokenclaw run openai
```

In that example:

- `http://127.0.0.1:4003/v1` is the local TokenClaw base URL your OpenAI SDK or tool uses.
- `https://resource.openai.azure.com/openai/deployments/my-deployment?...` is the upstream provider base URL TokenClaw forwards to.
- `--openai-auth-mode proxy` uses credentials from the TokenClaw process environment. `--openai-auth-mode client` forwards each client's `Authorization` header instead.

TokenClaw preserves upstream path and query components such as Azure `api-version`,
avoids duplicate `/v1` route segments, and redacts userinfo or sensitive query
values in CLI and health output.

## Claude API apps

Configure the local Claude target:

```bash
tokenclaw activate claude
tokenclaw run claude
```

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

## Codex VS Code and CLI

`tokenclaw activate codex` configures Codex through the user-level Codex
`openai_base_url` setting:

```bash
tokenclaw activate codex
tokenclaw run openai
```

This writes or updates:

```toml
# ~/.codex/config.toml
openai_base_url = "http://127.0.0.1:4003/v1"
```

Start VS Code from a shell or launcher where Codex can access your existing
OpenAI credentials:

```bash
code .
```

For a one-off CLI check:

```bash
codex exec --config 'openai_base_url="http://127.0.0.1:4003/v1"' "Reply with ok"
```

Do not put this provider setting in a project `.codex/config.toml`; Codex treats
provider/auth settings as machine-local.

## Claude VS Code and Claude Code

`tokenclaw activate claude-vscode` writes a TokenClaw-managed non-secret env file
for the Claude base URL and adds a `source ~/.tokenclaw/claude-vscode.env` line to
your shell profile (`~/.zshrc`, `~/.bashrc`, or `~/.profile`). On Linux desktops
it also writes `~/.config/environment.d/tokenclaw.conf` so VS Code launched from
GNOME or another graphical launcher can inherit `ANTHROPIC_BASE_URL` after your
next login. It does not write `ANTHROPIC_API_KEY` or token values. Use
`--no-shell-profile` if you manage shell startup files yourself.

```bash
tokenclaw activate claude-vscode
tokenclaw run claude
```

The managed env file contains the local Claude proxy base URL:

```text
ANTHROPIC_BASE_URL=http://127.0.0.1:4000
```

Open a new terminal, or source your shell profile, then launch VS Code or Claude
Code from a shell that has your Claude credentials:

```bash
export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_API_KEY"
code .
```

For VS Code launched from a graphical desktop, log out and back in before
launching it from the desktop environment. Keep secrets in your user
environment, not in repository files.

## Claude Desktop on Linux

`tokenclaw activate claude-desktop` updates the Claude Desktop `.desktop`
launcher and writes `~/.config/environment.d/tokenclaw.conf` so Claude Desktop
and its spawned Claude CLI subprocesses inherit the local TokenClaw Anthropic
base URL. It does not write `ANTHROPIC_API_KEY` or token values.

```bash
tokenclaw activate claude-desktop
tokenclaw run claude
```

By default TokenClaw patches `~/.local/share/applications/claude-desktop.desktop`
when present. If only `/usr/share/applications/claude-desktop.desktop` exists,
copy it to the user applications directory or pass `--force` when you intend to
patch the system-level file with suitable permissions. Use `--desktop-file PATH`
for custom launcher locations.

After activation, log out and back in so the systemd user environment is
loaded by desktop-launched apps. To apply it immediately in the current session,
run:

```bash
systemctl --user import-environment ANTHROPIC_BASE_URL
```

## GitHub Copilot non-goal

GitHub Copilot is not currently a normal TokenClaw base-url activation target.
The TokenClaw base-url path works for clients/tools that let the user set an
OpenAI-compatible or Anthropic-compatible provider base URL. Copilot is mediated
through GitHub/Copilot product integrations, accounts, policy, and extension
behavior. Future Copilot support should be designed as a separate integration,
not as `tokenclaw activate copilot` silently editing an assumed provider URL.

```bash
tokenclaw activate copilot
# unsupported: GitHub Copilot is not a base-url target; see README.md
```

## How does it work?

TokenClaw sits between tools such as Codex, Claude Code, VS Code extensions, or your own API app and the provider API.

```text
your app / IDE plugin -> TokenClaw on localhost -> OpenAI or Anthropic
```

It currently supports:

| Provider mode | Routes |
| --- | --- |
| Anthropic / Claude-compatible | `POST /v1/messages` |
| OpenAI-compatible | `POST /v1/responses`, `POST /v1/chat/completions`, WebSocket `/v1/responses`, plus files/uploads passthrough |

TokenClaw gives you local visibility into coding-agent traffic:

- estimated tokens, spend, savings, and recent activity
- which app, session, provider, and model generated calls
- routing, prompt-crunching, cache, retry, backoff, and error decisions
- policy state and whether local policy files need reload

It is meant for answering "where did the tokens and cost go?" without sending prompts or responses to another service.

## Manual proxy fallback

The packaged install exposes one public command, `tokenclaw`. Use `tokenclaw run`
for normal provider proxies. Advanced direct proxy invocation is available under
the hidden internal namespace for development and diagnostics.

Run the Anthropic-compatible proxy directly:

```bash
tokenclaw internal proxy --provider anthropic --host 127.0.0.1 --port 4000
```

Run the OpenAI-compatible proxy directly:

```bash
tokenclaw internal proxy --provider openai --openai-auth-mode proxy --host 127.0.0.1 --port 4003
```

`--openai-auth-mode proxy` means TokenClaw uses `TOKENCLAW_OPENAI_API_KEY` or
`OPENAI_API_KEY` from the proxy environment when forwarding upstream. Use
`--openai-auth-mode client` if you want each client request's `Authorization`
header to be forwarded.

Run tests:

```bash
python -m unittest discover -s tests
```

## Managed control plane boundary

`tokenclaw_server` is the future managed control plane for metadata-only policy
recommendations. It is not a provider proxy and should not receive or forward
raw provider request bodies. Local TokenClaw still owns provider forwarding,
request mutation, cache lookup/storage, local rule files, rollback, and the
read-only dashboard.

## Dashboard

When a proxy is running, open the dashboard from that proxy:

```text
http://127.0.0.1:4000/tokenclaw/dashboard
http://127.0.0.1:4003/tokenclaw/dashboard
```

Or run a dashboard-only process that reads the same local database:

```bash
tokenclaw internal dashboard --host 127.0.0.1 --port 4002
```

Then open:

```text
http://127.0.0.1:4002/tokenclaw/dashboard
```

The dashboard tells you:

- recent calls and sessions
- estimated tokens, cost, and savings
- usage by provider, model, app, and session when labels are available
- cache hits/misses/skips
- routing and prompt-crunch decisions
- retries, errors, rate-limit/backoff state
- active local policies and reload status

## Codex traffic

Codex API-key traffic uses the OpenAI-compatible local proxy through `tokenclaw activate codex`.
For OAuth/subscription unattended worker runs, use direct Codex execution (`TOKENCLAW_CODEX_TRANSPORT=exec`).
The previous experimental Codex app-server relay on ports 4013 and 4014 has been retired.

Inspect recent Codex/OpenAI-compatible traffic through normal stats and dashboard commands:

```bash
tokenclaw stats
```

## Defaults and privacy

- Local database: `~/.tokenclaw/tokenclaw.sqlite3`
- Prompt/response body logging: off by default
- Enable raw body logging only for local debugging:

```bash
export TOKENCLAW_LOG_BODIES=1
```

- Keep proxy ports on `127.0.0.1` unless you add your own network/auth boundary.
- TokenClaw is not an authentication gateway.
- Cost and token numbers are estimates unless the upstream provider returns exact usage.
- If TokenClaw is unsure about routing, crunching, or caching, it forwards the request unchanged and records why.

## Common configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOKENCLAW_DB` | `~/.tokenclaw/tokenclaw.sqlite3` | Local SQLite metadata DB |
| `TOKENCLAW_DATABASE_URL` | unset | Use Postgres instead of SQLite |
| `TOKENCLAW_CACHE` | `1` | Enable exact local cache where safe |
| `TOKENCLAW_CACHE_TOOL_CALLS` | `0` | Cache tool-call requests only if you accept the risk |
| `TOKENCLAW_ROUTING` | `1` | Enable local model routing |
| `TOKENCLAW_LOG_BODIES` | `0` | Store raw request/response bodies |
| `TOKENCLAW_HOST` | `0.0.0.0` | Proxy host default |
| `TOKENCLAW_PORT` | `4000` | Proxy port default |
| `TOKENCLAW_DASHBOARD_HOST` | `0.0.0.0` | Standalone dashboard host |
| `TOKENCLAW_DASHBOARD_PORT` | `4002` | Standalone dashboard port |
| `TOKENCLAW_RECOMMENDATIONS_ENABLED` | `0` | Enable metadata-only managed recommendation calls |
| `TOKENCLAW_RECOMMENDATION_SERVER_URL` | unset | Managed optimizer URL for recommendation and policy-decision calls |
| `TOKENCLAW_POLICY_DECISIONS_ENABLED` | `0` | Use managed `/v1/policy-decision` responses for local actions |
| `TOKENCLAW_POLICY_DECISION_MIN_CONFIDENCE` | `0.75` | Minimum managed routing confidence before local apply |
| `TOKENCLAW_POLICY_DECISION_CANARY_FRACTION` | `0.0` | Fraction of eligible managed routing decisions to apply |
| `TOKENCLAW_MANAGED_API_KEY` | unset | Bearer token for non-loopback managed servers |

For local managed-server development, set `TOKENCLAW_RECOMMENDATION_SERVER_URL=http://127.0.0.1:4100`.
That URL is treated as loopback-only and does not require `TOKENCLAW_MANAGED_API_KEY`.
Remote managed servers still require an API key. Managed calls send derived
feature metadata only; provider request bodies stay local. The legacy singular
switches `TOKENCLAW_RECOMMENDATION_ENABLED` and `TOKENCLAW_POLICY_DECISION_ENABLED`
are still accepted for existing installations.

## Local cache rules

Cache policy is loaded from the first existing path in this order:

```text
TOKENCLAW_CACHE_RULES
config/cache_rules.yaml
~/.tokenclaw/cache_rules.yaml
bundled tokenclaw/cache_rules.yaml
```

The bundled cache rules include a reviewed local rule for Claude Code
`short-completion` streaming requests routed to Haiku. It only applies when
metadata says the request is non-tool, high-cacheability, static information,
and not time-sensitive or user-specific. The first four matching successful
calls warm the rule; the fifth matching call is allowed to store the streamed
response, and later identical session-scoped calls can replay it.

If you maintain your own `config/cache_rules.yaml` or
`~/.tokenclaw/cache_rules.yaml`, include an equivalent pattern rule:

```yaml
pattern_rules:
  - id: local-haiku-short-completion-streaming-cache
    enabled: true
    policy_source: local-manual
    conditions:
      pattern_hashes: ["sha256:*"]
      source_surface: anthropic_messages
      app_family: claude_code
      model_pattern: haiku
      category: short-completion
      stream: true
      has_tools: false
      cacheability_bucket: high
      static_information_hint: true
      time_sensitive_hint: false
      user_specific_hint: false
      exact_cache_candidate_hint: true
    rollout:
      schema: tokenclaw.pattern_policy_rollout.v1
      recommendation_mode: canary-only
      canary_enabled: true
      canary_fraction: 1.0
      canary_salt: local-haiku-short-completion-streaming-cache-v1
      canary_unit: request_fingerprint
    action:
      type: exact_cache_pattern
      streaming: true
      allow_tool_calls: false
      safe_invalidation: false
      scope: session
      min_call_count: 5
```

## Advanced policy and diagnostics

The README keeps the happy path short. Advanced local policy review/apply/rollback, managed recommendation bridge, replayability reports, routing experiments, and promotion diagnostics are available under `tokenclaw internal`. Commands that need optional managed storage, zstd decoding, or OpenAI realtime bridging will print the matching extra install hint instead of making those dependencies part of the first-run install.

List them with:

```bash
tokenclaw internal --list
```
