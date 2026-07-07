# TokenClaw Docker image

The Docker image runs the local TokenClaw proxy stack with managed/control-plane features opted out by default. It starts:

- Anthropic-compatible proxy on `4000`
- read-only dashboard on `4002`
- OpenAI-compatible proxy on `4003`

```bash
docker run --rm \
  -p 4000:4000 \
  -p 4002:4002 \
  -p 4003:4003 \
  -v tokenclaw-data:/data \
  -v tokenclaw-config:/config \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  lutzkuen/tokenclaw:latest
```

Then point clients at the local proxy URLs:

```text
OpenAI-compatible: http://127.0.0.1:4003/v1
Anthropic-compatible: http://127.0.0.1:4000
Dashboard: http://127.0.0.1:4002/tokenclaw/dashboard
```

The image persists local metadata in `/data/tokenclaw.sqlite3` and activation/config files in `/config`.

## Defaults

The image installs `tokenclaw[server]` and intentionally does not install managed storage extras. These environment variables are set in the image so managed server behavior stays off unless explicitly overridden:

```text
TOKENCLAW_LOCAL_RULES_ONLY=1
TOKENCLAW_MANAGED=0
TOKENCLAW_MANAGED_ROUTING=0
TOKENCLAW_MANAGED_CRUNCH=0
TOKENCLAW_MANAGED_CACHE=0
TOKENCLAW_RECOMMENDATIONS_ENABLED=0
TOKENCLAW_POLICY_DECISIONS_ENABLED=0
```

Local crunching, local cache, local routing, provider proxying, and the read-only dashboard remain available.

## Run a single proxy

The default command is `tokenclaw start`. You can pass TokenClaw subcommands directly; `start` will create the matching local activation profile when it is missing:

```bash
docker run --rm -p 4003:4003 -p 4002:4002 -v tokenclaw-data:/data -v tokenclaw-config:/config lutzkuen/tokenclaw:latest start --openai

docker run --rm -p 4000:4000 -p 4002:4002 -v tokenclaw-data:/data -v tokenclaw-config:/config lutzkuen/tokenclaw:latest start --claude
```

Use `--no-dashboard` if you want only the provider proxy process.

## Publishing

The GitHub Actions workflow builds the Docker image on pull requests and pushes it to Docker Hub on `master`, version tags (`v*`), and manual dispatches.

Configure these repository secrets before the first publish run:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```
