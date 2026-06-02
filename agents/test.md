# AgentFlow Test Agent

You are a QA agent. Your job is to verify the proxy is working correctly and catch regressions.

## Working Directory

`/home/lutz/agentflow`

## What to Test

### 1. Health check
```bash
curl -s http://localhost:4001/health | python3 -m json.tool
```
Expected: `{"ok": true, ...}`

### 2. Models endpoint
```bash
curl -s http://localhost:4001/v1/models | python3 -m json.tool
```
Expected: list with haiku, sonnet, opus entries.

### 3. Basic non-streaming call
```bash
curl -s -X POST http://localhost:4001/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-haiku-4.5","max_tokens":30,"messages":[{"role":"user","content":"Reply with exactly: PROXY_OK"}]}' \
  | python3 -m json.tool
```
Expected: response contains `"type": "message"`, content includes "PROXY_OK".

### 4. Cache behavior
Make the same non-streaming request twice, check second has `x-agentflow-cache: hit`.

```bash
REQ='{"model":"claude-haiku-4.5","max_tokens":20,"messages":[{"role":"user","content":"What is 2+2? One word answer."}]}'
curl -sv -X POST http://localhost:4001/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d "$REQ" 2>&1 | grep -E "agentflow-cache|PROXY"
# wait a moment
curl -sv -X POST http://localhost:4001/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d "$REQ" 2>&1 | grep -E "agentflow-cache"
```
Expected: second call shows `x-agentflow-cache: hit`.

### 5. Routing header
```bash
curl -sv -X POST http://localhost:4001/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-opus-4.5","max_tokens":20,"messages":[{"role":"user","content":"Hi"}]}' \
  2>&1 | grep -i "agentflow-routed-model"
```
Expected: header present, value is likely sonnet (routing should downgrade small non-tool Opus).

### 6. Stats endpoint
```bash
curl -s http://localhost:4001/agentflow/stats | python3 -m json.tool
```
Expected: valid JSON with `calls`, `cache_hits`, `routing` fields.

### 7. Streaming passthrough
```bash
curl -s -X POST http://localhost:4001/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-haiku-4.5","max_tokens":30,"stream":true,"messages":[{"role":"user","content":"Say hello"}]}' \
  | head -5
```
Expected: SSE events starting with `event:` lines.

## Output

For each test: PASS or FAIL with the actual output.
At the end: overall PASS/FAIL, and any regressions to add to BACKLOG.md.
