# MSCP-Server — Cowork Trigger MCP Server

Production-ready MCP server that triggers **Anthropic Cowork workflows** programmatically
via the Claude Managed Agents API and Claude Code Routines.

## Tools

| Tool | API | Purpose |
|---|---|---|
| `cowork_create_agent` | `POST /v1/agents` | Register a reusable agent definition |
| `cowork_start_session` | `POST /v1/sessions` | **Trigger a Cowork workflow session** |
| `cowork_send_event` | `POST /v1/sessions/{id}/events` | Steer a running session |
| `cowork_get_session_status` | `GET /v1/sessions/{id}` | Poll progress and read outputs |
| `cowork_list_sessions` | `GET /v1/sessions` | List and audit all workflow runs |
| `cowork_fire_routine` | `POST /v1/claude_code/routines/{id}/fire` | Fire a Claude Code Routine |
| `cowork_list_agents` | `GET /v1/agents` | Discover registered agent IDs |

## Quick deploy to Render.com

1. Connect this repo in [Render Dashboard](https://dashboard.render.com)
2. New → **Blueprint** → select this repo → Apply
3. Enter your `ANTHROPIC_API_KEY` when prompted
4. Render builds and deploys automatically

## Local development

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
pip install -r requirements.txt
python server.py
# Server starts on http://localhost:8080
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | YES | From console.anthropic.com → API Keys |
| `COWORK_DEFAULT_MODEL` | Optional | Default: claude-sonnet-4-6 |
| `MCP_PORT` | Optional | Default: 8080 |
| `REQUEST_TIMEOUT_SECONDS` | Optional | Default: 120 |
