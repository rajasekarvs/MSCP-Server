#!/usr/bin/env python3
"""
cowork_trigger_mcp — Production MCP Server
==========================================
Triggers Anthropic Cowork workflows via the Claude Managed Agents API.

CORRECT API FLOW (mandatory):
  1. POST /v1/agents  — create agent ONCE, store agent_id
  2. POST /v1/sessions — reference agent_id every run
  3. POST /v1/sessions/{id}/events — send task to running session
  4. GET  /v1/sessions/{id}/events — poll results

Key rules from official docs:
  - Sessions take ONLY an agent ID pointer — no inline model/system/tools
  - agents.create() is a setup step, NOT called per-run
  - Beta header: managed-agents-2026-04-01

Transport: Streamable HTTP (port 8080) for Render.com deployment
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("cowork_trigger_mcp")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL: str = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_VERSION: str = "2023-06-01"
MANAGED_AGENTS_BETA: str = "managed-agents-2026-04-01"
DEFAULT_MODEL: str = os.environ.get("COWORK_DEFAULT_MODEL", "claude-sonnet-4-6")
MCP_PORT: int = int(os.environ.get("MCP_PORT", "8080"))
REQUEST_TIMEOUT: float = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120"))

if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY is not set.")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP headers
# ─────────────────────────────────────────────────────────────────────────────
def _headers() -> Dict[str, str]:
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": MANAGED_AGENTS_BETA,
        "content-type": "application/json",
    }

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — shared HTTP client
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def app_lifespan(app: Any) -> AsyncIterator[Dict[str, Any]]:
    async with httpx.AsyncClient(
        base_url=ANTHROPIC_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        logger.info("cowork_trigger_mcp started — model=%s", DEFAULT_MODEL)
        yield {"client": client}
    logger.info("cowork_trigger_mcp shutting down.")

# ─────────────────────────────────────────────────────────────────────────────
# MCP server
# ─────────────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "cowork_trigger_mcp",
    lifespan=app_lifespan,
    host="0.0.0.0",
    port=MCP_PORT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _client(ctx: Context) -> httpx.AsyncClient:
    return ctx.request_context.lifespan_state["client"]

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)

def _err(exc: Exception, resource: str = "resource") -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        try:
            detail = exc.response.json().get("error", {}).get("message", exc.response.text[:300])
        except Exception:
            detail = exc.response.text[:300]
        hints = {
            401: "Check ANTHROPIC_API_KEY is valid.",
            403: f"API key lacks access to {resource}.",
            404: f"{resource} not found — verify the ID.",
            422: f"Invalid request body: {detail}",
            429: f"Rate limited. Retry after {exc.response.headers.get('retry-after','?')}s.",
        }
        return f"Error {code}: {hints.get(code, detail)}"
    if isinstance(exc, httpx.TimeoutException):
        return f"Timeout after {REQUEST_TIMEOUT}s — increase REQUEST_TIMEOUT_SECONDS."
    return f"Error ({type(exc).__name__}): {str(exc)[:300]}"

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class CreateAgentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=3, max_length=100,
        description="Name for this agent (e.g. 'Sales Report Agent').")
    system_prompt: str = Field(..., min_length=20, max_length=20000,
        description="Full system prompt describing the agent's role and behaviour.")
    model: Optional[str] = Field(default=None,
        description=f"Anthropic model ID. Defaults to {DEFAULT_MODEL}.")
    enable_bash: bool = Field(default=True, description="Enable bash tool.")
    enable_files: bool = Field(default=True, description="Enable file operations tool.")
    enable_web_search: bool = Field(default=False, description="Enable web search tool.")


class StartSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent_id: str = Field(..., min_length=5, max_length=128,
        description="Agent ID from cowork_create_agent (e.g. 'agent_01ABC...'). "
                    "Create one first — sessions require a pre-created agent.")
    task: str = Field(..., min_length=10, max_length=8000,
        description="Natural-language task for the agent to complete. Be specific.")
    title: Optional[str] = Field(default=None, max_length=200,
        description="Optional human-readable title for this session.")

    @field_validator("task")
    @classmethod
    def task_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task must not be blank.")
        return v.strip()


class SendEventInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    session_id: str = Field(..., min_length=5, max_length=128,
        description="Session ID from cowork_start_session.")
    message: str = Field(..., min_length=1, max_length=8000,
        description="Follow-up message to steer the running agent.")


class GetSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=5, max_length=128,
        description="Session ID from cowork_start_session.")
    max_events: int = Field(default=20, ge=1, le=100,
        description="Max recent events to return.")


class ListSessionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: Optional[str] = Field(default=None,
        description="Filter by agent ID.")
    status: Optional[str] = Field(default=None,
        description="Filter: 'running', 'completed', 'failed', 'interrupted'.")
    limit: int = Field(default=20, ge=1, le=100)


class ListAgentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=100)


class UpdateAgentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent_id: str = Field(..., min_length=5, max_length=128,
        description="Agent ID to update.")
    system_prompt: Optional[str] = Field(default=None, max_length=20000,
        description="New system prompt (creates a new version).")
    name: Optional[str] = Field(default=None, max_length=100,
        description="New agent name.")


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_create_agent
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_create_agent",
    annotations={"title": "Create Cowork Agent", "readOnlyHint": False},
)
async def cowork_create_agent(params: CreateAgentInput, ctx: Context) -> str:
    """
    Create a reusable Managed Agent definition. Do this ONCE and store the
    returned agent_id — pass it to cowork_start_session for every workflow run.

    Args:
        name: Human-readable agent name.
        system_prompt: Full system prompt (role, behaviour, output format).
        model: Anthropic model ID (default: claude-sonnet-4-6).
        enable_bash: Allow the agent to run shell commands (default: true).
        enable_files: Allow file read/write operations (default: true).
        enable_web_search: Allow web search (default: false).

    Returns agent_id — save this, you need it for every session.
    """
    client = _client(ctx)

    tools = []
    if params.enable_bash:
        tools.append({"type": "bash_20250124", "name": "bash"})
    if params.enable_files:
        tools.append({"type": "text_editor_20250429", "name": "str_replace_based_edit_tool"})
    if params.enable_web_search:
        tools.append({"type": "web_search_20250305", "name": "web_search"})

    body: Dict[str, Any] = {
        "name": params.name,
        "model": params.model or DEFAULT_MODEL,
        "system": params.system_prompt,
        "tools": tools,
    }

    await ctx.log_info("Creating agent", {"name": params.name})
    try:
        resp = await client.post("/v1/agents", json=body, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _err(exc, "Agents API")

    agent_id = data.get("id", "")
    logger.info("Agent created: %s", agent_id)

    return "\n".join([
        "## ✅ Agent Created",
        f"- **Agent ID:** `{agent_id}`  ← save this for cowork_start_session",
        f"- **Name:** {data.get('name')}",
        f"- **Model:** {data.get('model')}",
        f"- **Version:** {data.get('version', 1)}",
        f"- **Created:** {data.get('created_at', _utcnow())}",
        "",
        "**Next step:** call `cowork_start_session` with this agent_id.",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_update_agent
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_update_agent",
    annotations={"title": "Update Cowork Agent", "readOnlyHint": False},
)
async def cowork_update_agent(params: UpdateAgentInput, ctx: Context) -> str:
    """
    Update an existing agent's system prompt or name. Each update creates a new
    immutable version — existing sessions keep their pinned version.

    Use this instead of creating a new agent when you want to change behaviour.

    Args:
        agent_id: Agent to update.
        system_prompt: New system prompt (optional).
        name: New agent name (optional).

    Returns the new version number.
    """
    client = _client(ctx)
    body: Dict[str, Any] = {}
    if params.system_prompt:
        body["system"] = params.system_prompt
    if params.name:
        body["name"] = params.name

    if not body:
        return "Error: Provide at least one of system_prompt or name to update."

    try:
        resp = await client.post(f"/v1/agents/{params.agent_id}", json=body, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _err(exc, f"Agent {params.agent_id}")

    return "\n".join([
        f"## ✅ Agent Updated: `{params.agent_id}`",
        f"- **New version:** {data.get('version')}",
        f"- **Updated:** {data.get('updated_at', _utcnow())}",
        "",
        "New sessions will use this version. Running sessions keep their pinned version.",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_list_agents
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_list_agents",
    annotations={"title": "List Cowork Agents", "readOnlyHint": True},
)
async def cowork_list_agents(params: ListAgentsInput, ctx: Context) -> str:
    """
    List all Managed Agent definitions in your Anthropic organisation.
    Use this to find agent IDs before calling cowork_start_session.

    Args:
        limit: Max agents to return (default 20).
    """
    client = _client(ctx)
    try:
        resp = await client.get("/v1/agents", headers=_headers(), params={"limit": params.limit})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _err(exc, "Agents API")

    agents: List[Dict[str, Any]] = data.get("data", data.get("agents", []))
    if not agents:
        return "No agents found. Create one with cowork_create_agent first."

    lines = [f"## Registered Agents ({len(agents)} found)", ""]
    for a in agents:
        lines.append(
            f"- **{a.get('name', 'Unnamed')}** — "
            f"ID: `{a.get('id')}` | "
            f"Model: {a.get('model', 'N/A')} | "
            f"Version: {a.get('version', '?')} | "
            f"Created: {a.get('created_at', 'N/A')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_start_session
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_start_session",
    annotations={"title": "Start Cowork Workflow Session", "readOnlyHint": False},
)
async def cowork_start_session(params: StartSessionInput, ctx: Context) -> str:
    """
    Start a Managed Agent session to execute a Cowork workflow. Requires a
    pre-created agent_id from cowork_create_agent.

    The session is asynchronous — it runs in the background on Anthropic's
    infrastructure. Use cowork_get_session_status to poll progress.

    Args:
        agent_id: From cowork_create_agent. REQUIRED — sessions cannot be
                  created without a pre-existing agent.
        task: Full natural-language task description. Be specific — include
              file paths, data sources, expected output format.
        title: Optional label for this session.

    Returns session_id — use with cowork_get_session_status.

    Examples:
        task="Read /data/sales.xlsx, compute revenue by region, write
              markdown summary to /out/q2_report.md"
        task="Search for GDPR enforcement actions in 2026 and summarise
              the top 5 in bullet points"
    """
    client = _client(ctx)

    # Step 1: Create the session referencing the agent
    session_body: Dict[str, Any] = {
        "agent": params.agent_id,   # string shorthand = latest version
    }
    if params.title:
        session_body["title"] = params.title

    await ctx.log_info("Creating session", {"agent_id": params.agent_id})
    await ctx.report_progress(0.2, "Creating session...")

    try:
        resp = await client.post("/v1/sessions", json=session_body, headers=_headers())
        resp.raise_for_status()
        session = resp.json()
    except Exception as exc:
        return _err(exc, "Sessions API")

    session_id = session.get("id", "")
    await ctx.log_info("Session created", {"session_id": session_id})
    await ctx.report_progress(0.5, "Sending task to agent...")

    # Step 2: Send the task as the first user event
    event_body: Dict[str, Any] = {
        "type": "user",
        "content": [{"type": "text", "text": params.task}],
    }

    try:
        eresp = await client.post(
            f"/v1/sessions/{session_id}/events",
            json=event_body,
            headers=_headers(),
        )
        eresp.raise_for_status()
    except Exception as exc:
        return (
            f"Session created (`{session_id}`) but failed to send task: {_err(exc, 'Events API')}\n"
            f"Retry with cowork_send_event using session_id=`{session_id}`."
        )

    await ctx.report_progress(1.0, "Task sent — agent is running.")

    lines = [
        "## ✅ Cowork Workflow Session Started",
        f"- **Session ID:** `{session_id}`  ← use with cowork_get_session_status",
        f"- **Agent ID:** `{params.agent_id}`",
        f"- **Status:** running",
        f"- **Started:** {_utcnow()}",
        "",
        "The agent is running autonomously. "
        "Call `cowork_get_session_status` to poll progress and read output.",
    ]
    if session.get("session_url"):
        lines.insert(4, f"- **Live view:** {session['session_url']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_get_session_status
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_get_session_status",
    annotations={"title": "Get Session Status", "readOnlyHint": True},
)
async def cowork_get_session_status(params: GetSessionInput, ctx: Context) -> str:
    """
    Poll the status and output of a running Cowork workflow session.

    Args:
        session_id: From cowork_start_session.
        max_events: Number of recent events to return (default 20).

    Returns status ('running', 'completed', 'failed', 'interrupted')
    and the latest agent output events.
    """
    client = _client(ctx)
    await ctx.report_progress(0.2, "Fetching session...")

    try:
        resp = await client.get(
            f"/v1/sessions/{params.session_id}",
            headers=_headers(),
        )
        resp.raise_for_status()
        session = resp.json()
    except Exception as exc:
        return _err(exc, f"Session {params.session_id}")

    await ctx.report_progress(0.6, "Fetching events...")

    try:
        eresp = await client.get(
            f"/v1/sessions/{params.session_id}/events",
            headers=_headers(),
            params={"limit": params.max_events},
        )
        eresp.raise_for_status()
        events: List[Dict[str, Any]] = eresp.json().get("data", [])
    except Exception as exc:
        events = []
        await ctx.log_error("Events fetch failed", {"error": str(exc)})

    await ctx.report_progress(1.0, "Done.")

    status = session.get("status", "unknown")
    icons = {"running": "⏳", "completed": "✅", "failed": "❌", "interrupted": "⚠️"}
    icon = icons.get(status, "🔵")

    lines = [
        f"## {icon} Session `{params.session_id}`",
        f"- **Status:** {status}",
        f"- **Agent:** `{session.get('agent_id', 'N/A')}`",
        f"- **Created:** {session.get('created_at', 'N/A')}",
        f"- **Updated:** {session.get('updated_at', 'N/A')}",
    ]
    if session.get("session_url"):
        lines.append(f"- **Live view:** {session['session_url']}")

    if events:
        lines += ["", f"### Recent Events ({len(events)})"]
        for ev in events:
            ev_type = ev.get("type", "?")
            # Extract text from content blocks
            content = ev.get("content", [])
            text = " ".join(
                b.get("text", "")[:200]
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text:
                lines.append(f"- **[{ev_type}]** {text}")
            else:
                lines.append(f"- **[{ev_type}]** *(non-text content)*")
    else:
        lines.append("\n_No events yet — agent may still be starting._")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_send_event
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_send_event",
    annotations={"title": "Send Event to Session", "readOnlyHint": False},
)
async def cowork_send_event(params: SendEventInput, ctx: Context) -> str:
    """
    Send a follow-up message to a running session to steer the agent,
    add context, or approve a proposed action.

    Args:
        session_id: From cowork_start_session.
        message: Instruction or context to send to the agent.
    """
    client = _client(ctx)
    body: Dict[str, Any] = {
        "type": "user",
        "content": [{"type": "text", "text": params.message}],
    }

    try:
        resp = await client.post(
            f"/v1/sessions/{params.session_id}/events",
            json=body,
            headers=_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _err(exc, f"Session {params.session_id}")

    return (
        f"## ✅ Event Sent\n"
        f"- **Session:** `{params.session_id}`\n"
        f"- **Event ID:** `{data.get('id', 'N/A')}`\n"
        f"- **Sent at:** {_utcnow()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_list_sessions
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_list_sessions",
    annotations={"title": "List Cowork Sessions", "readOnlyHint": True},
)
async def cowork_list_sessions(params: ListSessionsInput, ctx: Context) -> str:
    """
    List recent Managed Agent sessions with optional filtering.

    Args:
        agent_id: Filter by agent ID (optional).
        status: Filter by 'running', 'completed', 'failed', 'interrupted' (optional).
        limit: Max sessions to return (default 20).
    """
    client = _client(ctx)
    query: Dict[str, Any] = {"limit": params.limit}
    if params.agent_id:
        query["agent_id"] = params.agent_id
    if params.status:
        query["status"] = params.status

    try:
        resp = await client.get("/v1/sessions", headers=_headers(), params=query)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _err(exc, "Sessions API")

    sessions: List[Dict[str, Any]] = data.get("data", data.get("sessions", []))
    if not sessions:
        return "No sessions found."

    icons = {"running": "⏳", "completed": "✅", "failed": "❌", "interrupted": "⚠️"}
    lines = [f"## Sessions ({len(sessions)} found)", ""]
    for s in sessions:
        st = s.get("status", "?")
        lines.append(
            f"{icons.get(st,'🔵')} `{s.get('id')}` — **{st}** | "
            f"agent: `{s.get('agent_id','?')}` | "
            f"updated: {s.get('updated_at','N/A')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — inject health route into MCP's Starlette app
# ─────────────────────────────────────────────────────────────────────────────
async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "cowork_trigger_mcp"})


if __name__ == "__main__":
    app = mcp.streamable_http_app()
    # Inject health check routes so Render.com HEAD / doesn't 404
    app.routes.insert(0, Route("/", _health))
    app.routes.insert(1, Route("/health", _health))
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT, log_level="info")
