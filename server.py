#!/usr/bin/env python3
"""
cowork_trigger_mcp — Production MCP Server
==========================================
Provides tools to trigger and manage Cowork-style workflows programmatically
using two official Anthropic APIs:

  1. Claude Managed Agents Sessions API  (POST /v1/sessions)
     → the production API powering Cowork's autonomous execution
     → beta header: managed-agents-2026-04-01

  2. Claude Code Routines /fire endpoint  (POST /v1/claude_code/routines/{id}/fire)
     → triggers pre-configured Claude Code Routines from external systems
     → beta header: experimental-cc-routine-2026-04-01

Transport : Streamable HTTP (port 8080) — suitable for Render / Railway / Fly.io
Auth      : ANTHROPIC_API_KEY via env var (never hardcoded)

Tools exposed
─────────────
  cowork_create_agent          Create / register a reusable Managed Agent definition
  cowork_start_session         Start a Managed Agent session (= trigger a Cowork workflow)
  cowork_send_event            Send a follow-up message / steer a running session
  cowork_get_session_status    Poll session status and read streamed events
  cowork_list_sessions         List recent sessions with filtering
  cowork_fire_routine          Fire a Claude Code Routine via its API trigger endpoint
  cowork_list_agents           List registered Managed Agent definitions
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# Logging  (stderr only — never stdout for stdio-compatible servers)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("cowork_trigger_mcp")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment
# ─────────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL: str = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
)
ANTHROPIC_VERSION: str = "2023-06-01"

# Beta headers
MANAGED_AGENTS_BETA: str = "managed-agents-2026-04-01"
ROUTINES_BETA: str = "experimental-cc-routine-2026-04-01"

# Default model for new agent definitions
DEFAULT_MODEL: str = os.environ.get("COWORK_DEFAULT_MODEL", "claude-sonnet-4-6")

MCP_PORT: int = int(os.environ.get("MCP_PORT", "8080"))
REQUEST_TIMEOUT: float = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120"))

# Validate at startup
if not ANTHROPIC_API_KEY:
    logger.error(
        "ANTHROPIC_API_KEY is not set. "
        "Export it before starting the server."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP client (lifespan-managed, reused across all requests)
# ─────────────────────────────────────────────────────────────────────────────
def _base_headers() -> Dict[str, str]:
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _managed_agents_headers() -> Dict[str, str]:
    return {**_base_headers(), "anthropic-beta": MANAGED_AGENTS_BETA}


def _routines_headers() -> Dict[str, str]:
    return {**_base_headers(), "anthropic-beta": ROUTINES_BETA}


@asynccontextmanager
async def app_lifespan(app: Any) -> AsyncIterator[Dict[str, Any]]:
    """Create one shared AsyncClient for the server's lifetime."""
    async with httpx.AsyncClient(
        base_url=ANTHROPIC_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        logger.info(
            "cowork_trigger_mcp started — base_url=%s model=%s",
            ANTHROPIC_BASE_URL,
            DEFAULT_MODEL,
        )
        yield {"client": client}
    logger.info("cowork_trigger_mcp shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# MCP server  (port/host set here — run() does NOT accept these kwargs)
# ─────────────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "cowork_trigger_mcp",
    lifespan=app_lifespan,
    host="0.0.0.0",   # bind all interfaces — required for Render / Docker
    port=MCP_PORT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────
def _client(ctx: Context) -> httpx.AsyncClient:
    return ctx.request_context.lifespan_state["client"]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handle_http_error(exc: Exception, resource: str = "resource") -> str:
    """Convert httpx exceptions into actionable error strings."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        try:
            body = exc.response.json()
            detail = body.get("error", {}).get("message", exc.response.text[:300])
        except Exception:
            detail = exc.response.text[:300]

        if code == 401:
            return (
                "Error 401 Unauthorized: Check that ANTHROPIC_API_KEY is valid "
                "and has not expired."
            )
        if code == 403:
            return f"Error 403 Forbidden: Your API key does not have access to {resource}."
        if code == 404:
            return (
                f"Error 404 Not Found: {resource} does not exist. "
                "Verify the ID is correct."
            )
        if code == 422:
            return f"Error 422 Unprocessable: {detail}"
        if code == 429:
            retry = exc.response.headers.get("retry-after", "unknown")
            return f"Error 429 Rate Limited: retry after {retry}s."
        if code >= 500:
            return f"Error {code} Server Error: {detail}. Retry in a few seconds."
        return f"Error {code}: {detail}"
    if isinstance(exc, httpx.TimeoutException):
        return f"Error: Request timed out after {REQUEST_TIMEOUT}s. Retry or increase REQUEST_TIMEOUT_SECONDS."
    if isinstance(exc, httpx.ConnectError):
        return "Error: Cannot connect to Anthropic API. Check network/firewall."
    return f"Error ({type(exc).__name__}): {str(exc)[:300]}"


def _fmt_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & shared input types
# ─────────────────────────────────────────────────────────────────────────────
class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic input models
# ─────────────────────────────────────────────────────────────────────────────

class CreateAgentInput(BaseModel):
    """Input model for cowork_create_agent."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(
        ...,
        description="Short human-readable name for this agent definition "
                    "(e.g. 'Sales Report Agent', 'Contract Reviewer').",
        min_length=3,
        max_length=100,
    )
    system_prompt: str = Field(
        ...,
        description=(
            "The agent's full system prompt. Describe its role, tools it may use, "
            "output format, and any constraints. "
            "Example: 'You are a financial analyst. Read spreadsheets, run Python "
            "calculations, and produce a concise executive summary.'"
        ),
        min_length=20,
        max_length=20000,
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            f"Anthropic model ID (e.g. 'claude-sonnet-4-6', 'claude-opus-4-6'). "
            f"Defaults to {DEFAULT_MODEL}."
        ),
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description=(
            "Built-in tools to enable. Options: 'bash', 'file_operations', "
            "'web_search', 'web_fetch'. Leave null to enable all."
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class StartSessionInput(BaseModel):
    """Input model for cowork_start_session."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description=(
            "Full natural-language description of the task this session should complete. "
            "Be specific — include file paths, data sources, recipient names, "
            "expected output format, and any deadline constraints.\n\n"
            "Examples:\n"
            "  • 'Read /data/q2_sales.xlsx, compute revenue by region, "
            "and write a 300-word markdown summary to /out/q2_report.md'\n"
            "  • 'Search the web for the latest GDPR enforcement actions in 2026 "
            "and email a 5-bullet digest to legal@corp.com'"
        ),
        min_length=20,
        max_length=8000,
    )
    agent_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of a pre-registered Managed Agent definition (from cowork_create_agent). "
            "If omitted, an inline agent is created using system_prompt and model."
        ),
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Inline system prompt when agent_id is not provided. "
            "Required if agent_id is omitted."
        ),
        max_length=20000,
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            f"Model override (e.g. 'claude-opus-4-6'). "
            f"Ignored when agent_id is set. Defaults to {DEFAULT_MODEL}."
        ),
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional key/value pairs injected into the task message as structured context "
            "(e.g. {'customer_id': 'C-001', 'fiscal_year': 2026})."
        ),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task must not be blank.")
        return v.strip()


class SendEventInput(BaseModel):
    """Input model for cowork_send_event."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Session ID returned by cowork_start_session.",
        min_length=5,
        max_length=128,
    )
    message: str = Field(
        ...,
        description=(
            "Follow-up instruction or steering message sent to the running agent. "
            "Use this to redirect, add context, or approve/deny a proposed action."
        ),
        min_length=1,
        max_length=8000,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class GetSessionStatusInput(BaseModel):
    """Input model for cowork_get_session_status."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        ...,
        description="Session ID returned by cowork_start_session.",
        min_length=5,
        max_length=128,
    )
    max_events: Optional[int] = Field(
        default=20,
        description="Maximum number of recent events to return (1–100).",
        ge=1,
        le=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class ListSessionsInput(BaseModel):
    """Input model for cowork_list_sessions."""
    model_config = ConfigDict(extra="forbid")

    agent_id: Optional[str] = Field(
        default=None,
        description="Filter by agent ID.",
        max_length=128,
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: 'running', 'completed', 'failed', 'interrupted'.",
        max_length=32,
    )
    limit: Optional[int] = Field(
        default=20,
        description="Maximum sessions to return (1–100).",
        ge=1,
        le=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class FireRoutineInput(BaseModel):
    """Input model for cowork_fire_routine."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    routine_id: str = Field(
        ...,
        description=(
            "Claude Code Routine ID from claude.ai/code/routines. "
            "Format: trig_01XXXXXXXXXX (shown in the routine's detail page)."
        ),
        min_length=5,
        max_length=128,
    )
    routine_bearer_token: str = Field(
        ...,
        description=(
            "Dedicated bearer token for this routine. Generated once at creation "
            "in the Routine detail page → 'Generate token'. Store securely — shown only once."
        ),
        min_length=10,
        max_length=512,
    )
    additional_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional text appended to the routine's base prompt for this specific run. "
            "Use to pass dynamic context (e.g. an alert payload, a file path, a ticket ID)."
        ),
        max_length=4000,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


class ListAgentsInput(BaseModel):
    """Input model for cowork_list_agents."""
    model_config = ConfigDict(extra="forbid")

    limit: Optional[int] = Field(
        default=20,
        description="Maximum agents to return (1–100).",
        ge=1,
        le=100,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' or 'json'.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private API helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _post(
    client: httpx.AsyncClient,
    path: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
) -> Dict[str, Any]:
    resp = await client.post(path, json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()


async def _get(
    client: httpx.AsyncClient,
    path: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resp = await client.get(path, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def _build_inline_agent(system_prompt: str, model: Optional[str]) -> Dict[str, Any]:
    """Inline agent definition embedded directly in the session request."""
    return {
        "model": model or DEFAULT_MODEL,
        "system_prompt": system_prompt,
        "tools": [
            {"type": "bash"},
            {"type": "file_operations"},
            {"type": "web_search"},
            {"type": "web_fetch"},
        ],
    }


def _inject_context(task: str, context: Optional[Dict[str, Any]]) -> str:
    """Append structured context to the task message."""
    if not context:
        return task
    ctx_block = "\n\n**Context provided by caller:**\n" + "\n".join(
        f"- `{k}`: {v}" for k, v in context.items()
    )
    return task + ctx_block


def _format_session_md(data: Dict[str, Any]) -> str:
    status = data.get("status", "unknown")
    icon = {"running": "⏳", "completed": "✅", "failed": "❌", "interrupted": "⚠️"}.get(
        status, "🔵"
    )
    lines = [
        f"## {icon} Session `{data.get('id', 'N/A')}`",
        f"- **Status:** {status}",
        f"- **Agent:** {data.get('agent_id', 'inline')}",
        f"- **Created:** {data.get('created_at', 'N/A')}",
        f"- **Updated:** {data.get('updated_at', 'N/A')}",
    ]
    if data.get("session_url"):
        lines.append(f"- **Live view:** {data['session_url']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_create_agent
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_create_agent",
    annotations={
        "title": "Create Cowork Agent Definition",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def cowork_create_agent(params: CreateAgentInput, ctx: Context) -> str:
    """
    Register a reusable Managed Agent definition in the Anthropic platform.

    Call this once to define an agent's model, system prompt, and tools.
    The returned agent_id can be referenced by cowork_start_session across
    multiple workflow runs without repeating the configuration each time.

    Args:
        params (CreateAgentInput):
            - name (str): Human-readable agent name (e.g. 'Contract Review Agent').
            - system_prompt (str): Full system prompt for the agent.
            - model (Optional[str]): Anthropic model ID (defaults to claude-sonnet-4-6).
            - tools (Optional[List[str]]): Built-in tools to enable.
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Markdown or JSON containing the agent_id and creation metadata.
        {
            "id": str,            # agent_id — store this for cowork_start_session
            "name": str,
            "model": str,
            "created_at": str
        }

    Error Handling:
        - Error 401: ANTHROPIC_API_KEY invalid or missing.
        - Error 422: Invalid request body — check system_prompt length and model name.
    """
    client = _client(ctx)
    await ctx.log_info("Creating agent definition", {"name": params.name})

    tool_map = {
        "bash": {"type": "bash"},
        "file_operations": {"type": "file_operations"},
        "web_search": {"type": "web_search"},
        "web_fetch": {"type": "web_fetch"},
    }
    if params.tools:
        tools = [tool_map[t] for t in params.tools if t in tool_map]
    else:
        tools = list(tool_map.values())

    body: Dict[str, Any] = {
        "name": params.name,
        "model": params.model or DEFAULT_MODEL,
        "system_prompt": params.system_prompt,
        "tools": tools,
    }

    try:
        data = await _post(client, "/v1/agents", body, _managed_agents_headers())
    except Exception as exc:
        return _handle_http_error(exc, "Agents API")

    await ctx.log_info("Agent created", {"id": data.get("id")})
    await ctx.report_progress(1.0, "Agent created.")

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json(data)

    return "\n".join([
        f"## ✅ Agent Created: {data.get('name', params.name)}",
        f"- **Agent ID:** `{data.get('id')}`  ← save this for cowork_start_session",
        f"- **Model:** {data.get('model', params.model or DEFAULT_MODEL)}",
        f"- **Created:** {data.get('created_at', _utcnow())}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_start_session
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_start_session",
    annotations={
        "title": "Start Cowork Workflow Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def cowork_start_session(params: StartSessionInput, ctx: Context) -> str:
    """
    Start a Claude Managed Agent session to execute a Cowork-style workflow.

    This is the primary trigger tool. It creates a stateful agent session on
    Anthropic's managed infrastructure, where Claude autonomously executes the
    task using bash, file operations, web search, and any configured MCP tools.

    Sessions are asynchronous — they run in the background. Use
    cowork_get_session_status to poll progress and retrieve results.

    Args:
        params (StartSessionInput):
            - task (str): Full natural-language task description.
            - agent_id (Optional[str]): Pre-registered agent ID. If omitted,
              system_prompt and model are used to create an inline agent.
            - system_prompt (Optional[str]): Required if agent_id is omitted.
            - model (Optional[str]): Model override (ignored if agent_id set).
            - context (Optional[dict]): Key/value pairs injected into task message.
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Session metadata including session_id and live session_url.
        {
            "id": str,              # session_id — use with cowork_get_session_status
            "status": str,          # 'running' initially
            "session_url": str,     # Live view URL (claude.ai/code)
            "created_at": str
        }

    Examples:
        - Trigger a sales report: task="Read /data/sales.xlsx and write a
          markdown summary by region to /out/report.md"
        - Summarise emails: task="Check my inbox, find unread messages from
          the legal team this week, and draft a priority list"
        - With agent_id: Use a pre-configured agent for consistent behaviour
          across multiple workflow runs.

    Error Handling:
        - Error 401: ANTHROPIC_API_KEY invalid.
        - Error 422: Missing system_prompt when agent_id not supplied.
        - Error 429: Rate limit — retry after the indicated delay.
    """
    client = _client(ctx)

    if not params.agent_id and not params.system_prompt:
        return (
            "Error: Either agent_id or system_prompt must be provided. "
            "Run cowork_create_agent first, or supply an inline system_prompt."
        )

    full_task = _inject_context(params.task, params.context)
    await ctx.log_info("Starting session", {"task_preview": full_task[:120]})
    await ctx.report_progress(0.10, "Sending session request to Anthropic…")

    body: Dict[str, Any] = {
        "input": {"text": full_task},
    }

    if params.agent_id:
        body["agent_id"] = params.agent_id
    else:
        body["agent"] = _build_inline_agent(
            params.system_prompt or "",  # validated above
            params.model,
        )

    try:
        data = await _post(client, "/v1/sessions", body, _managed_agents_headers())
    except Exception as exc:
        return _handle_http_error(exc, "Sessions API")

    session_id = data.get("id", "")
    session_url = data.get("session_url", "")
    await ctx.log_info("Session started", {"id": session_id, "url": session_url})
    await ctx.report_progress(1.0, "Session running.")

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json(data)

    lines = [
        "## ✅ Cowork Workflow Session Started",
        f"- **Session ID:** `{session_id}`  ← use with cowork_get_session_status",
        f"- **Status:** {data.get('status', 'running')}",
    ]
    if session_url:
        lines.append(f"- **Live view:** {session_url}")
    lines += [
        f"- **Started:** {data.get('created_at', _utcnow())}",
        "",
        "The agent is now running autonomously. "
        "Call `cowork_get_session_status` to poll progress.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_send_event
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_send_event",
    annotations={
        "title": "Send Event to Running Cowork Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def cowork_send_event(params: SendEventInput, ctx: Context) -> str:
    """
    Send a follow-up message or steering instruction to a running Managed Agent session.

    Use this to redirect the agent, add missing context, approve a proposed action,
    or ask for clarification mid-execution.

    Args:
        params (SendEventInput):
            - session_id (str): From cowork_start_session.
            - message (str): Instruction, correction, or context to add.
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Acknowledgement from the Anthropic API confirming the event was queued.

    Examples:
        - "Focus only on EMEA data, ignore APAC"
        - "The file is at /data/v2/report.csv not /data/report.csv"
        - "Yes, proceed with the proposed change"

    Error Handling:
        - Error 404: session_id does not exist or session has already completed.
        - Error 409: Session is not in a state that accepts new events.
    """
    client = _client(ctx)
    path = f"/v1/sessions/{params.session_id}/events"
    body: Dict[str, Any] = {
        "type": "user",
        "content": [{"type": "text", "text": params.message}],
    }

    try:
        data = await _post(client, path, body, _managed_agents_headers())
    except Exception as exc:
        return _handle_http_error(exc, f"Session {params.session_id}")

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json(data)

    return (
        f"## ✅ Event Sent to Session `{params.session_id}`\n"
        f"- **Event ID:** `{data.get('id', 'N/A')}`\n"
        f"- **Sent at:** {data.get('created_at', _utcnow())}\n\n"
        "The agent will process your message in the next iteration."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_get_session_status
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_get_session_status",
    annotations={
        "title": "Get Cowork Session Status and Events",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def cowork_get_session_status(
    params: GetSessionStatusInput, ctx: Context
) -> str:
    """
    Retrieve the current status and recent event history of a Managed Agent session.

    Poll this tool to check whether a workflow started with cowork_start_session
    has completed, is still running, or has failed. Returns the latest events
    (tool calls, outputs, status messages) from the session's event stream.

    Args:
        params (GetSessionStatusInput):
            - session_id (str): From cowork_start_session.
            - max_events (Optional[int]): Number of recent events to return (default 20).
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Session status with the last N events from the event stream.
        {
            "id": str,
            "status": str,           # 'running' | 'completed' | 'failed' | 'interrupted'
            "created_at": str,
            "updated_at": str,
            "session_url": str,
            "events": [              # Latest events, newest last
                {
                    "type": str,     # 'assistant' | 'tool_use' | 'tool_result' | 'user'
                    "content": [...],
                    "created_at": str
                }
            ]
        }

    Error Handling:
        - Error 404: session_id does not exist.
    """
    client = _client(ctx)
    await ctx.report_progress(0.20, "Fetching session metadata…")

    try:
        session = await _get(
            client,
            f"/v1/sessions/{params.session_id}",
            _managed_agents_headers(),
        )
    except Exception as exc:
        return _handle_http_error(exc, f"Session {params.session_id}")

    await ctx.report_progress(0.60, "Fetching event stream…")

    try:
        events_resp = await _get(
            client,
            f"/v1/sessions/{params.session_id}/events",
            _managed_agents_headers(),
            params={"limit": params.max_events},
        )
        events: List[Dict[str, Any]] = events_resp.get("events", [])
    except Exception as exc:
        events = []
        await ctx.log_error("Failed to fetch events", {"error": str(exc)})

    await ctx.report_progress(1.0, "Done.")

    combined = {**session, "events": events}

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json(combined)

    status = session.get("status", "unknown")
    icon = {"running": "⏳", "completed": "✅", "failed": "❌", "interrupted": "⚠️"}.get(
        status, "🔵"
    )
    lines = [
        f"## {icon} Session Status: `{params.session_id}`",
        f"- **Status:** {status}",
        f"- **Created:** {session.get('created_at', 'N/A')}",
        f"- **Updated:** {session.get('updated_at', 'N/A')}",
    ]
    if session.get("session_url"):
        lines.append(f"- **Live view:** {session['session_url']}")

    if events:
        lines += ["", f"### Last {len(events)} Events"]
        for ev in events[-params.max_events:]:
            ev_type = ev.get("type", "unknown")
            content_blocks = ev.get("content", [])
            # Extract text from content blocks
            text_parts = [
                b.get("text", "")
                for b in content_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            preview = " ".join(text_parts)[:300]
            lines.append(f"- **[{ev_type}]** {preview}")
    else:
        lines.append("\n_No events recorded yet._")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_list_sessions
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_list_sessions",
    annotations={
        "title": "List Cowork Workflow Sessions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def cowork_list_sessions(params: ListSessionsInput, ctx: Context) -> str:
    """
    List recent Managed Agent sessions with optional filtering.

    Use this to get an overview of workflow runs, find specific sessions by
    status, or audit what Cowork tasks have been executed recently.

    Args:
        params (ListSessionsInput):
            - agent_id (Optional[str]): Filter by agent definition ID.
            - status (Optional[str]): Filter by 'running', 'completed', 'failed', 'interrupted'.
            - limit (Optional[int]): Max sessions to return (default 20, max 100).
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Paginated list of sessions.
        {
            "sessions": [
                {
                    "id": str,
                    "status": str,
                    "agent_id": str,
                    "created_at": str,
                    "updated_at": str
                }
            ],
            "count": int,
            "has_more": bool
        }

    Error Handling:
        - Error 401: Invalid API key.
    """
    client = _client(ctx)
    query_params: Dict[str, Any] = {"limit": params.limit}
    if params.agent_id:
        query_params["agent_id"] = params.agent_id
    if params.status:
        query_params["status"] = params.status

    try:
        data = await _get(
            client, "/v1/sessions", _managed_agents_headers(), params=query_params
        )
    except Exception as exc:
        return _handle_http_error(exc, "Sessions API")

    sessions: List[Dict[str, Any]] = data.get("sessions", data.get("data", []))
    has_more: bool = data.get("has_more", False)

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json({
            "sessions": sessions,
            "count": len(sessions),
            "has_more": has_more,
        })

    if not sessions:
        return "No sessions found matching the given filters."

    lines = [f"## Cowork Sessions ({len(sessions)} returned, has_more={has_more})", ""]
    status_icons = {
        "running": "⏳", "completed": "✅", "failed": "❌", "interrupted": "⚠️"
    }
    for s in sessions:
        st = s.get("status", "unknown")
        icon = status_icons.get(st, "🔵")
        lines.append(
            f"{icon} `{s.get('id')}` — **{st}** | "
            f"agent: `{s.get('agent_id', 'inline')}` | "
            f"updated: {s.get('updated_at', 'N/A')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_fire_routine
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_fire_routine",
    annotations={
        "title": "Fire a Claude Code Routine",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def cowork_fire_routine(params: FireRoutineInput, ctx: Context) -> str:
    """
    Trigger a pre-configured Claude Code Routine via its dedicated API endpoint.

    Claude Code Routines are saved workflows (prompt + repo + connectors) that
    run on Anthropic's cloud infrastructure. Each routine with an API trigger
    has a unique HTTP endpoint and bearer token. Firing it starts a new
    autonomous Claude Code session on Anthropic's servers.

    Use this when you have already configured a Routine in claude.ai/code/routines
    and want to trigger it programmatically (e.g. from a CI pipeline, monitoring
    alert, or another agent).

    Args:
        params (FireRoutineInput):
            - routine_id (str): From claude.ai/code/routines — the routine's trigger ID
              (format: trig_01XXXXXXXXXX).
            - routine_bearer_token (str): The per-routine bearer token generated at
              creation time. Shown only once — store in a secret manager.
            - additional_text (Optional[str]): Dynamic context appended to the routine's
              base prompt (e.g. an alert payload, a ticket ID, a file path).
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: Session metadata including session_id and session_url.
        {
            "claude_code_session_id": str,   # Use to track this run
            "claude_code_session_url": str,  # Open in browser to watch live execution
            "fired_at": str
        }

    Examples:
        - Fire a nightly report routine for an ad-hoc run:
            routine_id="trig_01ABC...", additional_text="Run for March 2026 instead of April"
        - Trigger alert triage from a monitoring webhook:
            routine_id="trig_01XYZ...", additional_text="Sentry alert SEN-4521: NullPointerException in prod"

    Error Handling:
        - Error 401: Invalid routine_bearer_token.
        - Error 404: routine_id does not exist or API trigger is not enabled.
        - Error 429: Daily run limit reached (Pro: 5/day, Max: 15/day, Team/Enterprise: 25/day).

    Note:
        The /fire endpoint is under beta header 'experimental-cc-routine-2026-04-01'.
        Anthropic maintains backward compatibility for the two previous beta versions.
    """
    client = _client(ctx)
    path = f"/v1/claude_code/routines/{params.routine_id}/fire"

    # Routine requests use a per-routine bearer token, NOT the API key
    headers = {
        "Authorization": f"Bearer {params.routine_bearer_token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ROUTINES_BETA,
        "content-type": "application/json",
    }

    body: Dict[str, Any] = {}
    if params.additional_text:
        body["text"] = params.additional_text

    await ctx.log_info("Firing routine", {"routine_id": params.routine_id})
    await ctx.report_progress(0.30, "Sending fire request…")

    try:
        resp = await client.post(path, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _handle_http_error(exc, f"Routine {params.routine_id}")

    await ctx.report_progress(1.0, "Routine fired.")

    session_id = data.get("claude_code_session_id", "")
    session_url = data.get("claude_code_session_url", "")
    fired_at = _utcnow()

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json({**data, "fired_at": fired_at})

    lines = [
        f"## 🚀 Routine `{params.routine_id}` Fired",
        f"- **Session ID:** `{session_id}`",
    ]
    if session_url:
        lines.append(f"- **Live view:** {session_url}")
    lines += [
        f"- **Fired at:** {fired_at}",
        "",
        "Open the live view URL in your browser to watch execution in real time.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: cowork_list_agents
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="cowork_list_agents",
    annotations={
        "title": "List Registered Cowork Agents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def cowork_list_agents(params: ListAgentsInput, ctx: Context) -> str:
    """
    List all Managed Agent definitions registered in your Anthropic organisation.

    Use this to discover existing agent IDs before calling cowork_start_session,
    or to audit which agents have been created.

    Args:
        params (ListAgentsInput):
            - limit (Optional[int]): Max agents to return (default 20, max 100).
            - response_format (ResponseFormat): 'markdown' or 'json'.

    Returns:
        str: List of agent definitions.
        {
            "agents": [
                {
                    "id": str,        # Pass to cowork_start_session as agent_id
                    "name": str,
                    "model": str,
                    "created_at": str
                }
            ],
            "count": int
        }

    Error Handling:
        - Error 401: Invalid API key.
    """
    client = _client(ctx)
    try:
        data = await _get(
            client,
            "/v1/agents",
            _managed_agents_headers(),
            params={"limit": params.limit},
        )
    except Exception as exc:
        return _handle_http_error(exc, "Agents API")

    agents: List[Dict[str, Any]] = data.get("agents", data.get("data", []))

    if params.response_format == ResponseFormat.JSON:
        return _fmt_json({"agents": agents, "count": len(agents)})

    if not agents:
        return (
            "No agent definitions found. "
            "Create one with cowork_create_agent first."
        )

    lines = [f"## Registered Cowork Agents ({len(agents)} found)", ""]
    for agent in agents:
        lines.append(
            f"- **{agent.get('name', 'Unnamed')}** — "
            f"ID: `{agent.get('id')}` | "
            f"Model: {agent.get('model', 'N/A')} | "
            f"Created: {agent.get('created_at', 'N/A')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
