#!/usr/bin/env python3
"""cowork_trigger_mcp — Production MCP Server for Claude.ai custom connector."""

import contextlib, json, logging, os, sys
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx, uvicorn
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger("cowork_trigger_mcp")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
DEFAULT_MODEL = os.environ.get("COWORK_DEFAULT_MODEL", "claude-sonnet-4-6")
MCP_PORT = int(os.environ.get("MCP_PORT", "8080"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120"))

if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY is not set.")

def _headers():
    return {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
            "anthropic-beta": "managed-agents-2026-04-01", "content-type": "application/json"}

_http_client: Optional[httpx.AsyncClient] = None

def _client(ctx: Context) -> httpx.AsyncClient:
    assert _http_client is not None
    return _http_client

mcp = FastMCP("cowork_trigger_mcp", host="0.0.0.0", port=MCP_PORT)

def _utcnow(): return datetime.now(timezone.utc).isoformat()

def _err(exc: Exception, resource: str = "resource") -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        try: detail = exc.response.json().get("error", {}).get("message", exc.response.text[:500])
        except: detail = exc.response.text[:500]
        hints = {401:"Check ANTHROPIC_API_KEY.", 403:f"No access to {resource}.",
                 404:f"{resource} not found.", 422:f"Bad request: {detail}",
                 429:f"Rate limited. Retry after {exc.response.headers.get('retry-after','?')}s."}
        msg = f"Error {code}: {hints.get(code, detail)}"
        logger.error("API error %s — %s", msg, exc.response.text[:500])
        return msg
    msg = f"Error ({type(exc).__name__}): {str(exc)[:300]}"
    logger.error(msg); return msg

class CreateAgentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., min_length=3, max_length=100, description="Agent name.")
    system_prompt: str = Field(..., min_length=20, max_length=20000, description="System prompt.")
    model: Optional[str] = Field(default=None, description=f"Model ID. Default: {DEFAULT_MODEL}.")

class StartSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent_id: str = Field(..., min_length=5, max_length=128, description="Agent ID from cowork_create_agent.")
    task: str = Field(..., min_length=10, max_length=8000, description="Task to complete.")
    title: Optional[str] = Field(default=None, max_length=200)
    @field_validator("task")
    @classmethod
    def not_blank(cls, v):
        if not v.strip(): raise ValueError("blank")
        return v.strip()

class SendEventInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    session_id: str = Field(..., min_length=5, max_length=128, description="Session ID.")
    message: str = Field(..., min_length=1, max_length=8000, description="Message to send.")

class GetSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=5, max_length=128, description="Session ID.")
    max_events: int = Field(default=20, ge=1, le=100)

class ListSessionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    limit: int = Field(default=20, ge=1, le=100)

class ListAgentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=100)

class UpdateAgentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    agent_id: str = Field(..., min_length=5, max_length=128)
    system_prompt: Optional[str] = Field(default=None, max_length=20000)
    name: Optional[str] = Field(default=None, max_length=100)

@mcp.tool(name="cowork_create_agent", annotations={"title":"Create Cowork Agent"})
async def cowork_create_agent(params: CreateAgentInput, ctx: Context) -> str:
    """Create a Managed Agent. Returns agent_id — store it and pass to cowork_start_session."""
    client = _client(ctx)
    try:
        resp = await client.post("/v1/agents", headers=_headers(), json={
            "name": params.name, "model": params.model or DEFAULT_MODEL,
            "system": params.system_prompt, "tools": [{"type": "agent_toolset_20260401"}]})
        resp.raise_for_status(); data = resp.json()
    except Exception as exc: return _err(exc, "Agents API")
    logger.info("Agent created: %s", data.get("id"))
    return f"## ✅ Agent Created\n- **ID:** `{data.get('id')}`  ← save for cowork_start_session\n- **Name:** {data.get('name')}\n- **Model:** {data.get('model')}"

@mcp.tool(name="cowork_update_agent", annotations={"title":"Update Cowork Agent"})
async def cowork_update_agent(params: UpdateAgentInput, ctx: Context) -> str:
    """Update an agent's name or system prompt. Creates a new version."""
    client = _client(ctx)
    body = {k:v for k,v in {"system":params.system_prompt,"name":params.name}.items() if v}
    if not body: return "Error: provide system_prompt or name."
    try:
        resp = await client.post(f"/v1/agents/{params.agent_id}", headers=_headers(), json=body)
        resp.raise_for_status(); data = resp.json()
    except Exception as exc: return _err(exc, f"Agent {params.agent_id}")
    return f"## ✅ Updated `{params.agent_id}` → version {data.get('version')}"

@mcp.tool(name="cowork_list_agents", annotations={"title":"List Cowork Agents","readOnlyHint":True})
async def cowork_list_agents(params: ListAgentsInput, ctx: Context) -> str:
    """List all registered Managed Agents in your organisation."""
    client = _client(ctx)
    try:
        resp = await client.get("/v1/agents", headers=_headers(), params={"limit":params.limit})
        resp.raise_for_status(); data = resp.json()
    except Exception as exc: return _err(exc, "Agents API")
    agents = data.get("data", [])
    if not agents: return "No agents found. Create one with cowork_create_agent."
    lines = [f"## Agents ({len(agents)})", ""]
    for a in agents:
        model_id = a.get("model", {}).get("id", "N/A") if isinstance(a.get("model"), dict) else str(a.get("model","N/A"))
        lines.append(f"- **{a.get('name','Unnamed')}** — `{a.get('id')}` | {model_id}")
    return "\n".join(lines)

@mcp.tool(name="cowork_start_session", annotations={"title":"Start Cowork Session"})
async def cowork_start_session(params: StartSessionInput, ctx: Context) -> str:
    """Start a Managed Agent session. agent_id must come from cowork_create_agent."""
    client = _client(ctx)
    await ctx.report_progress(0.2, "Creating session...")
    try:
        resp = await client.post("/v1/sessions", headers=_headers(),
            json={"agent": params.agent_id, **({"title":params.title} if params.title else {})})
        resp.raise_for_status(); session = resp.json()
    except Exception as exc: return _err(exc, "Sessions API")
    sid = session.get("id","")
    await ctx.report_progress(0.6, "Sending task...")
    try:
        er = await client.post(f"/v1/sessions/{sid}/events", headers=_headers(),
            json={"type":"user","content":[{"type":"text","text":params.task}]})
        er.raise_for_status()
    except Exception as exc:
        return f"Session `{sid}` created but task failed: {_err(exc,'Events API')}"
    await ctx.report_progress(1.0, "Running.")
    lines = ["## ✅ Session Started", f"- **Session ID:** `{sid}`", f"- **Agent:** `{params.agent_id}`",
             f"- **Started:** {_utcnow()}", "", "Poll with `cowork_get_session_status`."]
    if session.get("session_url"): lines.insert(3, f"- **Live:** {session['session_url']}")
    return "\n".join(lines)

@mcp.tool(name="cowork_get_session_status", annotations={"title":"Get Session Status","readOnlyHint":True})
async def cowork_get_session_status(params: GetSessionInput, ctx: Context) -> str:
    """Poll a session's status and recent output events."""
    client = _client(ctx)
    try:
        resp = await client.get(f"/v1/sessions/{params.session_id}", headers=_headers())
        resp.raise_for_status(); session = resp.json()
    except Exception as exc: return _err(exc, f"Session {params.session_id}")
    try:
        er = await client.get(f"/v1/sessions/{params.session_id}/events",
            headers=_headers(), params={"limit":params.max_events})
        er.raise_for_status(); events = er.json().get("data",[])
    except: events = []
    st = session.get("status","?")
    icon = {"running":"⏳","completed":"✅","failed":"❌","interrupted":"⚠️"}.get(st,"🔵")
    lines = [f"## {icon} `{params.session_id}`", f"- **Status:** {st}",
             f"- **Updated:** {session.get('updated_at','N/A')}"]
    if events:
        lines += ["", f"### Events ({len(events)})"]
        for ev in events:
            text = " ".join(b.get("text","")[:200] for b in ev.get("content",[])
                            if isinstance(b,dict) and b.get("type")=="text")
            lines.append(f"- **[{ev.get('type','?')}]** {text or '*(non-text)*'}")
    return "\n".join(lines)

@mcp.tool(name="cowork_send_event", annotations={"title":"Send Event to Session"})
async def cowork_send_event(params: SendEventInput, ctx: Context) -> str:
    """Send a follow-up message to steer a running session."""
    client = _client(ctx)
    try:
        resp = await client.post(f"/v1/sessions/{params.session_id}/events", headers=_headers(),
            json={"type":"user","content":[{"type":"text","text":params.message}]})
        resp.raise_for_status(); data = resp.json()
    except Exception as exc: return _err(exc, f"Session {params.session_id}")
    return f"## ✅ Event Sent\n- **ID:** `{data.get('id','N/A')}`\n- **At:** {_utcnow()}"

@mcp.tool(name="cowork_list_sessions", annotations={"title":"List Sessions","readOnlyHint":True})
async def cowork_list_sessions(params: ListSessionsInput, ctx: Context) -> str:
    """List recent Managed Agent sessions."""
    client = _client(ctx)
    q = {"limit":params.limit, **({"agent_id":params.agent_id} if params.agent_id else {}),
         **({"status":params.status} if params.status else {})}
    try:
        resp = await client.get("/v1/sessions", headers=_headers(), params=q)
        resp.raise_for_status(); data = resp.json()
    except Exception as exc: return _err(exc, "Sessions API")
    sessions = data.get("data",[])
    if not sessions: return "No sessions found."
    icon = {"running":"⏳","completed":"✅","failed":"❌","interrupted":"⚠️"}
    lines = [f"## Sessions ({len(sessions)})", ""]
    for s in sessions:
        st = s.get("status","?")
        lines.append(f"{icon.get(st,'🔵')} `{s.get('id')}` — **{st}** | {s.get('updated_at','N/A')}")
    return "\n".join(lines)

# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp_app = mcp.streamable_http_app()   # route is /mcp
    session_mgr = mcp.session_manager

    @contextlib.asynccontextmanager
    async def lifespan(app):
        global _http_client
        async with httpx.AsyncClient(base_url=ANTHROPIC_BASE_URL,
                timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            _http_client = client
            logger.info("started — model=%s port=%d", DEFAULT_MODEL, MCP_PORT)
            async with session_mgr.run():
                yield
        _http_client = None
        logger.info("shutdown.")

    # Health endpoint
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status":"ok","service":"cowork_trigger_mcp","mcp_url":"/mcp"})

    # ASGI app: health at / and /health, MCP app handles everything else
    # (including /mcp and /mcp/ — we strip trailing slash before delegating)
    async def root_app(scope, receive, send):
        if scope["type"] == "lifespan":
            # Run our lifespan manually
            from starlette.applications import Starlette
            sl = Starlette(lifespan=lifespan, routes=[])
            await sl(scope, receive, send)
            return

        path = scope.get("path", "/")
        # Normalise: /mcp/ → /mcp
        if path != "/" and path.endswith("/"):
            scope = dict(scope)
            scope["path"] = path.rstrip("/")
            scope["raw_path"] = scope["path"].encode()

        if scope.get("type") == "http" and scope.get("path") in ("/", "/health"):
            req = Request(scope, receive)
            resp = await health(req)
            await resp(scope, receive, send)
        else:
            await mcp_app(scope, receive, send)

    uvicorn.run(root_app, host="0.0.0.0", port=MCP_PORT, log_level="info")
