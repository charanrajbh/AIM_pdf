"""
Handles all MCP plumbing: connecting to each configured server, discovering
its tools, and routing a tool call to whichever server registered that tool
name. Nothing here decides *which* tool to call — that's the agent's job.
Opens a fresh connection per call rather than holding sessions open, which
is simpler and avoids stale-session issues for a small number of tools.

Beyond plain routing this module also:
  * records why a server produced no tools (unreachable vs. name mismatch),
    instead of failing silently;
  * auto-fills server-level default arguments the model omitted (notably the
    MongoDB `database` name, which every MongoDB tool requires);
  * repairs object/array arguments the model sent as JSON strings;
  * distinguishes "errored" from "succeeded but returned nothing", because a
    MongoDB miss looks like success and would otherwise be read as "no data";
  * probes both servers for their real schema so the prompt can be primed.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

from config import (
    MAX_SCHEMA_BLOCK_CHARS,
    MONGODB_COLLECTIONS,
    MONGODB_DATABASE,
    SERVER_CONNECT_TIMEOUT,
    SERVERS,
    TOOL_CALL_TIMEOUT,
)

# Populated by discover_tools(); maps tool name -> server key ("mongodb"/"mysql")
TOOL_TO_SERVER: dict[str, str] = {}

# Tool name -> JSON Schema of its arguments. Used to decide which defaults are
# applicable and which arguments need JSON coercion.
TOOL_SCHEMAS: dict[str, dict] = {}

# Per-server discovery report, surfaced in the UI so a silent mismatch between
# `allowed_tools` and the server's real tool names is visible.
DIAGNOSTICS: dict[str, dict] = {}

# Outcome of each tool call made by discover_schema(), recorded for the workflow
# log. Populated fresh on every discover_schema() call.
SCHEMA_PROBE_LOG: list[dict] = []


# ------------------------------------------------------------------
# Connection helpers
# ------------------------------------------------------------------
def _connect(server_key: str):
    """Return the right async context manager for a server's transport type."""
    server = SERVERS[server_key]
    if server["transport"] == "streamable_http":
        return streamablehttp_client(server["url"])
    elif server["transport"] == "sse":
        return sse_client(server["url"])
    raise ValueError(f"Unknown transport for server '{server_key}': {server['transport']}")


@asynccontextmanager
async def _session(server_key: str):
    """Open a transport, wrap it in an initialized MCP session, yield it."""
    async with _connect(server_key) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def _run_on_server(server_key: str, operation, timeout: float):
    """
    Run `operation(session)` against a server under a wall-clock timeout.
    The whole connect/initialize/call sequence runs inside one task so anyio's
    cancel scopes unwind cleanly if the timeout fires.
    """

    async def _inner():
        async with _session(server_key) as session:
            return await operation(session)

    return await asyncio.wait_for(_inner(), timeout=timeout)


def _describe_error(exc: BaseException) -> str:
    """Flatten nested/grouped exceptions into one readable line."""
    if isinstance(exc, asyncio.TimeoutError):
        return "timed out"

    parts: list[str] = []

    def walk(err: BaseException) -> None:
        nested = getattr(err, "exceptions", None)
        if nested:
            for sub in nested:
                walk(sub)
        else:
            parts.append(f"{type(err).__name__}: {err}".strip().rstrip(":"))

    walk(exc)
    # dict.fromkeys keeps order while dropping duplicate messages
    return "; ".join(dict.fromkeys(parts)) or repr(exc)


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------
async def discover_tools() -> list[dict]:
    """
    Connect to every configured server, list its tools, and build the combined
    tool catalog in OpenAI function-calling format, keeping only tools listed
    in each server's 'allowed_tools'.

    Anything dropped by that filter is recorded in DIAGNOSTICS and logged. A
    reachable server that contributes zero tools is the failure mode that used
    to be invisible: the model simply had nothing to call.
    """
    catalog: list[dict] = []
    TOOL_TO_SERVER.clear()
    TOOL_SCHEMAS.clear()
    DIAGNOSTICS.clear()

    for server_key, server in SERVERS.items():
        allowed = list(server.get("allowed_tools", []))
        report = {
            "label": server.get("label", server_key),
            "url": server["url"],
            "reachable": False,
            "error": None,
            "registered": [],
            "offered_but_filtered": [],
            "allowed_but_missing": [],
        }
        DIAGNOSTICS[server_key] = report

        try:
            tools_result = await _run_on_server(
                server_key, lambda s: s.list_tools(), SERVER_CONNECT_TIMEOUT
            )
        except Exception as exc:
            report["error"] = _describe_error(exc)
            print(f"[warning] Could not reach '{server_key}' server at "
                  f"{server['url']}: {report['error']}")
            continue

        report["reachable"] = True
        allowed_set = set(allowed)

        for tool in tools_result.tools:
            if tool.name not in allowed_set:
                report["offered_but_filtered"].append(tool.name)
                continue

            schema = tool.inputSchema or {"type": "object", "properties": {}}
            TOOL_TO_SERVER[tool.name] = server_key
            TOOL_SCHEMAS[tool.name] = schema
            report["registered"].append(tool.name)
            catalog.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            })

        report["allowed_but_missing"] = sorted(allowed_set - set(report["registered"]))

        if not report["registered"]:
            print(f"[error] '{server_key}' is reachable but NONE of its tools matched "
                  f"allowed_tools={sorted(allowed_set)}. The server offers: "
                  f"{report['offered_but_filtered']}. Update allowed_tools in config.py "
                  f"— the model currently has no way to query this database.")
        elif report["allowed_but_missing"]:
            print(f"[warning] '{server_key}' does not expose these allowed tools: "
                  f"{report['allowed_but_missing']}")

    return catalog


# ------------------------------------------------------------------
# Argument repair
# ------------------------------------------------------------------
def _schema_properties(tool_name: str) -> dict:
    return (TOOL_SCHEMAS.get(tool_name) or {}).get("properties") or {}


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _apply_default_arguments(server_key: str, tool_name: str, arguments: dict):
    """
    Fill in server-level defaults the model left out — chiefly the MongoDB
    `database` name. Only applied when the tool actually accepts the argument
    and the model supplied nothing for it; an explicit value always wins.
    """
    defaults = SERVERS.get(server_key, {}).get("default_arguments") or {}
    if not defaults:
        return arguments, []

    properties = _schema_properties(tool_name)
    repaired = dict(arguments)
    injected: list[str] = []

    for key, value in defaults.items():
        if key not in properties:
            continue
        if _is_blank(repaired.get(key)):
            repaired[key] = value
            injected.append(key)

    return repaired, injected


def _coerce_structured_arguments(tool_name: str, arguments: dict):
    """
    Models sometimes send an object/array argument as a JSON *string*
    (`filter='{"student_id": 5}'`). Parse those back into real structures so
    the server does not reject them.
    """
    properties = _schema_properties(tool_name)
    repaired = dict(arguments)
    coerced: list[str] = []

    for key, value in list(repaired.items()):
        if not isinstance(value, str):
            continue
        expected = (properties.get(key) or {}).get("type")
        if expected not in ("object", "array"):
            continue
        text = value.strip()
        if not text or text[0] not in "{[":
            continue
        try:
            repaired[key] = json.loads(text)
            coerced.append(key)
        except (TypeError, ValueError):
            pass

    return repaired, coerced


# ------------------------------------------------------------------
# Result interpretation
# ------------------------------------------------------------------
_ERROR_RE = re.compile(
    r"^\s*(error|exception|traceback|failed|mongoservererror|mongoerror|"
    r"operationfailure|writeerror)\b",
    re.IGNORECASE,
)

_EMPTY_LITERALS = {"", "[]", "{}", "null", "none", "no results", "no documents",
                   "no rows", "no records", "empty set", "[no output]"}

_EMPTY_RE = re.compile(
    r"\b(found|returned|matched|fetched)\s+0\b"
    r"|\b0\s+(document|documents|result|results|row|rows|record|records)\b"
    r"|^\s*no\s+(document|documents|result|results|row|rows|record|records|matching|data)\b",
    re.IGNORECASE,
)


def _extract_text(result) -> str:
    """Join every text block of a tool result; fall back to repr for others."""
    blocks = getattr(result, "content", None) or []
    parts = []
    for block in blocks:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "\n".join(p for p in parts if p) if parts else ""


def _looks_like_error(text: str) -> bool:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return bool(_ERROR_RE.match(first_line))


def _looks_empty(text: str) -> bool:
    """
    True when the call succeeded but carried no data. A MongoDB query against
    the wrong database or collection lands here, not in the error path — which
    is exactly why it used to be reported to the user as "the data isn't there".
    """
    stripped = text.strip()
    if stripped.lower() in _EMPTY_LITERALS:
        return True
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        pass
    else:
        if isinstance(parsed, (list, dict)) and not parsed:
            return True
        if isinstance(parsed, (int, float)) and parsed == 0:
            return True
    return bool(_EMPTY_RE.search(stripped))


def _result(success: bool, text: str, server: str | None, *, empty: bool = False,
            injected: list[str] | None = None, coerced: list[str] | None = None) -> dict:
    return {
        "success": success,
        "text": text,
        "server": server,
        "empty": empty,
        "injected_arguments": injected or [],
        "coerced_arguments": coerced or [],
    }


# ------------------------------------------------------------------
# Tool invocation
# ------------------------------------------------------------------
async def call_tool(tool_name: str, arguments: dict | None = None) -> dict:
    """
    Call a tool by name, routing to whichever server registered it.

    Returns a dict with:
      success            — False for transport failures and server-reported errors
      text               — the tool's textual output (or the failure reason)
      server             — which server handled it, if known
      empty              — True when the call succeeded but returned no data
      injected_arguments — defaults filled in on the model's behalf
      coerced_arguments  — JSON-string arguments parsed into real structures
    """
    arguments = dict(arguments or {})

    server_key = TOOL_TO_SERVER.get(tool_name)
    if server_key is None:
        known = ", ".join(sorted(TOOL_TO_SERVER)) or "none"
        return _result(
            False,
            f"Unknown or disallowed tool '{tool_name}' — it isn't registered or "
            f"permitted on any connected server. Available tools: {known}.",
            None,
        )

    if tool_name not in SERVERS.get(server_key, {}).get("allowed_tools", []):
        return _result(False, f"Tool '{tool_name}' is not allowed on server '{server_key}'.",
                       server_key)

    arguments, injected = _apply_default_arguments(server_key, tool_name, arguments)
    arguments, coerced = _coerce_structured_arguments(tool_name, arguments)
    if injected:
        filled = {key: arguments[key] for key in injected}
        print(f"[info] '{tool_name}': auto-filled omitted argument(s) {filled}")
    if coerced:
        print(f"[info] '{tool_name}': parsed JSON-string argument(s) {coerced}")

    try:
        result = await _run_on_server(
            server_key,
            lambda s: s.call_tool(tool_name, arguments=arguments),
            TOOL_CALL_TIMEOUT,
        )
    except Exception as exc:
        return _result(
            False,
            f"Tool call to '{tool_name}' on '{server_key}' failed: {_describe_error(exc)}",
            server_key,
            injected=injected,
            coerced=coerced,
        )

    text = _extract_text(result)
    is_error = bool(getattr(result, "isError", False)) or _looks_like_error(text)

    return _result(
        not is_error,
        text,
        server_key,
        empty=(not is_error) and _looks_empty(text),
        injected=injected,
        coerced=coerced,
    )


# ------------------------------------------------------------------
# Schema probing (fulfils the "authoritative schema" the prompt promises)
# ------------------------------------------------------------------
def _clip(text: str, limit: int = MAX_SCHEMA_BLOCK_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... schema output truncated]"


async def _probe(tool_name: str, arguments: dict) -> dict:
    """call_tool wrapper that records the outcome into SCHEMA_PROBE_LOG."""
    result = await call_tool(tool_name, arguments)
    SCHEMA_PROBE_LOG.append({
        "tool": tool_name,
        "arguments": arguments,
        "server": result.get("server"),
        "success": result.get("success"),
        "empty": result.get("empty"),
        "output_chars": len(result.get("text") or ""),
        "error": None if result.get("success") else (result.get("text") or "")[:500],
    })
    return result


def _first_supported_param(tool_name: str, candidates: tuple[str, ...]) -> str | None:
    """Pick whichever argument name this tool actually declares."""
    properties = _schema_properties(tool_name)
    for name in candidates:
        if name in properties:
            return name
    return None


def _parse_collection_names(text: str) -> list[str]:
    """
    Best-effort extraction of collection names from `list-collections` output.
    Only trusts real JSON — no guessing from prose — and returns [] when the
    shape is unfamiliar so callers fall back to the configured names.
    """
    candidates = []
    stripped = (text or "").strip()
    if not stripped:
        return []

    payloads = []
    try:
        payloads.append(json.loads(stripped))
    except (TypeError, ValueError):
        for line in stripped.splitlines():
            line = line.strip()
            if line[:1] in ("{", "["):
                try:
                    payloads.append(json.loads(line))
                except (TypeError, ValueError):
                    continue

    for payload in payloads:
        if isinstance(payload, dict):
            payload = payload.get("collections", payload)
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("collection")
                if isinstance(name, str):
                    candidates.append(name)

    return list(dict.fromkeys(c for c in candidates if c and not c.startswith("system.")))


async def _mysql_schema_block() -> str:
    """
    Read the real MySQL table/column names. One information_schema query gets
    everything without any output parsing; get_schema_info is the fallback.
    """
    parts = []

    if "execute_sql" in TOOL_TO_SERVER:
        param = _first_supported_param("execute_sql", ("query", "sql", "statement", "command"))
        if param:
            probe = await _probe("execute_sql", {param: (
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )})
            if probe["success"] and not probe["empty"]:
                parts.append("Tables and columns (table, column, type):\n"
                             + _clip(probe["text"]))

    if not parts and "get_schema_info" in TOOL_TO_SERVER:
        probe = await _probe("get_schema_info", {})
        if probe["success"] and not probe["empty"]:
            parts.append(_clip(probe["text"]))

    return "\n\n".join(parts)


async def _mongodb_schema_block() -> str:
    """Read the real MongoDB collection names and their inferred field shapes."""
    parts = []
    collections: list[str] = []

    if "list-collections" in TOOL_TO_SERVER:
        probe = await _probe("list-collections", {"database": MONGODB_DATABASE})
        if probe["success"]:
            parts.append(f"Collections in database `{MONGODB_DATABASE}`:\n"
                         + _clip(probe["text"]))
            collections = _parse_collection_names(probe["text"])

    # Configured collections come first so the ones this app relies on are
    # always described, even if list-collections output could not be parsed.
    for name in MONGODB_COLLECTIONS:
        if name not in collections:
            collections.append(name)

    if "collection-schema" in TOOL_TO_SERVER:
        for name in collections[:5]:
            probe = await _probe("collection-schema", {
                "database": MONGODB_DATABASE,
                "collection": name,
            })
            if probe["success"] and not probe["empty"]:
                parts.append(f"Fields of `{MONGODB_DATABASE}.{name}`:\n"
                             + _clip(probe["text"]))

    return "\n\n".join(parts)


async def discover_schema() -> str:
    """
    Probe both servers for their actual names and return one text block for the
    system prompt. Never raises — a source that cannot be probed is simply
    reported as such, and the declared schema in config.py still applies.
    """
    blocks = []
    SCHEMA_PROBE_LOG.clear()
    probes = (
        ("mysql", "MySQL", _mysql_schema_block),
        ("mongodb", "MongoDB", _mongodb_schema_block),
    )

    for server_key, title, probe in probes:
        if server_key not in SERVERS:
            continue
        registered = DIAGNOSTICS.get(server_key, {}).get("registered")
        if not registered:
            blocks.append(f"### {title}\n(no tools available — server unreachable "
                          f"or no allowed tool matched)")
            continue
        try:
            body = await probe()
        except Exception as exc:
            body = f"(schema probe failed: {_describe_error(exc)})"
        blocks.append(f"### {title}\n{body or '(schema probe returned nothing)'}")

    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------
async def check_server_status() -> dict[str, bool]:
    """Quick reachability check for each configured server, for UI display."""
    status = {}
    for server_key in SERVERS:
        try:
            await _run_on_server(server_key, lambda s: s.list_tools(),
                                 SERVER_CONNECT_TIMEOUT)
            status[server_key] = True
        except Exception:
            status[server_key] = False
    return status
