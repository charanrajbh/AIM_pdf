"""
Streamlit GUI for the school data assistant with Token Trace UI.

Session state holds the system prompt separately from the conversation, so
trimming history can never discard it — losing the system prompt used to strip
the model of the MongoDB database/collection names it needs to query at all.

Every turn is also written to the workflow log (see workflow_logger.py); the
sidebar exposes the live counters and the path to the JSON files.
"""

import asyncio
import json

import streamlit as st

import mcp_manager
import workflow_logger
from agent import run_agent
from config import LOG_ENABLED, MAX_HISTORY_TURNS, SERVERS, build_system_prompt

st.set_page_config(page_title="School Data Assistant", page_icon="🏫", layout="centered")

st.title("School Data Assistant")
st.caption(
    "Ask about structured records (marks, attendance) or unstructured notes "
    "(activities, remarks, projects). The agent automatically figures out "
    "which database and tool to use."
)


def run_async(coro):
    """Safely execute an async coroutine without breaking persistent event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


def connect_to_servers() -> None:
    """
    Discover tools, then read the real schema from both servers and bake it into
    the system prompt. The prompt claims the live schema is authoritative, so it
    has to actually be fetched. The whole discovery phase is logged, since a
    MongoDB outage or a tool-name mismatch shows up here and nowhere else.
    """
    st.session_state.tools = run_async(mcp_manager.discover_tools())
    live_schema = run_async(mcp_manager.discover_schema()) if st.session_state.tools else ""
    st.session_state.live_schema = live_schema
    st.session_state.system_prompt = build_system_prompt(live_schema)

    st.session_state.logger.log_startup(
        diagnostics=mcp_manager.DIAGNOSTICS,
        tools=st.session_state.tools,
        live_schema=live_schema,
        schema_probes=mcp_manager.SCHEMA_PROBE_LOG,
    )


# --- One-time setup per session ---
if "logger" not in st.session_state:
    st.session_state.logger = workflow_logger.WorkflowLogger()

if "tools" not in st.session_state:
    with st.spinner("Connecting to MySQL and MongoDB MCP servers..."):
        connect_to_servers()

# Conversation turns only — the system prompt lives in its own session key.
if "turns" not in st.session_state:
    st.session_state.turns = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # [(role, text, token_trace)]


def render_token_trace(token_trace: dict) -> None:
    """Helper function to render a clean token trace in Streamlit."""
    if not token_trace:
        return

    with st.expander("🪙 Token Usage Trace"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt Tokens", token_trace["total_prompt_tokens"])
        col2.metric("Completion Tokens", token_trace["total_completion_tokens"])
        col3.metric("Total Tokens", token_trace["total_tokens"])

        st.markdown("---")
        st.markdown("**Step-by-Step Breakdown:**")

        for step in token_trace.get("steps", []):
            tools_str = (f" → `Tools: {', '.join(step['tools_called'])}`"
                         if step["tools_called"] else "")
            st.markdown(
                f"- **Step {step['step']} ({step['type']}){tools_str}:** "
                f"`Prompt: {step['prompt_tokens']}` | "
                f"`Completion: {step['completion_tokens']}` | "
                f"`Subtotal: {step['total_tokens']}`"
            )


def render_workflow_trace(run: dict) -> None:
    """Render one logged run — the exact tool calls, arguments and outcomes."""
    if not run:
        return

    counters = run.get("counters", {})
    label = (f"🔎 Workflow Trace — {run.get('outcome')} · "
             f"{counters.get('tool_calls', 0)} tool call(s) · "
             f"{run.get('duration_ms', 0)} ms")

    with st.expander(label):
        cols = st.columns(4)
        cols[0].metric("LLM calls", counters.get("llm_calls", 0))
        cols[1].metric("Tool calls", counters.get("tool_calls", 0))
        cols[2].metric("Tool errors", counters.get("tool_errors", 0))
        cols[3].metric("Empty results", counters.get("empty_results", 0))

        for step in run.get("steps", []):
            kind = step.get("kind")

            if kind == "llm_call":
                requested = ", ".join(tool["name"] for tool in step.get("requested_tools", []))
                st.markdown(
                    f"**{step['seq']}. LLM** ({step.get('purpose')}) — "
                    f"{step.get('duration_ms')} ms, "
                    f"{step.get('usage', {}).get('total_tokens', 0)} tokens"
                    + (f" → requested `{requested}`" if requested else "")
                )

            elif kind == "tool_call":
                if not step.get("success"):
                    icon = "❌"
                elif step.get("empty"):
                    icon = "⚠️"
                else:
                    icon = "✅"
                st.markdown(
                    f"**{step['seq']}. {icon} `{step.get('tool')}`** on "
                    f"*{step.get('server')}* — {step.get('duration_ms')} ms, "
                    f"{step.get('output_chars', 0)} chars out"
                )
                st.code(json.dumps(step.get("requested_arguments") or {}, indent=2),
                        language="json")
                if step.get("injected_arguments"):
                    st.caption(f"auto-filled by mcp_manager: "
                               f"{', '.join(step['injected_arguments'])}")
                if step.get("coerced_arguments"):
                    st.caption(f"JSON-string arguments parsed: "
                               f"{', '.join(step['coerced_arguments'])}")
                if step.get("empty"):
                    st.caption("Returned no data — the model was told this is not "
                               "proof the data is absent.")

            elif kind == "retry":
                state = "exhausted" if step.get("exhausted") else "retrying"
                st.markdown(f"**{step['seq']}. 🔁 Retry** {step.get('attempt')}/"
                            f"{step.get('limit')} ({state})")
                for failure in step.get("failures", []):
                    st.caption(failure)

            elif kind == "llm_error":
                st.markdown(f"**{step['seq']}. 🚫 LLM error** — {step.get('error')}")

            elif kind == "note":
                st.markdown(f"**{step['seq']}. ℹ️ {step.get('message')}**")


def render_server_status() -> None:
    """
    Show per-server health, and specifically call out a reachable server whose
    tools were all dropped by the allowed_tools filter — that used to fail
    silently and left the model with nothing to call.
    """
    for server_key, server in SERVERS.items():
        report = mcp_manager.DIAGNOSTICS.get(server_key, {})
        registered = report.get("registered", [])
        icon = "🟢" if registered else "🔴"
        st.markdown(f"{icon} **{server['label']}**  \n{len(registered)} tool(s) available")

        if registered:
            st.caption("Tools: " + ", ".join(registered))
            missing = report.get("allowed_but_missing") or []
            if missing:
                st.warning(f"Not exposed by the server: {', '.join(missing)}")
        elif not report.get("reachable"):
            st.error(f"Unreachable at {server['url']}\n\n{report.get('error') or 'no response'}")
        else:
            offered = report.get("offered_but_filtered") or []
            st.error(
                "Server is reachable but none of its tools matched "
                f"`allowed_tools` in config.py. It offers: "
                f"{', '.join(offered) if offered else '(nothing)'}"
            )


def render_monitoring() -> None:
    """Rolling counters from logs/metrics.json plus the log file locations."""
    logger = st.session_state.logger

    if not LOG_ENABLED or not logger.enabled:
        st.caption("Workflow logging is disabled "
                   "(set WORKFLOW_LOG_ENABLED=true to enable).")
        return

    metrics = logger.read_metrics()
    runs = max(metrics.get("runs", 0), 1)

    col1, col2 = st.columns(2)
    col1.metric("Questions logged", metrics.get("runs", 0))
    col2.metric("Tool calls", metrics.get("tool_calls", 0))
    col1.metric("Tool errors", metrics.get("tool_errors", 0))
    col2.metric("Empty results", metrics.get("empty_results", 0))
    col1.metric("Avg tokens / question", metrics.get("total_tokens", 0) // runs)
    col2.metric("Avg latency", f"{metrics.get('run_duration_ms_total', 0) // runs} ms")

    outcomes = metrics.get("outcomes") or {}
    if outcomes:
        st.caption("Outcomes: " + ", ".join(f"{key}={value}"
                                            for key, value in sorted(outcomes.items())))

    failures = metrics.get("tool_failures") or {}
    if failures:
        st.warning("Failing tools: " + ", ".join(f"{key} ×{value}"
                                                 for key, value in sorted(failures.items())))

    st.caption(f"**Session:** `{logger.session_id}`")
    st.caption(f"**Stream:** `{logger.stream_path}`")
    st.caption(f"**This session:** `{logger.session_path}`")
    st.caption(f"**Metrics:** `{logger.metrics_path}`")

    try:
        with open(logger.session_path, "rb") as handle:
            st.download_button("⬇️ Download session JSON", handle.read(),
                               file_name=f"{logger.session_id}.json",
                               mime="application/json")
    except OSError:
        st.caption("(no session file written yet)")


# --- Sidebar: connection status, monitoring and controls ---
with st.sidebar:
    st.subheader("Data sources")
    render_server_status()

    st.divider()
    if st.button("🔄 Re-check connections"):
        with st.spinner("Reconnecting..."):
            connect_to_servers()
        st.rerun()

    if st.button("🗑️ Clear conversation"):
        st.session_state.turns = []
        st.session_state.display_messages = []
        st.rerun()

    with st.expander("🔍 Live schema sent to the model"):
        st.code(st.session_state.get("live_schema") or "(not available)", language="text")

    st.divider()
    st.subheader("Monitoring")
    render_monitoring()

if not st.session_state.tools:
    st.error(
        "No tools discovered from either server. Make sure both MCP servers "
        "are running, then click **Re-check connections** in the sidebar."
    )

# --- Render existing conversation ---
for item in st.session_state.display_messages:
    role, text = item[0], item[1]
    trace = item[2] if len(item) > 2 else None
    run = item[3] if len(item) > 3 else None

    with st.chat_message(role):
        st.markdown(text)
        if role == "assistant":
            render_token_trace(trace)
            render_workflow_trace(run)

# --- New user input ---
user_question = st.chat_input("Ask a question about students...")

if user_question:
    st.session_state.display_messages.append(("user", user_question, None, None))
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, _compact, token_trace = run_async(
                run_agent(
                    user_question,
                    st.session_state.tools,
                    st.session_state.turns,
                    system_prompt=st.session_state.system_prompt,
                    logger=st.session_state.logger,
                )
            )
        run = (st.session_state.logger.recent_runs(1) or [None])[0]
        st.markdown(answer)
        render_token_trace(token_trace)
        render_workflow_trace(run)

    st.session_state.display_messages.append(("assistant", answer, token_trace, run))

    # Trim conversation turns only. The system prompt is not stored here, so it
    # can never be trimmed away.
    st.session_state.turns.append({"role": "user", "content": user_question})
    st.session_state.turns.append({"role": "assistant", "content": answer})
    st.session_state.turns = st.session_state.turns[-(MAX_HISTORY_TURNS * 2):]
