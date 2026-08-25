"""
The agentic loop with detailed step-by-step token usage tracing.

Loop contract:
  * the system prompt is always message[0] and is never trimmed away;
  * only the last MAX_HISTORY_TURNS user/assistant exchanges are replayed;
  * failing tool calls get MAX_QUERY_RETRIES rounds to self-correct;
  * MAX_AGENT_STEPS bounds the loop, so it always terminates;
  * when the budget runs out the model is asked for a final tool-free answer
    rather than being replaced by a canned apology;
  * every LLM round-trip and tool call is written to the workflow log.
"""

import json
import time

from openai import OpenAI

import mcp_manager
import workflow_logger
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_AGENT_STEPS,
    MAX_HISTORY_TURNS,
    MAX_QUERY_RETRIES,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TOOL_OUTPUT_ITEMS,
    SYSTEM_PROMPT,
)

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

APOLOGY_MESSAGE = (
    "I'm sorry — I wasn't able to retrieve that information after several "
    "attempts. Could you try rephrasing your question, or double-check that "
    "the data you're asking about actually exists?"
)

# Appended to a tool result that succeeded but carried no data. Without this the
# model reads an empty MongoDB result as proof the data does not exist, when it
# usually means the database/collection name or a value type was wrong.
EMPTY_RESULT_HINT = (
    "\n\n[system note: this call succeeded but returned no data. That does NOT "
    "prove the data is absent. Before telling the user nothing was found, "
    "verify the database/collection/table and field names (list-collections, "
    "collection-schema, get_schema_info) and check value types — e.g. "
    "student_id may be stored as a number while roll_no reads as a string, so "
    "try both 5 and \"5\".]"
)

# Fields worth forwarding from an assistant message on the next request.
# Everything else the provider echoes back is dropped to keep the payload small
# and avoid proxies rejecting unexpected keys.
_ASSISTANT_PASSTHROUGH_KEYS = ("reasoning_content", "thinking_blocks")


# ------------------------------------------------------------------
# Message plumbing
# ------------------------------------------------------------------
def _parse_tool_arguments(raw_arguments) -> dict:
    """OpenAI-format tool calls always send arguments as a JSON string."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_system_prompt(system_prompt: str | None, history: list[dict] | None) -> str:
    """
    Prefer an explicitly supplied prompt (the app passes the schema-primed one).
    Otherwise recover it from history, so callers that keep the system message
    in history[0] still work. Falls back to the static prompt.
    """
    if system_prompt and system_prompt.strip():
        return system_prompt
    for message in history or []:
        if message.get("role") == "system" and message.get("content"):
            return message["content"]
    return SYSTEM_PROMPT


def _recent_turns(history: list[dict] | None) -> list[dict]:
    """
    The most recent conversation turns, filtered to plain user/assistant text.
    Tool-call scaffolding from an earlier turn is dropped, so a tool_call can
    never arrive without its matching result.
    """
    turns = [
        {"role": message["role"], "content": message["content"]}
        for message in (history or [])
        if message.get("role") in ("user", "assistant") and message.get("content")
    ]
    return turns[-(MAX_HISTORY_TURNS * 2):] if MAX_HISTORY_TURNS > 0 else []


def _build_messages(system_prompt: str, recent: list[dict],
                    user_question: str) -> list[dict]:
    """Pinned system prompt + recent turns + the new question."""
    return (
        [{"role": "system", "content": system_prompt}]
        + recent
        + [{"role": "user", "content": user_question}]
    )


def _sanitize_assistant_message(message) -> dict:
    """Keep only the fields the next request needs, never dropping tool_calls."""
    dumped = message.model_dump(exclude_none=True)
    clean: dict = {"role": dumped.get("role") or "assistant"}

    if dumped.get("content"):
        clean["content"] = dumped["content"]
    if dumped.get("tool_calls"):
        clean["tool_calls"] = dumped["tool_calls"]
    for key in _ASSISTANT_PASSTHROUGH_KEYS:
        if dumped.get(key):
            clean[key] = dumped[key]

    if "content" not in clean and "tool_calls" not in clean:
        clean["content"] = ""
    return clean


def _compact_history(user_question: str, answer: str) -> list[dict]:
    """The two plain messages worth carrying into the next turn."""
    return [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": answer},
    ]


# ------------------------------------------------------------------
# Tool output compression
# ------------------------------------------------------------------
def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (f"{text[:max_chars]}\n[note: output truncated — showing "
            f"{max_chars} of {len(text)} characters]")


def _split_prose_and_json(text: str):
    """
    MongoDB returns one text block per document, which mcp_manager joins with
    newlines — often behind a prose line like "Found 3 documents in ...".
    Separate the two so documents can be counted and trimmed properly instead
    of being cut mid-JSON by a blind character slice.
    """
    prose, documents = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "{[":
            try:
                documents.append(json.loads(stripped))
                continue
            except (TypeError, ValueError):
                pass
        prose.append(stripped)
    return prose, documents


def _render_records(records: list, prose: list[str], max_chars: int,
                    max_items: int) -> str:
    kept = records[:max_items]
    header = "\n".join(prose[:3])
    note = (f"\n[note: showing {len(kept)} of {len(records)} records]"
            if len(records) > len(kept) else "")
    body = json.dumps(kept, default=str)
    budget = max(max_chars - len(header), 500)
    parts = [part for part in (header, _truncate(body, budget)) if part]
    return "\n".join(parts) + note


def compress_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS,
                         max_items: int = MAX_TOOL_OUTPUT_ITEMS) -> str:
    """
    Shrink a tool result to fit the context budget. Anything dropped is called
    out explicitly, so the model never mistakes a trimmed result for the whole
    dataset (or for a syntax error, when JSON was cut in half).
    """
    if not text:
        return "[no output]"

    stripped = text.strip()
    if not stripped:
        return "[empty output]"

    # Whole payload is valid JSON.
    try:
        data = json.loads(stripped)
    except (TypeError, ValueError):
        data = None

    if isinstance(data, list):
        return _render_records(data, [], max_chars, max_items)

    if isinstance(data, dict):
        rendered = json.dumps(data, default=str)
        if len(rendered) <= max_chars:
            return rendered
        keys = list(data)
        kept = keys[:max_items]
        note = (f"\n[note: showing {len(kept)} of {len(keys)} fields; omitted: "
                f"{keys[len(kept):][:10]}]") if len(keys) > len(kept) else ""
        body = json.dumps({key: data[key] for key in kept}, default=str)
        return _truncate(body, max_chars) + note

    # Newline-delimited documents, optionally behind a prose summary line.
    prose, documents = _split_prose_and_json(stripped)
    if documents:
        return _render_records(documents, prose, max_chars, max_items)

    return _truncate(stripped, max_chars)


# ------------------------------------------------------------------
# LLM calls
# ------------------------------------------------------------------
def _create_completion(messages: list[dict], tools: list[dict] | None = None):
    """Omit `tools` entirely when empty — some proxies reject a null/empty list."""
    kwargs = {"model": LLM_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)


def _record_usage(token_trace: dict, response, step_type: str,
                  tools_called: list[str]) -> None:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    step_total = getattr(usage, "total_tokens", 0) or 0

    token_trace["total_prompt_tokens"] += prompt_tokens
    token_trace["total_completion_tokens"] += completion_tokens
    token_trace["total_tokens"] += step_total

    token_trace["steps"].append({
        "step": len(token_trace["steps"]) + 1,
        "type": step_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": step_total,
        "tools_called": tools_called,
    })


def _new_token_trace() -> dict:
    return {
        "steps": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }


def _final_answer_without_tools(messages: list[dict], token_trace: dict,
                                user_question: str, failures: list[str],
                                reason: str, recorder, step: int):
    """
    Last resort: stop offering tools and make the model answer from what it has.
    Far more useful than a canned apology — it can report partial results, or
    explain exactly which lookup failed and why.
    """
    if reason == "step_limit":
        outcome = workflow_logger.OUTCOME_STEP_LIMIT
        nudge = (
            "You have used the maximum number of tool-call rounds. Do not "
            "request any more tools. Answer the user's question now using only "
            "the data already retrieved. If the data is incomplete, say which "
            "part you could not confirm."
        )
    else:
        outcome = workflow_logger.OUTCOME_TOOL_ERRORS
        detail = "\n".join(f"- {failure}" for failure in failures)
        nudge = (
            f"The following tool calls failed:\n{detail}\n\n"
            "Do not request any more tools. Answer using the data you already "
            "have. If you retrieved nothing usable, tell the user plainly what "
            "you tried and what went wrong, and suggest what would help."
        )

    recorder.record_note("forcing tool-free final answer", reason=reason,
                         failures=failures)

    probe = messages + [{"role": "user", "content": nudge}]
    started = time.perf_counter()
    try:
        response = _create_completion(probe)
        duration_ms = int((time.perf_counter() - started) * 1000)
        _record_usage(token_trace, response, "Forced Final Answer", [])
        message = response.choices[0].message
        answer = (message.content or "").strip()
        recorder.record_llm_call(
            step=step, purpose="forced_final_answer", duration_ms=duration_ms,
            usage=getattr(response, "usage", None), tool_calls=[],
            content=message.content, message_count=len(probe),
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        print(f"[warning] forced final answer failed: {exc}")
        recorder.record_llm_error(step=step, duration_ms=duration_ms, error=str(exc))
        answer = ""

    if not answer:
        answer = APOLOGY_MESSAGE
        outcome = workflow_logger.OUTCOME_APOLOGY

    recorder.finish(answer=answer, outcome=outcome)
    return answer, _compact_history(user_question, answer), token_trace


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
async def run_agent(user_question: str, tools: list[dict], history: list[dict],
                    system_prompt: str | None = None, logger=None):
    """
    Run one full turn of the agent while collecting token usage traces and
    writing the complete workflow to the JSON log.

    Returns: (answer_text, compact_history, token_trace)

    `system_prompt` should be the schema-primed prompt. If omitted it is
    recovered from history[role == "system"], then from config.SYSTEM_PROMPT.
    `logger` is a workflow_logger.WorkflowLogger; the shared default is used
    when the caller does not manage its own session.
    """
    resolved_prompt = _resolve_system_prompt(system_prompt, history)
    recent = _recent_turns(history)
    messages = _build_messages(resolved_prompt, recent, user_question)

    logger = logger or workflow_logger.get_logger()
    recorder = logger.start_run(
        question=user_question,
        tools=tools,
        system_prompt=resolved_prompt,
        history_turns=len(recent) // 2,
    )

    token_trace = _new_token_trace()
    error_attempts = 0
    step = 0

    for step in range(1, MAX_AGENT_STEPS + 1):
        started = time.perf_counter()
        try:
            response = _create_completion(messages, tools)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            recorder.record_llm_error(step=step, duration_ms=duration_ms, error=str(exc))
            answer = (f"I couldn't reach the language model, so I wasn't able to "
                      f"answer that. Details: {exc}")
            recorder.finish(answer=answer, outcome=workflow_logger.OUTCOME_LLM_ERROR)
            return answer, _compact_history(user_question, answer), token_trace

        duration_ms = int((time.perf_counter() - started) * 1000)
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])

        _record_usage(
            token_trace,
            response,
            "Tool Selection & Execution" if tool_calls else "Final Answer Generation",
            [call.function.name for call in tool_calls],
        )
        recorder.record_llm_call(
            step=step,
            purpose="tool_selection" if tool_calls else "final_answer",
            duration_ms=duration_ms,
            usage=getattr(response, "usage", None),
            tool_calls=tool_calls,
            content=message.content,
            message_count=len(messages),
        )
        messages.append(_sanitize_assistant_message(message))

        if not tool_calls:
            answer = (message.content or "").strip()
            outcome = (workflow_logger.OUTCOME_ANSWERED if answer
                       else workflow_logger.OUTCOME_APOLOGY)
            answer = answer or APOLOGY_MESSAGE
            recorder.finish(answer=answer, outcome=outcome)
            return answer, _compact_history(user_question, answer), token_trace

        failures: list[str] = []
        for call in tool_calls:
            tool_name = call.function.name
            arguments = _parse_tool_arguments(call.function.arguments)

            tool_started = time.perf_counter()
            result = await mcp_manager.call_tool(tool_name, arguments)
            tool_duration_ms = int((time.perf_counter() - tool_started) * 1000)

            content = compress_tool_output(result["text"])
            empty_hint_added = False
            if not result["success"]:
                failures.append(f"{tool_name}: {result['text'][:300]}")
            elif result.get("empty"):
                content += EMPTY_RESULT_HINT
                empty_hint_added = True

            recorder.record_tool_call(
                step=step,
                tool_name=tool_name,
                requested_arguments=arguments,
                result=result,
                duration_ms=tool_duration_ms,
                output_sent_to_model=content,
                empty_hint_added=empty_hint_added,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            })

        if failures:
            error_attempts += 1
            recorder.record_retry(step=step, attempt=error_attempts,
                                  limit=MAX_QUERY_RETRIES, failures=failures)
            if error_attempts >= MAX_QUERY_RETRIES:
                return _final_answer_without_tools(
                    messages, token_trace, user_question, failures,
                    "tool_errors", recorder, step,
                )

    return _final_answer_without_tools(
        messages, token_trace, user_question, [], "step_limit", recorder, step,
    )
