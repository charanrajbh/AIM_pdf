"""
Workflow monitoring and logging.

Every question the agent answers is recorded as one JSON document describing the
complete workflow: each LLM round-trip, each tool call with its *final*
arguments (after defaults were injected), whether the call errored or came back
empty, token usage, timings, and the final answer.

Three artefacts are written under LOG_DIR (default: ./logs):

  workflow.jsonl            Append-only event stream, one JSON object per line
                            (startup records and completed runs). Best for
                            tailing, grepping, or shipping to a log collector.
  sessions/<session>.json   The complete workflow for one app session, pretty
                            printed, rewritten atomically after every turn.
  metrics.json              Rolling aggregate counters across all sessions.

Logging is strictly best-effort: every public method swallows its own errors so
a logging problem can never break the agent loop.

NOTE: these files contain the questions asked, the queries issued, and the rows
and documents returned — i.e. real student data. Treat them as sensitive and
keep the logs/ directory out of version control. Set LOG_RAW_TOOL_OUTPUT=False
in config.py to record only sizes and status instead of payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from config import (
    LLM_MODEL,
    LOG_DIR,
    LOG_ENABLED,
    LOG_MAX_FIELD_CHARS,
    LOG_RAW_TOOL_OUTPUT,
    LOG_SYSTEM_PROMPT_TEXT,
    PROVIDER,
)

SCHEMA_VERSION = 1

# Run outcomes, recorded verbatim in the JSON so they can be counted.
OUTCOME_ANSWERED = "answered"
OUTCOME_TOOL_ERRORS = "tool_errors_exhausted"
OUTCOME_STEP_LIMIT = "step_limit_reached"
OUTCOME_LLM_ERROR = "llm_error"
OUTCOME_APOLOGY = "apology_fallback"

_WRITE_LOCK = threading.Lock()


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _clip(value, limit: int = LOG_MAX_FIELD_CHARS):
    """Bound a string field, marking how much was dropped."""
    if not isinstance(value, str) or limit <= 0 or len(value) <= limit:
        return value
    return f"{value[:limit]}... [clipped {len(value) - limit} chars]"


def _jsonable(value):
    """Coerce anything unserializable to a string so a record never fails to write."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _fingerprint(text: str) -> dict:
    """Identify a large static string (the system prompt) without storing it."""
    text = text or ""
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16],
        "text": text if LOG_SYSTEM_PROMPT_TEXT else None,
    }


def _empty_metrics() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "sessions": 0,
        "runs": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "empty_results": 0,
        "injected_arguments": 0,
        "retries": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "run_duration_ms_total": 0,
        "outcomes": {},
        "tool_usage": {},
        "tool_failures": {},
    }


# ------------------------------------------------------------------
# One question -> answer turn
# ------------------------------------------------------------------
class RunRecorder:
    """
    Accumulates the workflow for a single turn. Steps are appended in execution
    order, so the resulting JSON reads top-to-bottom as what actually happened.
    """

    def __init__(self, logger: "WorkflowLogger", question: str, tools: list | None,
                 system_prompt: str, history_turns: int):
        self.logger = logger
        self.run_id = f"{_stamp()}-{uuid.uuid4().hex[:6]}"
        self._t0 = time.perf_counter()
        self.record = {
            "schema_version": SCHEMA_VERSION,
            "type": "run",
            "session_id": logger.session_id,
            "run_id": self.run_id,
            "started_at": _now_iso(),
            "ended_at": None,
            "duration_ms": None,
            "provider": PROVIDER,
            "model": LLM_MODEL,
            "question": question,
            "answer": None,
            "outcome": None,
            "system_prompt": _fingerprint(system_prompt),
            "history_turns_replayed": history_turns,
            "tools_available": [
                (tool.get("function") or {}).get("name") for tool in (tools or [])
            ],
            "steps": [],
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "counters": {
                "llm_calls": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "empty_results": 0,
                "injected_arguments": 0,
                "retries": 0,
            },
        }

    # -- internal ------------------------------------------------------
    def _elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def _append(self, step: dict) -> None:
        step["seq"] = len(self.record["steps"]) + 1
        step["at_ms"] = self._elapsed_ms()
        self.record["steps"].append(step)

    def _bump(self, key: str, amount: int = 1) -> None:
        self.record["counters"][key] = self.record["counters"].get(key, 0) + amount

    # -- public --------------------------------------------------------
    def record_llm_call(self, *, step: int, purpose: str, duration_ms: int,
                        usage, tool_calls: list, content: str | None,
                        message_count: int) -> None:
        """One request/response against the model."""
        try:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0

            self.record["tokens"]["prompt"] += prompt_tokens
            self.record["tokens"]["completion"] += completion_tokens
            self.record["tokens"]["total"] += total_tokens
            self._bump("llm_calls")

            self._append({
                "kind": "llm_call",
                "step": step,
                "purpose": purpose,
                "duration_ms": duration_ms,
                "messages_sent": message_count,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "requested_tools": [
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments_raw": _clip(call.function.arguments),
                    }
                    for call in tool_calls
                ],
                "content": _clip(content) if content else None,
            })
        except Exception as exc:  # pragma: no cover - logging must never raise
            print(f"[warning] workflow log (llm_call) failed: {exc}")

    def record_llm_error(self, *, step: int, duration_ms: int, error: str) -> None:
        try:
            self._bump("llm_calls")
            self._append({
                "kind": "llm_error",
                "step": step,
                "duration_ms": duration_ms,
                "error": _clip(error),
            })
        except Exception as exc:  # pragma: no cover
            print(f"[warning] workflow log (llm_error) failed: {exc}")

    def record_tool_call(self, *, step: int, tool_name: str, requested_arguments: dict,
                         result: dict, duration_ms: int, output_sent_to_model: str,
                         empty_hint_added: bool) -> None:
        """
        One MCP tool invocation. `requested_arguments` is what the model asked
        for; `injected_arguments` / `coerced_arguments` show what mcp_manager
        repaired, which is the detail you need when diagnosing MongoDB calls.
        """
        try:
            self._bump("tool_calls")
            if not result.get("success"):
                self._bump("tool_errors")
            if result.get("empty"):
                self._bump("empty_results")
            injected = result.get("injected_arguments") or []
            if injected:
                self._bump("injected_arguments", len(injected))

            entry = {
                "kind": "tool_call",
                "step": step,
                "tool": tool_name,
                "server": result.get("server"),
                "duration_ms": duration_ms,
                "success": bool(result.get("success")),
                "empty": bool(result.get("empty")),
                "injected_arguments": injected,
                "coerced_arguments": result.get("coerced_arguments") or [],
                "empty_hint_added": empty_hint_added,
                "requested_arguments": _jsonable(requested_arguments),
                "output_chars": len(result.get("text") or ""),
                "output_sent_chars": len(output_sent_to_model or ""),
            }
            if LOG_RAW_TOOL_OUTPUT:
                entry["output_raw"] = _clip(result.get("text") or "")
                entry["output_sent_to_model"] = _clip(output_sent_to_model or "")
            self._append(entry)
        except Exception as exc:  # pragma: no cover
            print(f"[warning] workflow log (tool_call) failed: {exc}")

    def record_retry(self, *, step: int, attempt: int, limit: int,
                     failures: list[str]) -> None:
        try:
            self._bump("retries")
            self._append({
                "kind": "retry",
                "step": step,
                "attempt": attempt,
                "limit": limit,
                "exhausted": attempt >= limit,
                "failures": [_clip(failure, 500) for failure in failures],
            })
        except Exception as exc:  # pragma: no cover
            print(f"[warning] workflow log (retry) failed: {exc}")

    def record_note(self, message: str, **fields) -> None:
        """Free-form marker, e.g. why the loop stopped calling tools."""
        try:
            self._append({"kind": "note", "message": message,
                          **{key: _jsonable(value) for key, value in fields.items()}})
        except Exception as exc:  # pragma: no cover
            print(f"[warning] workflow log (note) failed: {exc}")

    def finish(self, *, answer: str, outcome: str) -> dict:
        """Seal the record and flush it to all three artefacts."""
        try:
            self.record["answer"] = _clip(answer, max(LOG_MAX_FIELD_CHARS, 2000))
            self.record["outcome"] = outcome
            self.record["ended_at"] = _now_iso()
            self.record["duration_ms"] = self._elapsed_ms()
            self.logger._flush_run(self.record)
        except Exception as exc:  # pragma: no cover
            print(f"[warning] workflow log (finish) failed: {exc}")
        return self.record


# ------------------------------------------------------------------
# Session-level logger
# ------------------------------------------------------------------
class WorkflowLogger:
    """One instance per app session. Cheap to create; safe to share."""

    def __init__(self, session_id: str | None = None, enabled: bool = LOG_ENABLED):
        self.enabled = enabled
        self.session_id = session_id or f"{_stamp()}-{uuid.uuid4().hex[:6]}"
        self.session = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": _now_iso(),
            "provider": PROVIDER,
            "model": LLM_MODEL,
            "startup": None,
            "runs": [],
        }
        self._counted_session = False
        if self.enabled:
            self._ensure_dirs()

    # -- paths ---------------------------------------------------------
    @property
    def stream_path(self) -> str:
        return os.path.join(LOG_DIR, "workflow.jsonl")

    @property
    def session_path(self) -> str:
        return os.path.join(LOG_DIR, "sessions", f"{self.session_id}.json")

    @property
    def metrics_path(self) -> str:
        return os.path.join(LOG_DIR, "metrics.json")

    def _ensure_dirs(self) -> None:
        try:
            os.makedirs(os.path.join(LOG_DIR, "sessions"), exist_ok=True)
        except OSError as exc:
            print(f"[warning] could not create log directory {LOG_DIR}: {exc}")
            self.enabled = False

    # -- writers -------------------------------------------------------
    def _append_stream(self, record: dict) -> None:
        with open(self.stream_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def _write_atomic(self, path: str, payload: dict) -> None:
        """Write via a temp file + replace so a reader never sees half a file."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        os.replace(tmp, path)

    def _update_metrics(self, record: dict) -> None:
        try:
            with open(self.metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, ValueError):
            metrics = _empty_metrics()

        defaults = _empty_metrics()
        for key, value in defaults.items():
            metrics.setdefault(key, value)

        if not self._counted_session:
            metrics["sessions"] += 1
            self._counted_session = True

        counters = record.get("counters", {})
        tokens = record.get("tokens", {})

        metrics["runs"] += 1
        for key in ("llm_calls", "tool_calls", "tool_errors", "empty_results",
                    "injected_arguments", "retries"):
            metrics[key] += counters.get(key, 0)
        metrics["prompt_tokens"] += tokens.get("prompt", 0)
        metrics["completion_tokens"] += tokens.get("completion", 0)
        metrics["total_tokens"] += tokens.get("total", 0)
        metrics["run_duration_ms_total"] += record.get("duration_ms") or 0

        outcome = record.get("outcome") or "unknown"
        metrics["outcomes"][outcome] = metrics["outcomes"].get(outcome, 0) + 1

        for step in record.get("steps", []):
            if step.get("kind") != "tool_call":
                continue
            name = step.get("tool") or "unknown"
            metrics["tool_usage"][name] = metrics["tool_usage"].get(name, 0) + 1
            if not step.get("success"):
                metrics["tool_failures"][name] = metrics["tool_failures"].get(name, 0) + 1

        metrics["updated_at"] = _now_iso()
        self._write_atomic(self.metrics_path, metrics)

    def _flush_run(self, record: dict) -> None:
        if not self.enabled:
            return
        with _WRITE_LOCK:
            try:
                self.session["runs"].append(record)
                self._append_stream(record)
                self._write_atomic(self.session_path, self.session)
                self._update_metrics(record)
            except Exception as exc:  # pragma: no cover
                print(f"[warning] could not write workflow log: {exc}")

    # -- public API ----------------------------------------------------
    def start_run(self, *, question: str, tools: list | None, system_prompt: str,
                  history_turns: int) -> RunRecorder:
        return RunRecorder(self, question, tools, system_prompt, history_turns)

    def log_startup(self, *, diagnostics: dict, tools: list | None,
                    live_schema: str, schema_probes: list | None = None) -> None:
        """
        Record the connection/discovery phase — which servers answered, which
        tools registered, which were filtered out, and what the schema probe
        found. This is where a MongoDB outage or a tool-name mismatch shows up.
        """
        if not self.enabled:
            return
        try:
            record = {
                "schema_version": SCHEMA_VERSION,
                "type": "startup",
                "session_id": self.session_id,
                "at": _now_iso(),
                "provider": PROVIDER,
                "model": LLM_MODEL,
                "tools_registered": [
                    (tool.get("function") or {}).get("name") for tool in (tools or [])
                ],
                "servers": _jsonable(diagnostics),
                "schema_probes": _jsonable(schema_probes or []),
                "live_schema_chars": len(live_schema or ""),
                "live_schema": _clip(live_schema or ""),
            }
            self.session["startup"] = record
            with _WRITE_LOCK:
                self._append_stream(record)
                self._write_atomic(self.session_path, self.session)
        except Exception as exc:  # pragma: no cover
            print(f"[warning] could not write startup log: {exc}")

    def read_metrics(self) -> dict:
        """Aggregate counters across all sessions, for a monitoring panel."""
        try:
            with open(self.metrics_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return _empty_metrics()

    def recent_runs(self, limit: int = 10) -> list[dict]:
        """Most recent runs in this session, newest first."""
        return list(reversed(self.session["runs"][-limit:]))


# A default logger for callers that do not manage their own session.
_default_logger: WorkflowLogger | None = None


def get_logger() -> WorkflowLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = WorkflowLogger()
    return _default_logger
