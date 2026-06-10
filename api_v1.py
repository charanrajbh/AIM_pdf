"""

api.py  — Plant Advisor

=======================

FastAPI streaming endpoint.
 
Query audit logging:

  - BLOCKED_INPUT  → written by runner.py validate_query()

  - BLOCKED_OUTPUT → written here after output guardrail fires

  - PASSED         → written here after full pipeline completes

  - ERROR          → written here on exception
 
All records go to logs/queries.log (query_audit=True tag).

"""
 
import os

from dotenv import load_dotenv
 
load_dotenv()
 
from fastapi import FastAPI, Request, Body

from fastapi.responses import StreamingResponse, JSONResponse

import json

import time

import uuid

import datetime
 
from master_agent import agent_app, llm as _pipeline_llm

from guard.runner import validate_query, validate_query_regex_only, validate_query_llama_tier

from guard.output_guardrail import validate_output, validate_output_regex_only

from logger import logger, print_request_waterfall

# ── Pre-guardrail translation helpers ────────────────────────────────────────
# We run detect_lang + translate_in BEFORE the guardrail so the semantic
# checkers always operate on English text regardless of the user's language.

from agents.language_detector import detect_language as _detect_language
from agents.translate_to_english import translate_to_english as _translate_to_english


async def _detect_and_translate(query: str) -> tuple[str, str]:
    """
    Run language detection + translation outside of LangGraph.
    Returns (detected_lang, english_query).
    The english_query is fed into the guardrail AND pre-loaded into the
    LangGraph state so detect_lang / translate_in nodes are skipped.
    """
    # Build a minimal state dict matching AgentState
    state: dict = {"user_query": query, "executed_nodes": []}

    # Step 1 — detect language
    lang_result = _detect_language(state)
    state.update(lang_result)

    detected_lang = state.get("detected_lang", "English")

    # Unsupported language — return raw query; pipeline will handle the
    # final_response message set by detect_language.
    if detected_lang == "unsupported":
        return detected_lang, query

    # Step 2 — translate if not English
    if detected_lang.lower() not in ["en", "english"]:
        trans_result = await _translate_to_english(state, _pipeline_llm)
        english_query = trans_result.get("english_query", query)
    else:
        english_query = query

    return detected_lang, english_query
 
# ── LangSmith tracing setup ─────────────────────────────────────────────────────
# Uses RunTree for a single root run per request with child runs for every step.
# This produces ONE clean trace with a full waterfall instead of multiple
# disconnected "OllamaChat" runs that LangChain auto-creates.
# LANGCHAIN_TRACING_V2 must remain True in .env for LangGraph internal traces,
# but we use metadata["root_run_id"] to link them to our parent span.

try:
    from langsmith import Client as LangSmithClient, RunTree
    _ls_client    = LangSmithClient()
    _ls_project   = os.environ.get("LANGCHAIN_PROJECT", "plant-advisor")
    _ls_available = True
    logger.info("LangSmith client initialised | project={}", _ls_project)
except Exception:
    _ls_client    = None
    _ls_project   = None
    _ls_available = False
    RunTree        = None
    logger.warning("LangSmith not available — traces will not appear in dashboard")
 
app = FastAPI()
 
 
# ─────────────────────────────────────────────────────────────────────────────

# Global guardrail toggle state

# ─────────────────────────────────────────────────────────────────────────────
 
_guardrail_state = {

    "input_enabled":  True,

    "output_enabled": True,

}
 
 
# ─────────────────────────────────────────────────────────────────────────────

# Guardrail management endpoints

# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/guardrail/status")

async def guardrail_status():

    return JSONResponse(content=_guardrail_state)
 
 
@app.post("/guardrail/toggle")

async def guardrail_toggle(payload: dict = Body(...)):

    guard_type = payload.get("type", "all")

    enabled    = payload.get("enabled", True)
 
    if guard_type in ("input", "all"):

        _guardrail_state["input_enabled"] = bool(enabled)

        logger.info(

            "Input guardrail toggled | enabled={}",

            _guardrail_state["input_enabled"],

        )
 
    if guard_type in ("output", "all"):

        _guardrail_state["output_enabled"] = bool(enabled)

        logger.info(

            "Output guardrail toggled | enabled={}",

            _guardrail_state["output_enabled"],

        )
 
    return JSONResponse(content={"status": "updated", "state": _guardrail_state})
 
 
# ─────────────────────────────────────────────────────────────────────────────

# LangSmith helpers

# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_semantic_rule(rejection_msg: str) -> tuple[str, str, str]:

    """

    Extract tier, category, and decision from the rejection message.
 
    runner.py embeds the rule in the block_rule field passed to _audit()

    as "SemanticScopeChecker:BLOCK_OUT_OF_SCOPE" or

    "SemanticHarmfulChecker:BYPASS_BMS".
 
    The rejection_msg itself is the user-facing safety text, so we

    infer tier from its content instead.
 
    Returns (tier, category, decision) — all strings for LangSmith tags.

    """

    msg = (rejection_msg or "").lower()
 
    # Tier 1 messages always contain this phrase

    if "only answer questions about the aim" in msg:

        if "unrelated topic" in msg:

            return "Tier1_Semantic", "SCOPE", "BLOCK_OUT_OF_SCOPE"

        else:

            return "Tier1_Semantic", "SCOPE", "BLOCK_TOO_GENERIC"
 
    # Tier 3 Llama Guard messages

    if "llama" in msg or "s1" in msg or "s2" in msg:

        return "Tier3_LlamaGuard", "LLAMA_GUARD", "BLOCK_HARMFUL"
 
    # Tier 2 — map known safety warning keywords to category names

    # These appear in the rejection messages defined in semantic_tier2_harmful_v3.py

    category_map = [

        ("steam explosion",           "WET_CHARGE"),

        ("moisture check",            "WET_CHARGE"),

        ("damp",                      "WET_CHARGE"),

        ("bms safety interlock",      "BYPASS_BMS"),

        ("burner management",         "BYPASS_BMS"),

        ("gas solenoid",              "FORCE_GAS_VALVE"),

        ("unburnt gas",               "FORCE_GAS_VALVE"),

        ("welding",                   "WELD_VESSEL"),

        ("drilling",                  "WELD_VESSEL"),

        ("furnace bay atmosphere",    "BREATHE_ATMOSPHERE"),

        ("respiratory",               "BREATHE_ATMOSPHERE"),

        ("cl2",                       "USE_CL2_NO_EXTRACT"),

        ("chlorine gas",              "USE_CL2_NO_EXTRACT"),

        ("fume extraction",           "USE_CL2_NO_EXTRACT"),

        ("shift metallurgist",        "CAST_OUT_OF_SPEC"),

        ("out-of-spec",               "CAST_OUT_OF_SPEC"),

        ("class f",                   "CHARGE_CLASS_F"),

        ("unknown alloy",             "CHARGE_CLASS_F"),

        ("magnesium",                 "MG_AFTER_FLUX"),

        ("salt flux",                 "MG_AFTER_FLUX"),

        ("casting pit",               "PIT_ENTRY_NO_PERMIT"),

        ("permit to work",            "PIT_ENTRY_NO_PERMIT"),

        ("loto",                      "MAINTENANCE_NO_LOTO"),

        ("lockout",                   "MAINTENANCE_NO_LOTO"),

        ("sabotage",                  "SABOTAGE"),

        ("emergency stop",            "SABOTAGE"),

        ("falsify",                   "FRAUD_FALSIFICATION"),

        ("quality records",           "FRAUD_FALSIFICATION"),

        ("process limit",             "VIOLATE_PROCESS_LIMIT"),

        ("process limits",            "VIOLATE_PROCESS_LIMIT"),

    ]

    for keyword, category in category_map:

        if keyword in msg:

            return "Tier2_Semantic", category, "BLOCK_HARMFUL"
 
    return "Tier2_Semantic", "UNKNOWN", "BLOCK_HARMFUL"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LangSmith RunTree tracing
# ONE root run per request with child spans for every pipeline step.
# Produces a single clean waterfall trace instead of disconnected OllamaChat runs.
# ─────────────────────────────────────────────────────────────────────────────

def _make_root_run(
    run_id:     str,
    query:      str,
    tags:       list[str],
) -> "RunTree | None":
    """
    Create and post the root RunTree span for this request.
    Returns the RunTree object (caller must call .end() when done)
    or None if LangSmith is unavailable.
    All child spans are attached via rt.create_child().
    """
    if not _ls_available or RunTree is None:
        return None
    try:
        rt = RunTree(
            name=f"plant-advisor | {run_id}",
            run_type="chain",
            project_name=_ls_project,
            inputs={"query": query},
            tags=tags,
        )
        rt.post()
        return rt
    except Exception as e:
        logger.warning("LangSmith root run failed | run_id={} error={}", run_id, str(e))
        return None


def _child(
    root:       "RunTree | None",
    name:       str,
    run_type:   str,
    inputs:     dict,
    tags:       list[str] | None = None,
) -> "RunTree | None":
    """
    Create and post a child span under root.
    Returns the child RunTree so caller can .end() it.
    """
    if root is None:
        return None
    try:
        child = root.create_child(
            name=name,
            run_type=run_type,
            inputs=inputs,
            tags=tags or [],
        )
        child.post()
        return child
    except Exception as e:
        logger.warning("LangSmith child span failed | name={} error={}", name, str(e))
        return None


def _end_child(
    child:   "RunTree | None",
    outputs: dict,
    error:   str | None = None,
) -> None:
    """End and patch a child span with outputs (and optional error)."""
    if child is None:
        return
    try:
        child.end(outputs=outputs, error=error)
        child.patch()
    except Exception as e:
        logger.warning("LangSmith end_child failed | error={}", str(e))


def _end_root(
    root:    "RunTree | None",
    outputs: dict,
    tags:    list[str] | None = None,
    error:   str | None = None,
) -> None:
    """End and patch the root span."""
    if root is None:
        return
    try:
        if tags:
            root.tags = list(set((root.tags or []) + tags))
        root.end(outputs=outputs, error=error)
        root.patch()
    except Exception as e:
        logger.warning("LangSmith end_root failed | error={}", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compat wrappers used by blocked/passed paths
# ─────────────────────────────────────────────────────────────────────────────

def _trace_blocked_query(
    run_id:        str,
    query:         str,
    stage:         str,
    block_rule:    str,
    block_reason:  str,
    latency_ms:    int,
    english_query: str | None = None,
    llm_response:  str | None = None,
    tier:          str | None = None,
    category:      str | None = None,
    decision:      str | None = None,
    score_in:      float | None = None,
    score_out:     float | None = None,
    harm_score:    float | None = None,
    matched_proto: str | None = None,
    # Root RunTree — when provided the trace is attached as a child;
    # when None a standalone flat run is created (backwards compat).
    root:          "RunTree | None" = None,
) -> None:
    if not _ls_available or _ls_client is None:
        return
    try:
        if tier is None:
            tier, category, decision = _parse_semantic_rule(block_reason)

        tags = ["blocked", f"{stage}_guardrail", block_rule]
        if tier:          tags.append(tier)
        if category and category != "UNKNOWN": tags.append(category)
        if decision:      tags.append(decision)
        if harm_score is not None:
            if   harm_score >= 0.90: tags.append("score_very_high")
            elif harm_score >= 0.75: tags.append("score_high")
            elif harm_score >= 0.70: tags.append("score_threshold")

        outputs = {
            "guardrail_passed": False,
            "block_stage":      stage,
            "block_rule":       block_rule,
            "block_reason":     block_reason,
            "latency_ms":       latency_ms,
            "tier":             tier,
            "category":         category,
            "decision":         decision,
            "score_in":         score_in,
            "score_out":        score_out,
            "harm_score":       harm_score,
            "matched_proto":    matched_proto,
        }
        if llm_response:
            outputs["llm_response"] = llm_response[:500]

        inputs = {
            "query":           query,
            "english_query":   english_query or query,
            "guardrail_stage": stage,
            "tier":            tier,
            "category":        category,
        }

        error_str = (
            f"[{stage.upper()} GUARDRAIL BLOCKED] "
            f"{tier or block_rule} | {category} | {decision} | "
            f"{block_reason[:200]}"
        )

        if root is not None:
            # Attach as child span so it appears in the waterfall
            child = _child(root, f"guardrail_blocked:{stage}", "chain", inputs, tags)
            _end_child(child, outputs, error=error_str)
        else:
            # Standalone run (backwards compat for output-guardrail blocks
            # where root may not be available)
            now = datetime.datetime.utcnow()
            _ls_client.create_run(
                name=f"plant-advisor-{run_id}",
                run_type="chain",
                project_name=_ls_project,
                inputs=inputs,
                outputs=outputs,
                error=error_str,
                tags=tags,
                start_time=now,
                end_time=now,
            )

        logger.debug(
            "LangSmith blocked trace | run_id={} tier={} category={} score_in={} harm_score={}",
            run_id, tier, category,
            f"{score_in:.3f}" if score_in is not None else "â",
            f"{harm_score:.3f}" if harm_score is not None else "â",
        )
    except Exception as e:
        logger.warning("LangSmith trace failed | run_id={} error={}", run_id, str(e))


def _trace_passed_query(
    run_id:        str,
    query:         str,
    latency_ms:    int,
    english_query: str | None = None,
    score_in:      float | None = None,
    score_out:     float | None = None,
    harm_score:    float | None = None,
    root:          "RunTree | None" = None,
) -> None:
    if not _ls_available or _ls_client is None:
        return
    try:
        tags = ["passed", "input_guardrail"]
        if score_in is not None and score_in < 0.50:  tags.append("t1_near_miss")
        if harm_score is not None and harm_score >= 0.55: tags.append("t2_near_miss")

        outputs = {
            "guardrail_passed": True,
            "latency_ms":       latency_ms,
            "score_in":         score_in,
            "score_out":        score_out,
            "harm_score":       harm_score,
        }
        inputs = {"query": query, "english_query": english_query or query}

        if root is not None:
            child = _child(root, "guardrail_passed", "chain", inputs, tags)
            _end_child(child, outputs)
        else:
            now = datetime.datetime.utcnow()
            _ls_client.create_run(
                name=f"plant-advisor-{run_id}",
                run_type="chain",
                project_name=_ls_project,
                inputs=inputs,
                outputs=outputs,
                tags=tags,
                start_time=now,
                end_time=now,
            )
    except Exception as e:
        logger.warning("LangSmith passed-query trace failed | run_id={} error={}", run_id, str(e))


# ─────────────────────────────────────────────────────────────────────────────

# Audit writer — for OUTPUT guardrail results and PASSED / ERROR outcomes

# Input guardrail audit is written inside runner.py validate_query()

# ─────────────────────────────────────────────────────────────────────────────
 
def _write_audit(

    run_id:           str,

    query:            str,

    status:           str,

    block_stage:      str  | None,

    block_rule:       str  | None,

    block_reason:     str  | None,

    detected_lang:    str  | None,

    chunks_retrieved: int,

    latency_ms:       int,

    input_guard_on:   bool,

    output_guard_on:  bool,

    llm_response:     str  | None = None,

) -> None:

    logger.bind(query_audit=True).info(

        "QUERY_AUDIT",

        run_id=run_id,

        query=query,

        status=status,

        block_stage=block_stage,

        block_rule=block_rule,

        block_reason=block_reason,

        llm_response=llm_response,

        detected_lang=detected_lang,

        chunks_retrieved=chunks_retrieved,

        latency_ms=latency_ms,

        input_guard_on=input_guard_on,

        output_guard_on=output_guard_on,

    )
 
 
# ─────────────────────────────────────────────────────────────────────────────

# Main streaming endpoint

# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/stream")

async def stream(query: str, request: Request):
 
    run_id           = str(uuid.uuid4())[:8]

    run_input_guard  = _guardrail_state["input_enabled"]

    run_output_guard = _guardrail_state["output_enabled"]
 
    async def generator():

        start            = time.time()
        first_token_time = None
        final_answer     = ""
        current_state    = {}
        root_rt          = None   # LangSmith root RunTree for this request

        log = logger.bind(run_id=run_id, query=query[:120])
        log.info(
            "Request started | input_guard={} output_guard={}",
            run_input_guard, run_output_guard,
        )

        # ── Create root LangSmith RunTree span ───────────────────────────────
        # ONE root run per request. Every step below is a child of this span.
        # This eliminates the stray "OllamaChat" runs LangChain emits separately.
        root_tags = [
            "plant-advisor",
            f"lang:{detected_lang_hint if False else 'pending'}",
            "input_guard:on" if run_input_guard  else "input_guard:off",
            "output_guard:on" if run_output_guard else "output_guard:off",
        ]
        root_rt = _make_root_run(run_id=run_id, query=query, tags=["plant-advisor"])

        try:

            # ── 0. DETECT LANGUAGE + TRANSLATE ───────────────────────────────
            t_detect_start = time.time()
            yield json.dumps({"type": "node", "node": "detect_lang"}) + "\n"

            child_detect = _child(
                root_rt, "detect_lang", "chain",
                {"query": query},
                tags=["detect_lang"],
            )

            detected_lang, english_query = await _detect_and_translate(query)
            t_detect_ms = round((time.time() - t_detect_start) * 1000)

            yield json.dumps({"type": "node_done", "node": "detect_lang"}) + "\n"

            _end_child(child_detect, {
                "detected_lang": detected_lang,
                "english_query": english_query,
                "latency_ms":    t_detect_ms,
            })

            # Translation child span (only when non-English)
            if detected_lang.lower() not in ["en", "english", "unsupported"]:
                yield json.dumps({"type": "node", "node": "translate_in"}) + "\n"
                child_trans = _child(
                    root_rt, "translate_in", "chain",
                    {"query": query, "detected_lang": detected_lang},
                    tags=["translate_in"],
                )
                _end_child(child_trans, {
                    "english_query": english_query,
                    "latency_ms":    t_detect_ms,
                })
                yield json.dumps({"type": "node_done", "node": "translate_in"}) + "\n"

            log.info(
                "Pre-guardrail translation | detected_lang={} english_query={}",
                detected_lang, english_query[:120],
            )

            # Update root run input with resolved language info
            if root_rt is not None:
                try:
                    root_rt.inputs.update({
                        "detected_lang":  detected_lang,
                        "english_query":  english_query,
                    })
                except Exception:
                    pass

            # Unsupported language — exit early
            if detected_lang == "unsupported":
                unsupported_msg = (
                    "This language is not supported yet. "
                    "Supported languages: English, German, Korean."
                )
                yield json.dumps({"type": "start"}) + "\n"
                yield json.dumps({"type": "token", "content": unsupported_msg}) + "\n"
                yield json.dumps({
                    "type": "done",
                    "latency": round(time.time() - start, 3),
                    "ttft":    round(time.time() - start, 3),
                    "final_response": unsupported_msg,
                    "metadata": [],
                }) + "\n"
                _end_root(root_rt, {
                    "final_response":  unsupported_msg,
                    "outcome":         "UNSUPPORTED_LANGUAGE",
                    "latency_ms":      round((time.time() - start) * 1000),
                }, tags=["unsupported_language"])
                return

            # ── 1. INPUT GUARDRAIL ────────────────────────────────────────────
            t_guard_start  = time.time()
            _t1_score_in   = None
            _t1_score_out  = None
            _t1_latency_ms = None
            _t2_harm_score = None
            _t2_matched    = None
            _t2_category   = None
            _t2_latency_ms = None
            _t3_latency_ms = None
            _t2_safe_pass  = False

            yield json.dumps({"type": "node", "node": "guardrail"}) + "\n"

            child_guard = _child(
                root_rt, "input_guardrail", "chain",
                {"english_query": english_query, "query": query},
                tags=["guardrail", "input"],
            )

            if run_input_guard:

                # ── Tier 1 ──────────────────────────────────────────────────
                t1_start = time.time()
                child_t1 = _child(
                    child_guard, "tier1_scope_check", "chain",
                    {"english_query": english_query},
                    tags=["tier1", "semantic"],
                )
                try:
                    from guard.semantic_tier1_scope_v3   import SemanticScopeChecker
                    from guard.semantic_tier2_harmful_v3 import SemanticHarmfulChecker
                    from guard.runner import _scope_checker, _harmful_checker

                    t1 = _scope_checker.check(english_query)
                    _t1_score_in   = t1.score_in_scope
                    _t1_score_out  = t1.score_out_scope
                    _t1_latency_ms = round((time.time() - t1_start) * 1000)

                    _end_child(child_t1, {
                        "decision":      t1.decision,
                        "score_in":      round(t1.score_in_scope, 3),
                        "score_out":     round(t1.score_out_scope, 3),
                        "margin":        round(t1.margin, 3),
                        "matched_in":    t1.matched_in,
                        "matched_out":   t1.matched_out,
                        "latency_ms":    _t1_latency_ms,
                        "passed":        t1.passed,
                    }, error=None if t1.passed else f"BLOCKED:{t1.decision}")

                    if not t1.passed:
                        passed        = False
                        rejection_msg = t1.rejection_msg
                        tier, category, decision = _parse_semantic_rule(rejection_msg)
                        log.warning(
                            "Tier1 blocked | run_id={} decision={} score_in={:.3f} score_out={:.3f}",
                            run_id, t1.decision, t1.score_in_scope, t1.score_out_scope,
                        )
                    else:
                        # ── Tier 2 ──────────────────────────────────────────
                        t2_start = time.time()
                        child_t2 = _child(
                            child_guard, "tier2_harmful_check", "chain",
                            {"english_query": english_query},
                            tags=["tier2", "semantic"],
                        )
                        t2 = _harmful_checker.check(english_query)
                        _t2_harm_score = t2.score
                        _t2_matched    = t2.matched_proto
                        _t2_category   = t2.category
                        _t2_latency_ms = round((time.time() - t2_start) * 1000)
                        _t2_safe_pass  = getattr(t2, "safe_passage", False)

                        _end_child(child_t2, {
                            "decision":      t2.decision,
                            "score":         round(t2.score, 3),
                            "raw_score":     round(t2.raw_score, 3),
                            "category":      t2.category,
                            "matched_proto": t2.matched_proto,
                            "safe_passage":  _t2_safe_pass,
                            "latency_ms":    _t2_latency_ms,
                            "passed":        t2.passed,
                        }, error=None if t2.passed else f"BLOCKED:{t2.category}")

                        if not t2.passed:
                            passed        = False
                            rejection_msg = t2.rejection_msg
                            tier          = "Tier2_Semantic"
                            category      = t2.category or "UNKNOWN"
                            decision      = "BLOCK_HARMFUL"
                            log.warning(
                                "Tier2 blocked | run_id={} category={} score={:.3f} proto={}",
                                run_id, t2.category, t2.score, (t2.matched_proto or "")[:60],
                            )
                        else:
                            passed = True; rejection_msg = None
                            tier = category = decision = None

                except Exception as e:
                    logger.error(
                        "Direct checker access failed | run_id={} error={} â falling back",
                        run_id, str(e),
                    )
                    passed, rejection_msg = validate_query_regex_only(
                        english_query, run_id=run_id,
                        input_guard_on=run_input_guard,
                        output_guard_on=run_output_guard,
                    )
                    tier = category = decision = None

                # ── Tier 3 LlamaGuard ────────────────────────────────────────
                if passed and not _t2_safe_pass:
                    from guard.llama_guard import _check_model_available
                    if _check_model_available():
                        t3_start  = time.time()
                        child_t3  = _child(
                            child_guard, "tier3_llama_guard", "llm",
                            {"english_query": english_query, "safe_passage_skipped": False},
                            tags=["tier3", "llama_guard"],
                        )
                        yield json.dumps({"type": "node", "node": "llama_guard_input"}) + "\n"

                        passed, rejection_msg = validate_query_llama_tier(
                            english_query, run_id=run_id,
                            input_guard_on=run_input_guard,
                            output_guard_on=run_output_guard,
                        )
                        _t3_latency_ms = round((time.time() - t3_start) * 1000)

                        yield json.dumps({"type": "node_done", "node": "llama_guard_input"}) + "\n"

                        _end_child(child_t3, {
                            "passed":     passed,
                            "latency_ms": _t3_latency_ms,
                            "model":      "llama-guard3:1b",
                        }, error=None if passed else "BLOCKED:LlamaGuard")

                        if not passed:
                            tier = "Tier3_LlamaGuard"; category = "LLAMA_GUARD"; decision = "BLOCK_HARMFUL"

                elif passed and _t2_safe_pass:
                    # Tier 3 skipped — log this as a child span so it's visible
                    child_t3_skip = _child(
                        child_guard, "tier3_llama_guard_SKIPPED", "chain",
                        {"reason": "safe_passage=True from Tier2"},
                        tags=["tier3", "skipped"],
                    )
                    _end_child(child_t3_skip, {"passed": True, "skipped": True, "reason": "safe_passage"})

            else:
                passed = True; rejection_msg = None
                tier = category = decision = None
                log.info("Input guardrail SKIPPED (disabled by toggle)")

            t_guard_ms = round((time.time() - t_guard_start) * 1000)

            # ── End guardrail child span ──────────────────────────────────────
            if not passed:
                if tier == "Tier1_Semantic":
                    input_block_rule = f"SemanticScopeChecker:{decision or 'BLOCK'}"
                elif tier == "Tier2_Semantic":
                    input_block_rule = f"SemanticHarmfulChecker:{category or 'UNKNOWN'}"
                elif tier == "Tier3_LlamaGuard":
                    input_block_rule = "LlamaGuard:BLOCK_HARMFUL"
                else:
                    input_block_rule = "Guardrail:UNKNOWN"

                _end_child(child_guard, {
                    "passed":      False,
                    "tier":        tier,
                    "category":    category,
                    "decision":    decision,
                    "t1_score_in": _t1_score_in,
                    "t1_score_out":_t1_score_out,
                    "t1_latency_ms": _t1_latency_ms,
                    "t2_harm_score": _t2_harm_score,
                    "t2_latency_ms": _t2_latency_ms,
                    "t3_latency_ms": _t3_latency_ms,
                    "total_latency_ms": t_guard_ms,
                    "block_rule":  input_block_rule,
                    "block_reason": (rejection_msg or "")[:300],
                }, error=f"BLOCKED:{tier}:{category}")

                log.warning(
                    "Input guardrail blocked | run_id={} tier={} category={} "
                    "score_in={} score_out={} harm_score={} reason={}",
                    run_id, tier or "unknown", category or "â",
                    f"{_t1_score_in:.3f}"   if _t1_score_in   is not None else "â",
                    f"{_t1_score_out:.3f}"  if _t1_score_out  is not None else "â",
                    f"{_t2_harm_score:.3f}" if _t2_harm_score is not None else "â",
                    (rejection_msg or "")[:200],
                )

                latency_input_blocked = round(time.time() - start, 3)
                _trace_blocked_query(
                    run_id=run_id, query=query, english_query=english_query,
                    stage="input", block_rule=input_block_rule,
                    block_reason=rejection_msg or "",
                    latency_ms=round(latency_input_blocked * 1000),
                    tier=tier, category=category, decision=decision,
                    score_in=_t1_score_in, score_out=_t1_score_out,
                    harm_score=_t2_harm_score, matched_proto=_t2_matched,
                    root=root_rt,
                )

                _end_root(root_rt, {
                    "outcome":         "BLOCKED_INPUT",
                    "tier":            tier,
                    "category":        category,
                    "latency_ms":      round(latency_input_blocked * 1000),
                    "block_reason":    (rejection_msg or "")[:300],
                }, tags=["blocked", "input_guardrail", tier or ""])

                # ── Terminal waterfall for blocked input ─────────────────
                print_request_waterfall(
                    run_id=run_id,
                    query=query,
                    detected_lang=detected_lang or "unknown",
                    input_guard_on=run_input_guard,
                    output_guard_on=run_output_guard,
                    guard_passed=False,
                    t1_score_in=_t1_score_in,
                    t1_score_out=_t1_score_out,
                    t1_latency_ms=_t1_latency_ms,
                    t2_harm_score=_t2_harm_score,
                    t2_safe_pass=_t2_safe_pass,
                    t2_latency_ms=_t2_latency_ms,
                    t3_latency_ms=_t3_latency_ms,
                    block_tier=tier,
                    block_category=category,
                    block_reason=rejection_msg,
                    node_timings={},
                    chunks_retrieved=0,
                    chunk_scores=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    ttft_ms=round(latency_input_blocked * 1000),
                    total_latency_ms=round(latency_input_blocked * 1000),
                    outcome="BLOCKED_INPUT",
                )
                yield json.dumps({"type": "guardrail_blocked", "node": "guardrail", "message": rejection_msg}) + "\n"
                yield json.dumps({
                    "type": "done",
                    "latency": latency_input_blocked,
                    "ttft":    latency_input_blocked,
                    "final_response": rejection_msg,
                    "metadata": [],
                }) + "\n"
                return

            # Guardrail passed
            _end_child(child_guard, {
                "passed":          True,
                "t1_score_in":     _t1_score_in,
                "t1_score_out":    _t1_score_out,
                "t1_latency_ms":   _t1_latency_ms,
                "t2_harm_score":   _t2_harm_score,
                "t2_safe_passage": _t2_safe_pass,
                "t2_latency_ms":   _t2_latency_ms,
                "t3_latency_ms":   _t3_latency_ms,
                "total_latency_ms": t_guard_ms,
            })

            log.info("Input guardrail passed")
            yield json.dumps({"type": "node_done", "node": "guardrail"}) + "\n"

            _trace_passed_query(
                run_id=run_id, query=query, english_query=english_query,
                latency_ms=round((time.time() - start) * 1000),
                score_in=_t1_score_in, score_out=_t1_score_out,
                harm_score=_t2_harm_score, root=root_rt,
            )

            # ── 2. LANGGRAPH PIPELINE ─────────────────────────────────────────
            valid_nodes = [
                "detect_lang", "translate_in", "normalize_query",
                "retrieve", "grade", "generate", "translate_out",
            ]

            # Per-node timing for waterfall
            _node_start_times: dict[str, float] = {}
            _node_timings:     dict[str, int]   = {}
            _node_children:    dict[str, "RunTree | None"] = {}

            # Pipeline-level child span
            child_pipeline = _child(
                root_rt, "langgraph_pipeline", "chain",
                {"english_query": english_query, "detected_lang": detected_lang},
                tags=["pipeline"],
            )

            state_input = {
                "user_query":     query,
                "detected_lang":  detected_lang,
                "english_query":  english_query,
                "executed_nodes": ["detect_lang"] + (
                    ["translate_in"] if detected_lang.lower() not in ["en", "english"] else []
                ),
            }

            try:
                async for event in agent_app.astream_events(
                    state_input,
                    version="v2",
                    config={
                        "run_name":   f"plant-advisor-{run_id}",
                        "tags":       [f"run_id:{run_id}"],
                        "metadata":   {
                            "run_id":      run_id,
                            # Linking metadata — LangSmith uses this to attach
                            # LangGraph's internal spans as children of our root.
                            "root_run_id": str(root_rt.id) if root_rt else "",
                        },
                    },
                ):
                    kind = event["event"]
                    name = event["name"]
                    tags = event.get("tags", [])

                    if kind == "on_chain_start" and name in valid_nodes:
                        log.debug("Node started | node={}", name)
                        _node_start_times[name] = time.time()
                        node_inputs = event["data"].get("input") or {}
                        _node_children[name] = _child(
                            child_pipeline, name, "chain",
                            inputs={"node": name, "input_keys": list(node_inputs.keys()) if isinstance(node_inputs, dict) else []},
                            tags=[name, "pipeline_node"],
                        )
                        yield json.dumps({"type": "node", "node": name}) + "\n"

                    if kind == "on_chain_end" and name in valid_nodes:
                        output = event["data"].get("output")
                        if isinstance(output, dict):
                            current_state.update(output)

                        node_ms  = round((time.time() - _node_start_times.get(name, time.time())) * 1000)
                        _node_timings[name] = node_ms
                        node_out = {k: str(v)[:200] for k, v in (output or {}).items() if v is not None} if isinstance(output, dict) else {}
                        node_out["latency_ms"] = node_ms

                        _end_child(_node_children.get(name), node_out)
                        log.debug("Node done | node={} latency_ms={}", name, node_ms)
                        yield json.dumps({"type": "node_done", "node": name}) + "\n"

                    # Token streaming — always stream to UI regardless of output guard.
                    # The output guard runs after full generation and sends a
                    # "replace" event to overwrite the already-streamed text if blocked.
                    if kind == "on_chat_model_stream" and "final_node" in tags:
                        chunk = event["data"].get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            if first_token_time is None:
                                first_token_time = time.time()
                                ttft_val = round(first_token_time - start, 3)
                                log.debug("First token | ttft={}s", ttft_val)
                                yield json.dumps({"type": "start"}) + "\n"
                            final_answer += chunk.content
                            yield json.dumps({"type": "token", "content": chunk.content}) + "\n"

                    # LLM usage metrics — capture token counts + speed from LLM end events
                    if kind == "on_chat_model_end":
                        llm_output = event["data"].get("output")
                        if llm_output and hasattr(llm_output, "generations"):
                            try:
                                gen = llm_output.generations[0][0]
                                usage = getattr(gen.message, "usage_metadata", None) or {}
                                prompt_tokens     = usage.get("input_tokens",  0)
                                completion_tokens = usage.get("output_tokens", 0)
                                total_tokens      = usage.get("total_tokens",  prompt_tokens + completion_tokens)
                                llm_elapsed       = round((time.time() - start) * 1000)
                                tokens_per_sec    = round(completion_tokens / max(llm_elapsed / 1000, 0.001), 1) if completion_tokens else None

                                child_llm = _child(
                                    child_pipeline, "llm_call", "llm",
                                    inputs={"model": event.get("name", "ollama")},
                                    tags=["llm", "generate"],
                                )
                                _end_child(child_llm, {
                                    "prompt_tokens":     prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens":      total_tokens,
                                    "tokens_per_sec":    tokens_per_sec,
                                    "latency_ms":        llm_elapsed,
                                })
                            except Exception:
                                pass

                pipeline_ms = round((time.time() - start) * 1000)
                _end_child(child_pipeline, {
                    "final_response_preview": final_answer[:200],
                    "chunks_retrieved":       len(current_state.get("retrieval_metadata", [])),
                    "latency_ms":             pipeline_ms,
                })

            except Exception as e:
                _end_child(child_pipeline, {}, error=str(e))
                raise

            # ── 3. OUTPUT GUARDRAIL ───────────────────────────────────────────
            latency = round(time.time() - start, 3)
            ttft    = round((first_token_time - start), 3) if first_token_time else latency

            state_final = current_state.get("final_response", "")
            if state_final:
                final_answer = state_final

            if not final_answer:
                final_answer = (
                    "\u2139\ufe0f No relevant information was found in the manual "
                    "for your query. Please rephrase or ask about a specific "
                    "AIM system component, fault, or procedure."
                )
                yield json.dumps({"type": "start"}) + "\n"
                yield json.dumps({"type": "token", "content": final_answer}) + "\n"

            metadata       = current_state.get("retrieval_metadata", [])
            context_chunks = [m.get("retrieved_context", "") for m in metadata]
            detected_lang  = current_state.get("detected_lang", detected_lang)

            if run_output_guard and final_answer:
                t_outguard_start = time.time()
                yield json.dumps({"type": "node", "node": "output_guardrail"}) + "\n"

                child_outguard = _child(
                    root_rt, "output_guardrail", "chain",
                    {"response_preview": final_answer[:200], "query": query},
                    tags=["guardrail", "output"],
                )

                # Regex check child
                # Regex/keyword check removed — LlamaGuard is the sole output check.
                # Fast-paths (refusal + safe structured response) are handled inside
                # classify_output() before LlamaGuard is called.
                yield json.dumps({"type": "node_done", "node": "output_guardrail"}) + "\n"
                out_passed = True
                out_msg    = None

                if out_passed:
                    from guard.llama_guard import _check_model_available, classify_output
                    if _check_model_available():
                        t_lg_out_start = time.time()
                        yield json.dumps({"type": "node", "node": "llama_guard_output"}) + "\n"

                        child_lg_out = _child(
                            child_outguard, "llama_guard_output", "llm",
                            {"response_preview": final_answer[:200], "query": query},
                            tags=["llama_guard", "output"],
                        )

                        lg_safe, lg_msg, lg_cat = classify_output(response=final_answer, query=query)
                        t_lg_out_ms = round((time.time() - t_lg_out_start) * 1000)

                        _end_child(child_lg_out, {
                            "passed":     lg_safe,
                            "latency_ms": t_lg_out_ms,
                            "model":      "llama-guard3:1b",
                        }, error=None if lg_safe else "BLOCKED:LlamaGuard")

                        yield json.dumps({"type": "node_done", "node": "llama_guard_output"}) + "\n"
                        if not lg_safe:
                            out_passed = False
                            out_msg    = lg_msg

                t_outguard_ms = round((time.time() - t_outguard_start) * 1000)

                if not out_passed:
                    log.warning("Output guardrail blocked | run_id={} reason={}", run_id, (out_msg or "")[:200])

                    _end_child(child_outguard, {
                        "passed":     False,
                        "latency_ms": t_outguard_ms,
                        "block_msg":  (out_msg or "")[:300],
                    }, error="BLOCKED_OUTPUT")

                    _write_audit(
                        run_id=run_id, query=query, status="BLOCKED_OUTPUT",
                        block_stage="output", block_rule="HarmfulResponseValidator",
                        block_reason=out_msg, detected_lang=detected_lang,
                        chunks_retrieved=len(metadata), latency_ms=round(latency * 1000),
                        input_guard_on=run_input_guard, output_guard_on=run_output_guard,
                        llm_response=final_answer,
                    )
                    _trace_blocked_query(
                        run_id=run_id, query=query, stage="output",
                        block_rule="HarmfulResponseValidator", block_reason=out_msg or "",
                        latency_ms=round(latency * 1000), llm_response=final_answer,
                        tier="OutputGuardrail", category="HARMFUL_RESPONSE",
                        decision="BLOCK_OUTPUT", root=root_rt,
                    )
                    _end_root(root_rt, {
                        "outcome":        "BLOCKED_OUTPUT",
                        "latency_ms":     round(latency * 1000),
                        "ttft_ms":        round(ttft * 1000),
                        "block_reason":   (out_msg or "")[:300],
                    }, tags=["blocked", "output_guardrail"])

                    # ── Terminal waterfall for blocked output ─────────────
                    _wf_chunks_meta2 = current_state.get("retrieval_metadata", [])
                    _wf_pt2  = current_state.get("prompt_tokens",     0) or 0
                    _wf_ct2  = current_state.get("completion_tokens", 0) or 0
                    print_request_waterfall(
                        run_id=run_id,
                        query=query,
                        detected_lang=detected_lang or "unknown",
                        input_guard_on=run_input_guard,
                        output_guard_on=run_output_guard,
                        guard_passed=False,
                        t1_score_in=_t1_score_in,
                        t1_score_out=_t1_score_out,
                        t1_latency_ms=_t1_latency_ms,
                        t2_harm_score=_t2_harm_score,
                        t2_safe_pass=_t2_safe_pass,
                        t2_latency_ms=_t2_latency_ms,
                        t3_latency_ms=_t3_latency_ms,
                        block_tier="OutputGuardrail",
                        block_category="HARMFUL_RESPONSE",
                        block_reason=out_msg,
                        node_timings=_node_timings,
                        chunks_retrieved=len(_wf_chunks_meta2),
                        chunk_scores=[round(m.get("rerank_score", 0) or 0, 3) for m in _wf_chunks_meta2[:5]],
                        prompt_tokens=_wf_pt2,
                        completion_tokens=_wf_ct2,
                        ttft_ms=round(ttft * 1000),
                        total_latency_ms=round(latency * 1000),
                        outcome="BLOCKED_OUTPUT",
                    )
                    yield json.dumps({"type": "output_guardrail_blocked", "node": "output_guardrail", "message": out_msg}) + "\n"
                    yield json.dumps({
                        "type": "done", "latency": latency, "ttft": ttft,
                        "final_response": out_msg, "metadata": metadata, "output_blocked": True,
                    }) + "\n"
                    return

                _end_child(child_outguard, {"passed": True, "latency_ms": t_outguard_ms})
                log.info("Output guardrail passed")
                # Tokens were already streamed live — nothing more to send here.

            # ── Done ─────────────────────────────────────────────────────────
            latency = round(time.time() - start, 3)
            ttft    = round((first_token_time - start), 3) if first_token_time else latency

            log.info(
                "Request done | latency={}s ttft={}s chunks={} lang={}",
                latency, ttft, len(metadata), detected_lang or "unknown",
            )

            # ── Terminal waterfall — printed after every completed request ────
            _wf_chunks_meta = current_state.get("retrieval_metadata", [])
            _wf_pt  = current_state.get("prompt_tokens",     0) or 0
            _wf_ct  = current_state.get("completion_tokens", 0) or 0
            print_request_waterfall(
                run_id=run_id,
                query=query,
                detected_lang=detected_lang or "unknown",
                input_guard_on=run_input_guard,
                output_guard_on=run_output_guard,
                guard_passed=True,
                t1_score_in=_t1_score_in,
                t1_score_out=_t1_score_out,
                t1_latency_ms=_t1_latency_ms,
                t2_harm_score=_t2_harm_score,
                t2_safe_pass=_t2_safe_pass,
                t2_latency_ms=_t2_latency_ms,
                t3_latency_ms=_t3_latency_ms,
                block_tier=None,
                block_category=None,
                block_reason=None,
                node_timings=_node_timings,
                chunks_retrieved=len(_wf_chunks_meta),
                chunk_scores=[round(m.get("rerank_score", 0) or 0, 3) for m in _wf_chunks_meta[:5]],
                prompt_tokens=_wf_pt,
                completion_tokens=_wf_ct,
                ttft_ms=round(ttft * 1000),
                total_latency_ms=round(latency * 1000),
                outcome="PASSED",
            )

            _write_audit(
                run_id=run_id, query=query, status="PASSED",
                block_stage=None, block_rule=None, block_reason=None,
                detected_lang=detected_lang, chunks_retrieved=len(metadata),
                latency_ms=round(latency * 1000),
                input_guard_on=run_input_guard, output_guard_on=run_output_guard,
            )

            # ── Finalise root run with full metrics ───────────────────────────
            chunks_meta = current_state.get("retrieval_metadata", [])
            prompt_tokens     = current_state.get("prompt_tokens",     0)
            completion_tokens = current_state.get("completion_tokens", 0)
            tokens_per_sec    = round(
                completion_tokens / max(ttft, 0.001), 1
            ) if completion_tokens else None

            _end_root(root_rt, {
                "outcome":              "PASSED",
                "final_response":       final_answer[:500],
                "detected_lang":        detected_lang,
                "english_query":        english_query,
                # ── Timing waterfall ──────────────────────────────────────
                "total_latency_ms":     round(latency * 1000),
                "ttft_ms":              round(ttft * 1000),
                "tokens_per_sec":       tokens_per_sec,
                # ── Guardrail scores ──────────────────────────────────────
                "t1_score_in":          _t1_score_in,
                "t1_score_out":         _t1_score_out,
                "t1_latency_ms":        _t1_latency_ms,
                "t2_harm_score":        _t2_harm_score,
                "t2_safe_passage":      _t2_safe_pass,
                "t2_latency_ms":        _t2_latency_ms,
                "t3_latency_ms":        _t3_latency_ms,
                # ── Retrieval ────────────────────────────────────────────
                "chunks_retrieved":     len(chunks_meta),
                "chunk_scores":         [round(m.get("score", 0), 3) for m in chunks_meta[:5]],
                "chunk_sources":        [m.get("source", "") for m in chunks_meta[:5]],
                # ── LLM metrics ──────────────────────────────────────────
                "prompt_tokens":        prompt_tokens,
                "completion_tokens":    completion_tokens,
            }, tags=["passed"])

            yield json.dumps({
                "type": "done", "latency": latency, "ttft": ttft,
                "final_response": final_answer, "metadata": metadata, "output_blocked": False,
            }) + "\n"

        except Exception as e:
            log.exception("Stream crashed | error={}", str(e))
            _end_root(root_rt, {}, error=str(e), tags=["error"])
            _write_audit(
                run_id=run_id, query=query, status="ERROR",
                block_stage=None, block_rule="Exception", block_reason=str(e)[:300],
                detected_lang=current_state.get("detected_lang"),
                chunks_retrieved=len(current_state.get("retrieval_metadata", [])),
                latency_ms=round((time.time() - start) * 1000),
                input_guard_on=run_input_guard, output_guard_on=run_output_guard,
            )
            yield json.dumps({"type": "error", "message": f"Stream crashed: {str(e)}"}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")
 
