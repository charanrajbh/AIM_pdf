import streamlit as st

import time

import json

import sys

import os

import re

import requests

import traceback
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
 
from agents.retriever_rag import SYSTEM_PROMPT
from settings import get_settings

_SETTINGS = get_settings()

try:
    import tiktoken
    _TEXT_TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TEXT_TOKENIZER = None


def estimate_text_tokens(text):
    """Estimate tokens for an explicit text block."""
    if not text:
        return 0
    value = str(text)
    if _TEXT_TOKENIZER is not None:
        try:
            return len(_TEXT_TOKENIZER.encode(value))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE))


def local_prompt_context_tokens(metadata_list):
    """Fallback token breakdown when an older backend is still running."""
    system_tokens = estimate_text_tokens(SYSTEM_PROMPT)
    retrieved_text = "\n\n".join(
        str(item.get("retrieved_context", ""))
        for item in (metadata_list or [])
        if item.get("retrieved_context")
    )
    retrieved_tokens = estimate_text_tokens(retrieved_text)
    return system_tokens, retrieved_tokens
 
# ── Page config ───────────────────────────────────────────────
 
st.set_page_config(page_title="Plant Advisor", page_icon="🛠️", layout="wide")
 
# ── Session state ─────────────────────────────────────────────
 
if "messages" not in st.session_state:

    st.session_state.messages = []
 
if "interaction_log" not in st.session_state:

    st.session_state.interaction_log = []
 
if "full_messages" not in st.session_state:

    st.session_state.full_messages = []
 
if "input_guardrail_on" not in st.session_state:

    st.session_state.input_guardrail_on = True
 
if "output_guardrail_on" not in st.session_state:

    st.session_state.output_guardrail_on = True

if "show_response_metrics" not in st.session_state:

    st.session_state.show_response_metrics = True

# Holds a suggested question that was clicked, so it can be fed into the
# same processing path as a typed query on the next rerun.
if "pending_query" not in st.session_state:

    st.session_state.pending_query = None
 
 
 
# ── API base URL ──────────────────────────────────────────────
 
API_BASE = _SETTINGS.ui.api_base
 
# ── PDF paths ─────────────────────────────────────────────────
 
_BASE = os.path.dirname(os.path.abspath(__file__))

PDF_SEARCH_DIRS = [

    os.path.join(_BASE, "pdfs"),

    os.path.join(_BASE, "data"),

    os.path.join(_BASE, "manuals"),

    _BASE,

]
 
MIN_RERANK_FOR_PAGE = 0.05
 
# Accent colour used only for retrieved-chunk cards

_YELLOW_RGB = (255, 220, 0)

_YELLOW_HEX = "#FFDC00"
 
 
# ── Guardrail toggle helpers ──────────────────────────────────
 
def _fetch_guardrail_status():

    try:

        resp = requests.get(
            f"{API_BASE}/guardrail/status",
            timeout=_SETTINGS.ui.status_timeout_seconds,
        )

        if resp.ok:

            data = resp.json()

            return data.get("input_enabled", True), data.get("output_enabled", True)

    except Exception:

        pass

    return (

        st.session_state.input_guardrail_on,

        st.session_state.output_guardrail_on,

    )
 
 
def _set_guardrail(guard_type: str, enabled: bool) -> bool:

    try:

        resp = requests.post(

            f"{API_BASE}/guardrail/toggle",

            json={"type": guard_type, "enabled": enabled},

            timeout=_SETTINGS.ui.toggle_timeout_seconds,

        )

        return resp.ok

    except Exception:

        return False
 
@st.cache_data(show_spinner=False)

def _find_pdf_path(pdf_name: str):

    if not pdf_name:

        return None

    candidates = [pdf_name]

    if not pdf_name.lower().endswith(".pdf"):

        candidates.append(pdf_name + ".pdf")

    for directory in PDF_SEARCH_DIRS:

        for candidate in candidates:

            full = os.path.join(directory, candidate)

            if os.path.isfile(full):

                return full

    return None
 
 
# ── PDF page renderer ─────────────────────────────────────────
#
# The retrieved source page is rendered exactly as it appears in the PDF.
# No highlight annotations or text overlays are added.
 
@st.cache_data(show_spinner=False, max_entries=50)
def _render_page_with_highlights(pdf_path, page_index, chunk_texts=(), dpi=180):
    """
    Render a PDF page as a PNG without adding text highlights.

    ``chunk_texts`` is retained only for backward compatibility with existing
    callers and cache keys. It is intentionally ignored.

    Parameters
    ----------
    pdf_path    : absolute path to the PDF file
    page_index  : 0-based page index
    chunk_texts : unused; retained for compatibility
    dpi         : render resolution

    Returns
    -------
    tuple[bytes | None, str | None]
        PNG bytes and an optional error message.
    """
    del chunk_texts

    try:
        import fitz
    except ImportError:
        return None, "PyMuPDF is not installed."

    doc = None

    try:
        doc = fitz.open(pdf_path)

        if page_index < 0 or page_index >= len(doc):
            return None, (
                f"Page {page_index + 1} out of range "
                f"(PDF has {len(doc)} pages)."
            )

        page = doc[page_index]
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        return pix.tobytes("png"), None

    except Exception:
        return None, traceback.format_exc()

    finally:
        if doc is not None:
            doc.close()


def render_pdf_preview(metadata_list: list, key_prefix: str = ""):

    if not metadata_list:

        st.info("No source chunks available.")

        return
 
    ranked   = [{**ch, "rank": i} for i, ch in enumerate(metadata_list)]

    pdf_name = ranked[0].get("pdf_name", "")

    pdf_path = _find_pdf_path(pdf_name)
 
    if not pdf_path:

        st.warning(

            f"**{pdf_name}** not found.  \n"

            + "\n".join(f"- `{d}`" for d in PDF_SEARCH_DIRS)

        )

        return
 
    relevant = [

        ch for ch in ranked

        if ch.get("rerank_score") is None or ch.get("rerank_score", 0) >= MIN_RERANK_FOR_PAGE

    ]

    if not relevant:

        relevant = ranked
 
    # Build page → chunks map

    all_pages = []

    for ch in relevant:

        if ch.get("page") is not None:

            try:

                all_pages.append(int(ch["page"]))

            except (TypeError, ValueError):

                pass
 
    is_zero_based = (0 in all_pages)

    pages: dict   = {}
 
    for ch in relevant:

        raw = ch.get("page")

        if raw is None:

            continue

        try:

            meta_pg = int(raw)

        except (TypeError, ValueError):

            continue
 
        true_idx = meta_pg if is_zero_based else meta_pg - 1

        true_idx = max(0, true_idx)

        ch["true_idx"] = true_idx

        pages.setdefault(true_idx, []).append(ch)
 
    if not pages:

        st.info("No valid page numbers found in chunk metadata.")

        return
 
    sorted_indices = sorted(pages.keys())
 
    st.markdown(

        f"**{os.path.basename(pdf_path)}** &nbsp;·&nbsp; "

        f"{len(relevant)} chunk(s) &nbsp;·&nbsp; {len(sorted_indices)} page(s)",

        unsafe_allow_html=True,

    )
 
    # Chunk score legend

    n_legend = min(len(relevant), 6)

    leg_cols = st.columns(n_legend)

    for i, ch in enumerate(relevant[:n_legend]):

        sc  = ch.get("rerank_score") or 0

        pg  = ch.get("true_idx", 0) + 1

        bar = "#1D9E75" if sc >= 0.9 else "#BA7517" if sc >= 0.4 else "#D85A30"

        leg_cols[i].markdown(

            f"<div style='border:1px solid {bar};border-radius:6px;"

            f"padding:4px 6px;text-align:center'>"

            f"<b style='font-size:11px;color:{bar}'>Chunk {i+1}</b><br>"

            f"<span style='font-size:10px;color:#555'>p.{pg} · {sc:.3f}</span>"

            f"</div>",

            unsafe_allow_html=True,

        )
 
    st.markdown("")
 
    # Page selector

    page_labels = []

    for idx in sorted_indices:

        top = max((c.get("rerank_score") or 0) for c in pages[idx])

        dot = "🟢" if top >= 0.9 else "🟡" if top >= 0.4 else "🔴"

        page_labels.append(f"{dot} Page {idx + 1}")
 
    radio_key = f"{key_prefix}_pdf_radio"
 
    if len(sorted_indices) > 1:

        chosen = st.radio(

            "Jump to page",

            options=page_labels,

            horizontal=True,

            label_visibility="visible",

            key=radio_key,

        )

        selected_idx = sorted_indices[page_labels.index(chosen)]

    else:

        st.markdown(f"Showing: **{page_labels[0]}**")

        selected_idx = sorted_indices[0]
 
    page_chunks = pages[selected_idx]
 
    # Retained for compatibility; the page renderer ignores chunk text

    chunk_texts = tuple(ch.get("retrieved_context", "") for ch in page_chunks)
 
    with st.spinner(f"Rendering page {selected_idx + 1}…"):

        png, err = _render_page_with_highlights(

            pdf_path=pdf_path,

            page_index=selected_idx,

            chunk_texts=chunk_texts,

        )
 
    if err:

        st.error("Render failed:")

        st.code(err, language="")

        return
 
    if not png:

        st.error("Render returned empty image.")

        return
 
    st.image(png, use_container_width=True)
 
    # Chunk detail cards below the image

    st.markdown("---")

    st.markdown("**Retrieved chunks on this page**")

    for ch in page_chunks:

        sc    = ch.get("rerank_score") or 0

        ctype = ch.get("chunk_type", "prose")

        ctx   = ch.get("retrieved_context", "")

        rank  = ch["rank"]
 
        bar_c = "#1D9E75" if sc >= 0.9 else "#BA7517" if sc >= 0.4 else "#D85A30"

        bar_p = min(int(sc * 100), 100)

        safe  = ctx.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
 
        st.markdown(

            f"<div style='border:1px solid {_YELLOW_HEX};border-radius:8px;"

            f"padding:10px 14px;margin-bottom:8px;background:{_YELLOW_HEX}18'>"

            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"

            f"<span style='background:{_YELLOW_HEX};color:#333;border-radius:10px;"

            f"padding:2px 10px;font-size:11px;font-weight:600'>Chunk {rank+1}</span>"

            f"<span style='font-size:11px;color:#666'>{ctype}</span>"

            f"<div style='flex:1;height:5px;background:#ddd;border-radius:3px;overflow:hidden'>"

            f"<div style='width:{bar_p}%;height:100%;background:{bar_c};border-radius:3px'>"

            f"</div></div>"

            f"<span style='font-size:11px;font-weight:600;color:{bar_c}'>{sc:.4f}</span>"

            f"</div>"

            f"<pre style='margin:0;font-size:11px;line-height:1.7;"

            f"white-space:pre-wrap;font-family:monospace;color:#333'>{safe}</pre>"

            f"</div>",

            unsafe_allow_html=True,

        )
 
 
# ── Sidebar ───────────────────────────────────────────────────
 
with st.sidebar:
 
    st.header("⚙️ Settings")
 
    st.info(

        f"""

Model: {_SETTINGS.llm.model}
 
Vector DB: ChromaDB
 
Embeddings: {_SETTINGS.retriever.embedding_model}
 
Reranker: {_SETTINGS.retriever.rerank_model}
 
Sparse: BM25S

"""

    )

    st.toggle(
        "Show response metrics",
        key="show_response_metrics",
        help=(
            "Show total request time and backend-reported "
            "input/output token counts below each answer."
        ),
    )
 
    st.divider()
 
    st.markdown("### 🛡️ Guardrail Controls")

    st.caption(

        "Toggle input and output safety checks independently. "

        "Changes take effect on the **next query**."

    )
 
    if "guardrail_synced" not in st.session_state:

        inp, out = _fetch_guardrail_status()

        st.session_state.input_guardrail_on  = inp

        st.session_state.output_guardrail_on = out

        st.session_state.guardrail_synced    = True
 
    col_in_label, col_in_btn = st.columns([3, 2])

    with col_in_label:

        st.markdown("**Input Guardrail**")

        st.caption("Blocks off-topic & harmful queries before the LLM runs.")

    with col_in_btn:

        in_on    = st.session_state.input_guardrail_on

        in_label = "✅ ON" if in_on else "⛔ OFF"

        in_color = "primary" if in_on else "secondary"

        if st.button(in_label, key="btn_input_guard", use_container_width=True,

                     type=in_color, help="Click to toggle input guardrail on / off"):

            ok = _set_guardrail("input", not in_on)

            if ok:

                st.session_state.input_guardrail_on = not in_on

                st.rerun()

            else:

                st.warning("Could not reach the API server. Is it running?")
 
    st.markdown("")
 
    col_out_label, col_out_btn = st.columns([3, 2])

    with col_out_label:

        st.markdown("**Output Guardrail**")

        st.caption("Blocks hallucinated values & unsafe instructions in LLM responses.")

    with col_out_btn:

        out_on    = st.session_state.output_guardrail_on

        out_label = "✅ ON" if out_on else "⛔ OFF"

        out_color = "primary" if out_on else "secondary"

        if st.button(out_label, key="btn_output_guard", use_container_width=True,

                     type=out_color, help="Click to toggle output guardrail on / off"):

            ok = _set_guardrail("output", not out_on)

            if ok:

                st.session_state.output_guardrail_on = not out_on

                st.rerun()

            else:

                st.warning("Could not reach the API server. Is it running?")
 
    st.markdown("")

    both_on = st.session_state.input_guardrail_on and st.session_state.output_guardrail_on

    none_on = not st.session_state.input_guardrail_on and not st.session_state.output_guardrail_on

    if both_on:

        st.success("🛡️ All guardrails active")

    elif none_on:

        st.error("⚠️ All guardrails disabled — responses unfiltered")

    else:

        parts = []

        if not st.session_state.input_guardrail_on:

            parts.append("Input OFF")

        if not st.session_state.output_guardrail_on:

            parts.append("Output OFF")

        st.warning(f"⚠️ Partial protection — {', '.join(parts)}")
 
    st.divider()
 
    json_data = json.dumps(st.session_state.interaction_log, indent=4)

    st.download_button(

        label="⬇️ Download Interaction JSON",

        data=json_data,

        file_name="plant_advisor_logs.json",

        mime="application/json",

    )
 
 
# ── Main UI ───────────────────────────────────────────────────
 
st.title("🛠️ Plant Advisor")
 
 
NODE_LABELS = {

    "guardrail":          "🛡️ Input Guardrail — Semantic check",

    "llama_guard_input":  "🧠 Input Guardrail — Llama Guard",

    "output_guardrail":   "🔍 Output Guardrail — Check",

    "llama_guard_output": "🧠 Output Guardrail — Llama Guard",

    "detect_lang":        "🌐 Language Detection",

    "translate_in":       "🔤 Translate to English",

    "normalize_query":    "🔄 Query Normalization",

    "retrieve":           "🔍 Hybrid Retrieval",

    "grade":              "📊 Relevance Grading",

    "generate":           "🤖 Generating Answer",

    "translate_out":      "🌍 Translate to User Language",

}
 
 
def _build_guardrail_banner_html(message: str, guard_type: str = "input") -> str:

    """
    Returns the banner as a plain HTML string.
    Used when writing directly into an st.empty() placeholder via
    placeholder.markdown(..., unsafe_allow_html=True) — this avoids
    the raw-CSS rendering bug that occurs with .container() wrappers.
    """

    icon  = "🛡️" if guard_type == "input" else "🔒"

    title = ("Input Guardrail — Query Blocked" if guard_type == "input"

             else "Output Guardrail — Response Blocked")
 
    raw_lines = [l for l in (message or "").split("\n") if l.strip()]

    headline  = raw_lines[0] if raw_lines else title

    body_text = "\n".join(raw_lines[1:]).strip() if len(raw_lines) > 1 else ""
 
    def _esc(s):

        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
 
    headline_safe = _esc(headline)

    body_html = (

        f"<p style='font-size:13px;color:#5C3A00;margin:8px 0 0;"

        f"white-space:pre-wrap;line-height:1.6'>{_esc(body_text)}</p>"

        if body_text else ""

    )
 
    return (

        f"<div style='background:#FFF3CD;border:2px solid #E8A000;"

        f"border-left:6px solid #E8A000;border-radius:8px;"

        f"padding:16px 20px;margin:8px 0 16px 0;'>"

        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>"

        f"<span style='font-size:22px'>{icon}</span>"

        f"<strong style='font-size:15px;color:#7A4F00'>{title}</strong>"

        f"</div>"

        f"<p style='font-size:14px;color:#5C3A00;margin:0;font-weight:600'>{headline_safe}</p>"

        f"{body_html}"

        f"</div>"

    )
 
 
def _render_guardrail_blocked_banner(message: str, guard_type: str = "input"):

    """Used for history replay via normal st.markdown (not inside a placeholder)."""

    st.markdown(

        _build_guardrail_banner_html(message, guard_type),

        unsafe_allow_html=True,

    )
 
 
def render_response_metrics(metrics: dict):

    """Render timing and the explicit token breakdown below an answer."""

    if not st.session_state.get("show_response_metrics", True):
        return

    if not metrics:
        return

    total_time = metrics.get("total_time_seconds")
    output_available = metrics.get(
        "output_token_count_available",
        metrics.get("token_counts_available", False),
    )

    def _seconds(value):
        try:
            return f"{float(value):.3f} s"
        except (TypeError, ValueError):
            return "N/A"

    def _tokens(value, available=True):
        if not available:
            return "N/A"
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return "N/A"

    st.caption("Response metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total time", _seconds(total_time))
    metric_cols[1].metric(
        "Input tokens (question only)",
        _tokens(metrics.get("question_tokens", metrics.get("input_tokens"))),
    )
    metric_cols[2].metric(
        "System + retrieved tokens",
        _tokens(metrics.get("system_and_retrieved_tokens")),
    )
    metric_cols[3].metric(
        "Output tokens",
        _tokens(metrics.get("output_tokens"), output_available),
    )

    system_tokens = metrics.get("system_prompt_tokens")
    retrieved_tokens = metrics.get("retrieved_content_tokens")
    if system_tokens is not None and retrieved_tokens is not None:
        st.caption(
            f"System prompt: {_tokens(system_tokens)} · "
            f"Retrieved content: {_tokens(retrieved_tokens)}"
        )

    st.caption(
        "Question and system/retrieved counts are text-token estimates; "
        "output tokens use model-reported usage when available."
    )


def stream_answer(user_input, response_placeholder):

    """
    Streams the answer token-by-token into response_placeholder.

    Token flow
    ----------
    api.py yields {"type":"token","content": <NEW_CHUNK>} for every LLM token
    regardless of output-guard state.  We append each chunk to `answer` and
    update the placeholder with the full accumulated string so Markdown renders
    correctly throughout (a trailing cursor ▌ is appended while streaming).

    If the output guardrail fires after generation, api.py sends
    {"type":"output_guardrail_blocked"} then {"type":"done"}.
    We replace the streamed text with the block banner at that point.
    """

    response = requests.get(

        f"{API_BASE}/stream",

        params={"query": user_input},

        stream=True,

        timeout=_SETTINGS.ui.stream_timeout_seconds,

    )
 
    answer            = ""

    metadata_list     = []

    first_token_time  = None

    start             = time.time()

    step_placeholder  = st.empty()

    output_blocked    = False

    input_blocked     = False

    block_message     = ""
    response_metrics  = {}
 
    def show_step(node, done=False, blocked=False):

        label = NODE_LABELS.get(node, node)

        if blocked:

            step_placeholder.error(f"🚫 **{label}** — Blocked")

        elif done:

            step_placeholder.success(f"✅ **{label}**")

        else:

            step_placeholder.info(f"⏳ **{label}** — Running...")
 
    try:

        for line in response.iter_lines():

            if not line:

                continue

            decoded = line.decode("utf-8").strip()

            if decoded.startswith('{"detail"}'):

                continue

            try:

                data = json.loads(decoded)

            except Exception:

                continue

            if not isinstance(data, dict) or "type" not in data:

                continue
 
            if data["type"] == "node":

                show_step(data["node"], done=False)
 
            elif data["type"] == "node_done":

                # Clear node badge immediately — no sleep, avoids delaying first token

                step_placeholder.empty()
 
            elif data["type"] == "guardrail_blocked":

                show_step("guardrail", blocked=True)

                msg = data.get("message", "Query blocked by input guardrail.")

                response_placeholder.markdown(

                    _build_guardrail_banner_html(msg, guard_type="input"),

                    unsafe_allow_html=True,

                )

                answer = msg

                input_blocked = True
                
 
            elif data["type"] == "output_guardrail_blocked":

                show_step("output_guardrail", blocked=True)

                output_blocked = True

                block_message  = data.get("message", "Response blocked by output guardrail.")
 
            elif data["type"] == "start":

                # LLM generation starting — clear progress badge

                step_placeholder.empty()

                if first_token_time is None:

                    first_token_time = time.time()
 
            elif data["type"] == "token":

                # Each event carries only the NEW chunk for this token.

                # Append and re-render the full markdown with a live cursor.

                chunk = data.get("content", "")

                if chunk:

                    answer += chunk

                    response_placeholder.markdown(answer + " ▌")
 
            elif data["type"] == "done":

                end = time.time()

                step_placeholder.empty()
 
                if output_blocked:

                    # Replace the streamed text with the guardrail block banner

                    response_placeholder.markdown(

                        _build_guardrail_banner_html(block_message, guard_type="output"),

                        unsafe_allow_html=True,

                    )

                    answer = block_message

                elif input_blocked:

                    pass  # banner already rendered during guardrail_blocked event

                else:

                    # Final clean render — removes the ▌ cursor

                    final = data.get("final_response", answer)

                    if final:

                        answer = final

                    response_placeholder.markdown(answer)
 
                client_latency = end - start
                client_ttft = (
                    first_token_time - start
                    if first_token_time else client_latency
                )

                server_metrics = data.get("metrics") or {}

                try:
                    latency = float(
                        server_metrics.get(
                            "total_time_seconds",
                            data.get("latency", client_latency),
                        )
                    )
                except (TypeError, ValueError):
                    latency = client_latency

                try:
                    ttft = float(
                        server_metrics.get(
                            "ttft_seconds",
                            data.get("ttft", client_ttft),
                        )
                    )
                except (TypeError, ValueError):
                    ttft = client_ttft

                metadata_list = data.get("metadata", [])
                local_system_tokens, local_retrieved_tokens = (
                    local_prompt_context_tokens(metadata_list)
                )

                question_tokens = server_metrics.get("question_tokens")
                if question_tokens is None:
                    question_tokens = estimate_text_tokens(user_input)

                system_prompt_tokens = server_metrics.get("system_prompt_tokens")
                if system_prompt_tokens is None:
                    system_prompt_tokens = local_system_tokens

                retrieved_content_tokens = server_metrics.get(
                    "retrieved_content_tokens"
                )
                if retrieved_content_tokens is None:
                    retrieved_content_tokens = local_retrieved_tokens

                system_and_retrieved_tokens = server_metrics.get(
                    "system_and_retrieved_tokens"
                )
                if system_and_retrieved_tokens is None:
                    system_and_retrieved_tokens = (
                        system_prompt_tokens + retrieved_content_tokens
                    )

                output_tokens = server_metrics.get(
                    "output_tokens", data.get("completion_tokens")
                )
                if not output_tokens and answer:
                    output_tokens = estimate_text_tokens(answer)

                output_token_count_available = bool(output_tokens)
                if not output_token_count_available:
                    output_token_count_available = server_metrics.get(
                        "output_token_count_available",
                        False,
                    )

                response_metrics = {
                    "ttft_seconds": ttft,
                    "model_ttft_seconds": server_metrics.get("model_ttft_seconds"),
                    "total_time_seconds": latency,
                    "input_tokens": question_tokens,
                    "question_tokens": question_tokens,
                    "system_prompt_tokens": system_prompt_tokens,
                    "retrieved_content_tokens": retrieved_content_tokens,
                    "system_and_retrieved_tokens": system_and_retrieved_tokens,
                    "output_tokens": output_tokens,
                    "model_input_tokens": server_metrics.get("model_input_tokens"),
                    "total_tokens": server_metrics.get("total_tokens"),
                    "llm_calls": server_metrics.get("llm_calls", 0),
                    "token_counts_available": server_metrics.get(
                        "token_counts_available", True
                    ),
                    "output_token_count_available": output_token_count_available,
                    "token_scope": server_metrics.get("token_scope"),
                    "token_count_method": server_metrics.get(
                        "token_count_method",
                        "cl100k_base text estimate with model-reported output usage",
                    ),
                }

                return (
                    answer, latency, ttft, response_metrics, metadata_list,
                    output_blocked, input_blocked,
                )
 
            elif data["type"] == "error":

                step_placeholder.empty()

                response_placeholder.error(data.get("message", "Unknown error"))

                return answer, 0, 0, {}, [], False, False
 
    except requests.exceptions.ChunkedEncodingError:

        step_placeholder.empty()

        response_placeholder.error("⚠️ Stream interrupted.")

        return answer, 0, 0, {}, [], False, False
 
    step_placeholder.empty()

    return answer, 0, 0, {}, [], False, False
 
 
def render_assistant_message(msg: dict, msg_idx: int):

    """Render a stored assistant message from chat history (history replay only)."""

    output_blocked = msg.get("output_blocked", False)

    input_blocked  = msg.get("input_blocked",  False)
 
    if input_blocked:

        _render_guardrail_blocked_banner(msg["content"], guard_type="input")

    elif output_blocked:

        _render_guardrail_blocked_banner(msg["content"], guard_type="output")

    else:

        st.markdown(msg["content"])

    render_response_metrics(msg.get("metrics", {}))
 
    # PDF expander — always starts closed, user opens manually

    metadata = msg.get("metadata", [])

    any_blocked = output_blocked or input_blocked

    if metadata and not any_blocked:

        with st.expander("📄 View source PDF", expanded=False):

            render_pdf_preview(metadata, key_prefix=f"msg_{msg_idx}")
  
 
# ── Replay chat history ───────────────────────────────────────
 
for idx, msg in enumerate(st.session_state.full_messages):

    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant":

            render_assistant_message(msg, idx)

        else:

            st.markdown(msg["content"])
 
 
# ── Suggested maintenance questions ───────────────────────────
# Pre-populated, in-scope prompts shown on an empty chat. Clicking one
# sets pending_query, which is picked up below and fed through the exact
# same stream_answer() / /stream pipeline as a typed question.

SUGGESTED_QUESTIONS = [
    "What is the lockout/tagout (LOTO) procedure before furnace or casting pit maintenance?",
    "What are the daily per-shift maintenance tasks for the casting pit?",
    "How often should the ILD graphite rotor be replaced?",
    "When should the casting table hydraulic oil be changed?",
    "How often are the furnace thermocouples replaced?",
    "What PPE is required for refractory maintenance work?",
]


def render_suggested_questions():
    """Render clickable maintenance suggestion chips."""
    st.markdown("##### 🛠️ Suggested maintenance questions")
    st.caption("Tap a question to send it straight to the assistant.")
    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_query = question
                st.rerun()


# Show suggestions only when the conversation is empty (welcome state).
if not st.session_state.full_messages:

    render_suggested_questions()


# ── Chat input ────────────────────────────────────────────────
# A clicked suggestion (pending_query) is treated exactly like a typed
# query. chat_input must still be called every run so the input box renders.

_typed_input = st.chat_input("Enter issue...")
_clicked_input = st.session_state.pending_query
st.session_state.pending_query = None
user_input = _typed_input or _clicked_input

if user_input:
 
    st.session_state.full_messages.append({"role": "user", "content": user_input})
 
    with st.chat_message("user"):

        st.markdown(user_input)
 
    with st.chat_message("assistant"):

        response_placeholder = st.empty()
 
        # stream_answer renders the answer live into response_placeholder

        (
            answer, latency, ttft, response_metrics, metadata_list,
            output_blocked, input_blocked,
        ) = stream_answer(

            user_input, response_placeholder

        )

        render_response_metrics(response_metrics)
 
        new_msg = {

            "role":           "assistant",

            "content":        answer,

            "metadata":       metadata_list,

            "output_blocked": output_blocked,

            "input_blocked":  input_blocked,
            "metrics":        response_metrics,

        }
 
    st.session_state.full_messages.append(new_msg)

    st.session_state.interaction_log.append({

        "question":       user_input,

        "answer":         answer,

        "output_blocked": output_blocked,

        "latency":        round(latency, 3),

        "ttft":           round(ttft, 3),
        "metrics":        response_metrics,

        "metadata":       metadata_list,

    })
  
    st.rerun()
