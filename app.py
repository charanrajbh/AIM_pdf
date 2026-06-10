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

 
 
# ── Token estimation ──────────────────────────────────────────
 
import tiktoken
 
tokenizer = tiktoken.get_encoding("cl100k_base")
 
def estimate_tokens(text):
    if not text:
        return 0
    return len(tokenizer.encode(text))
 
 
# ── API base URL ──────────────────────────────────────────────
 
API_BASE = "http://127.0.0.1:8000"
 
 
# ── Guardrail toggle helpers ──────────────────────────────────
 
def _fetch_guardrail_status():
    try:
        resp = requests.get(f"{API_BASE}/guardrail/status", timeout=2)
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
            timeout=3,
        )
        return resp.ok
    except Exception:
        return False
    
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
 
 
# ── Accurate PDF highlighter ──────────────────────────────────
#
# Strategy (most-precise → least-precise, stops as soon as hits are found):
#
#   1. Try the full cleaned sentence/line as-is.
#   2. Split into overlapping 8-word windows — try each window phrase.
#   3. Split into overlapping 5-word windows.
#   4. Split into overlapping 3-word windows.
#
# No single-keyword fallback — avoids lighting up unrelated words that
# merely share a common token with the chunk text.
#
# All hits are highlighted in yellow only.
 
@st.cache_data(show_spinner=False, max_entries=50)
def _render_page_with_highlights(pdf_path, page_index, chunk_texts, dpi=180):
    """
    Parameters
    ----------
    pdf_path    : absolute path to the PDF file
    page_index  : 0-based page index
    chunk_texts : tuple of raw retrieved_context strings for this page
    dpi         : render resolution
 
    Returns (png_bytes, error_string_or_None)
    """
    try:
        import fitz
    except ImportError:
        return None, "PyMuPDF is not installed."
 
    try:
        doc = fitz.open(pdf_path)
 
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            return None, f"Page {page_index + 1} out of range (PDF has {len(doc)} pages)."
 
        page = doc[page_index]
 
        # Yellow colour for fitz (0–1 scale)
        yellow = (_YELLOW_RGB[0] / 255.0,
                  _YELLOW_RGB[1] / 255.0,
                  _YELLOW_RGB[2] / 255.0)
 
        def _highlight_rects(rects):
            for rect in rects:
                try:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=yellow)
                    annot.set_opacity(0.55)
                    annot.update()
                except Exception:
                    pass
 
        def _sliding_windows(words, size):
            """Yield overlapping phrases of `size` words."""
            for i in range(len(words) - size + 1):
                yield " ".join(words[i: i + size])
 
        def _clean_line(raw: str) -> str:
            """Strip markdown/table artefacts, collapse whitespace."""
            text = re.sub(r'[\|\*\#\_`]', ' ', raw)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
 
        for chunk_text in chunk_texts:
            if not chunk_text or not chunk_text.strip():
                continue
 
            # Split chunk into individual lines, skip very short ones
            raw_lines = [l for l in chunk_text.split('\n') if len(l.strip()) > 6]
 
            for raw_line in raw_lines:
                line = _clean_line(raw_line)
                if not line:
                    continue
 
                words = line.split()
 
                # ── Pass 1: full line ────────────────────────────────────
                rects = page.search_for(line)
                if rects:
                    _highlight_rects(rects)
                    continue
 
                # ── Pass 2: 8-word windows ───────────────────────────────
                if len(words) >= 8:
                    found = False
                    for phrase in _sliding_windows(words, 8):
                        rects = page.search_for(phrase)
                        if rects:
                            _highlight_rects(rects)
                            found = True
                    if found:
                        continue
 
                # ── Pass 3: 5-word windows ───────────────────────────────
                if len(words) >= 5:
                    found = False
                    for phrase in _sliding_windows(words, 5):
                        rects = page.search_for(phrase)
                        if rects:
                            _highlight_rects(rects)
                            found = True
                    if found:
                        continue
 
                # ── Pass 4: 3-word windows (last resort) ─────────────────
                # Only run if the line is a meaningful length (avoids
                # highlighting generic 3-word combos from short lines)
                if len(words) >= 3 and len(line) > 20:
                    for phrase in _sliding_windows(words, 3):
                        rects = page.search_for(phrase)
                        if rects:
                            _highlight_rects(rects)
 
        zoom   = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)
        png    = pix.tobytes("png")
        doc.close()
        return png, None
 
    except Exception:
        return None, traceback.format_exc()
 
 
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
        f"{len(relevant)} chunk(s) &nbsp;·&nbsp; {len(sorted_indices)} page(s) &nbsp;·&nbsp; "
        f"<span style='background:{_YELLOW_HEX};padding:1px 6px;border-radius:3px;"
        f"font-size:11px;font-weight:600'>highlighted in yellow</span>",
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
 
    # Pass only the text strings — all yellow, no per-chunk colour
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
        """
Model: gemma3:4b
 
Vector DB: ChromaDB
 
Embeddings: BAAI/bge-base-en-v1.5
 
Reranker: BAAI/bge-reranker-base
 
Sparse: BM25S
"""
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
 
    "guardrail":              "🛡️ Input Guardrail — Semantic check",
 
    "llama_guard_input":      "🧠 Input Guardrail — Llama Guard",
 
    "output_guardrail":       "🔍 Output Guardrail — Regex check",
 
    "llama_guard_output":     "🧠 Output Guardrail — Llama Guard",
 
    "detect_lang":            "🌐 Language Detection",
 
    "translate_in":           "🔤 Translate to English",
 
    "normalize_query":        "🔄 Query Normalization",
 
 
    "retrieve":               "🔍 Hybrid Retrieval",
 
    "grade":                  "📊 Relevance Grading",
 
    "generate":               "🤖 Generating Answer",
 
    "translate_out":          "🌍 Translate to User Language",
 
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
        timeout=120,
    )

    answer            = ""
    metadata_list     = []
    first_token_time  = None
    start             = time.time()
    step_placeholder  = st.empty()
    output_blocked    = False
    input_blocked     = False
    block_message     = ""

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
                break

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

                latency       = end - start
                ttft          = (first_token_time - start) if first_token_time else latency
                metadata_list = data.get("metadata", [])
                return answer, latency, ttft, metadata_list, output_blocked, input_blocked

            elif data["type"] == "error":
                step_placeholder.empty()
                response_placeholder.error(data.get("message", "Unknown error"))
                return answer, 0, 0, [], False, False

    except requests.exceptions.ChunkedEncodingError:
        step_placeholder.empty()
        response_placeholder.error("⚠️ Stream interrupted.")
        return answer, 0, 0, [], False, False

    step_placeholder.empty()
    return answer, 0, 0, [], False, False

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

    # PDF expander — always starts closed, user opens manually
    metadata = msg.get("metadata", [])
    any_blocked = output_blocked or input_blocked
    if metadata and not any_blocked:
        with st.expander("📄 View source PDF with highlights", expanded=False):
            render_pdf_preview(metadata, key_prefix=f"msg_{msg_idx}")
 
 
# ── Render PDF expander for the LIVE message ──────────────────
# Called once after stream_answer() returns.
# Metrics have been moved to the terminal (logger.py waterfall).

def _render_live_message_footer(msg: dict, msg_idx: int):
    """Render PDF expander for the just-generated message (metrics removed — see terminal)."""
    output_blocked = msg.get("output_blocked", False)
    input_blocked  = msg.get("input_blocked",  False)
    metadata       = msg.get("metadata", [])

    any_blocked = output_blocked or input_blocked
    if metadata and not any_blocked:
        with st.expander("📄 View source PDF with highlights", expanded=False):
            render_pdf_preview(metadata, key_prefix=f"msg_{msg_idx}")
 
 
# ── Replay chat history ───────────────────────────────────────
 
for idx, msg in enumerate(st.session_state.full_messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg, idx)
        else:
            st.markdown(msg["content"])
 
 
# ── Chat input ────────────────────────────────────────────────
 
if user_input := st.chat_input("Enter issue..."):
 
    st.session_state.full_messages.append({"role": "user", "content": user_input})
 
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
 
        # stream_answer renders the answer live into response_placeholder
        answer, latency, ttft, metadata_list, output_blocked, input_blocked = stream_answer(
            user_input, response_placeholder
        )
 
        new_msg = {
            "role":           "assistant",
            "content":        answer,
            "metadata":       metadata_list,
            "output_blocked": output_blocked,
            "input_blocked":  input_blocked,
        }

        # Render PDF expander only (metrics shown in terminal)
        _render_live_message_footer(new_msg, len(st.session_state.full_messages))

    st.session_state.full_messages.append(new_msg)
    st.session_state.interaction_log.append({
        "question":       user_input,
        "answer":         answer,
        "output_blocked": output_blocked,
        "latency":        round(latency, 3),
        "ttft":           round(ttft, 3),
        "metadata":       metadata_list,
    })
 
 
