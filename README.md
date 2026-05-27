# 🛠️ Plant Advisor — RAG Pipeline for Aluminium Ingot Manufacturing

Plant Advisor is an **industrial RAG (Retrieval-Augmented Generation) assistant** built for the AIM-5000, AIM-7500, and AIM-10000 aluminium ingot manufacturing systems. It answers operator questions about faults, safety, and procedures using only the content of the official AIM manufacturing manual, with multilingual support and multi-tier safety guardrails.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Code Details](#3-code-details)
4. [What Needs to Be Installed](#4-what-needs-to-be-installed)
5. [Ollama Integration](#5-ollama-integration)
6. [Running the Application](#6-running-the-application)
7. [Logs](#7-logs)
8. [API Endpoints](#8-api-endpoints)

---

## 1. Project Overview

Plant Advisor is a **fully local, privacy-first AI assistant** that:

- Answers questions in **English, German, and Korean** (auto-detects language, translates to English internally, responds in the user's language)
- Retrieves answers **only from the AIM manufacturing manual** — never hallucinating beyond its source
- Applies a **three-tier input guardrail** and **multi-step output guardrail** to block harmful, off-topic, or unsafe responses
- Uses **local LLMs via Ollama** — no cloud API calls, no data leaves the machine
- Provides a **Streamlit chat UI** and a **FastAPI streaming backend**
- Classifies every answer as FAULT, SAFETY, or FACT and formats accordingly

---

## 2. Project Structure

```
plant_advisor/
│
├── agents/
│   ├── language_detector.py       # Detects query language (en/de/ko)
│   ├── query_normalizer.py        # Normalises industrial terminology
│   ├── retriever_rag.py           # Retrieve → Grade → Generate nodes
│   ├── Do_NOT_translate.py        # List of protected technical terms
│   ├── translator.py              # Core translation logic (EN↔DE↔KO)
│   ├── translate_to_english.py    # LangGraph node: input translation
│   └── translate_to_user_lang.py  # LangGraph node: output translation
│
├── chroma_backup/                 # Persisted ChromaDB vector store
│
├── guard/
│   ├── runner.py                  # Input guardrail orchestrator (Tier 1/2/3)
│   ├── validators.py              # Tier 1 (scope) + Tier 2 (harmful query) validators
│   ├── output_guardrail.py        # Output guardrail pipeline
│   ├── output_validators.py       # Regex rules for unsafe output detection
│   └── llama_guard.py             # Tier 3: Llama Guard 3 (1B) via Ollama
│
├── resources/
│   ├── hybrid_retriever.py        # Dense + BM25S + RRF + BGE reranker
│   └── load_resources.py          # Loads embeddings, ChromaDB, LLM, retriever
│
├── api.py                         # FastAPI streaming backend
├── app.py                         # Streamlit chat UI
├── master_agent.py                # LangGraph workflow definition
├── state.py                       # AgentState TypedDict
├── logger.py                      # Loguru logging setup
├── run_eval.py                    # Evaluation runner
├── rag_eval_dashboard.tsx         # React evaluation dashboard
├── requirements.txt               # Python dependencies
└── AIM_Aluminium_Ingot_Manufacturing_Manual_v2.pdf
```

---

## 3. Code Details

### `master_agent.py` — LangGraph Pipeline

Defines the full agentic workflow as a **LangGraph StateGraph**. Each node is a Python async function. The pipeline flow is:

```
detect_lang
    │
    ├─(non-English)─► translate_in ─►┐
    │                                 │
    └─(English)──────────────────────►normalize_query
                                               │
                                           guardrail
                                          /         \
                                    (pass)           (fail → END)
                                       │
                                    retrieve
                                       │
                                     grade
                                    /     \
                              (relevant)   (empty → END)
                                  │
                               generate
                                  │
                             translate_out
                                  │
                                 END
```

**Nodes:**

| Node | Description |
|------|-------------|
| `detect_lang` | Detects language using `langdetect`. Sets `detected_lang`. |
| `translate_in` | Translates non-English queries to English using a local `translategemma:4b` model via Ollama. |
| `normalize_query` | Expands/normalises industrial abbreviations and synonyms (e.g. "VDC" → "Vertical Direct Chill casting"). |
| `guardrail` | Runs the 3-tier input guardrail. Blocks off-topic or harmful queries. |
| `retrieve` | Calls HybridRetriever: dense (ChromaDB) + sparse (BM25S) + RRF + BGE reranker. Returns top 5 chunks. |
| `grade` | Checks if any documents were returned. If empty, ends pipeline. |
| `generate` | Calls the LLM with retrieved context and system prompt. Classifies answer as FAULT/SAFETY/FACT and formats accordingly. |
| `translate_out` | Translates English answer back to the user's detected language. |

### `state.py` — AgentState

TypedDict carrying all data through the pipeline:

```python
class AgentState(TypedDict):
    user_query:         str         # Original user input
    detected_lang:      str         # e.g. "de", "ko", "en"
    english_query:      str         # Normalised English query
    raw_english_query:  Optional[str]
    documents:          List[str]   # Retrieved context chunks
    retrieval_metadata: List[dict]  # Rerank scores, page, chunk type, etc.
    english_answer:     str         # LLM answer in English
    final_response:     str         # Final (possibly translated) answer
    is_relevant:        bool        # Whether any docs were found
    guardrail_passed:   bool
    guardrail_message:  str
    executed_nodes:     List[str]
```

### `resources/hybrid_retriever.py` — Retrieval Pipeline

**Architecture:**

```
User Query
    │
    ├──► Dense retrieval (ChromaDB cosine, top-20)
    │
    ├──► Sparse retrieval (BM25S keyword, top-20)
    │
    └──► Reciprocal Rank Fusion (RRF, k=60)
              │
         BGE-Reranker-Base (cross-encoder, BAAI/bge-reranker-base)
              │
         Top-5 final chunks → LLM context
```

Configuration:
- `top_k=5` — final chunks passed to LLM
- `dense_k=20` — dense candidates before fusion
- `sparse_k=20` — BM25S candidates before fusion

### `guard/runner.py` — Input Guardrail

Three-tier pipeline:

| Tier | Component | Latency | Method |
|------|-----------|---------|--------|
| Tier 1 | `ScopeValidator` | ~0ms | Keyword/regex — blocks off-topic queries |
| Tier 2 | `HarmfulQueryValidator` | ~0ms | Regex rule engine — blocks harmful queries |
| Tier 3 | `LlamaGuard` | ~200ms–12s | Llama Guard 3:1b via Ollama — semantic classifier |

Tiers 1+2 run synchronously. Tier 3 runs **asynchronously in parallel** with the LangGraph pipeline (OpenAI cookbook pattern). If the guardrail fires, the pipeline task is cancelled immediately.

### `guard/output_guardrail.py` — Output Guardrail

Multi-step pipeline:

| Step | Description | Latency |
|------|-------------|---------|
| 0A | Refusal fast-path — skips all checks if LLM said "not found" | ~0ms |
| 0B | Safe structured response fast-path — skips Llama Guard for legitimate safety explanations | ~0ms |
| 1 | Keyword topic relevance check | ~0ms |
| 2 | `HarmfulResponseValidator` regex rules (e.g. missing LOTO, insufficient PPE) + Llama Guard output classification | ~200ms |

### `guard/llama_guard.py` — Llama Guard 3:1b

- Calls `llama-guard3:1b` via Ollama at `http://localhost:11434`
- Uses a binary safety prompt ("safe" / "unsafe") tailored to industrial manufacturing context
- Inference timeout: 12 seconds
- Fails **open** (passes query through) if the model is unavailable — ensuring service continuity
- Includes availability check with 60-second retry cache

### `api.py` — FastAPI Backend

- `GET /guardrail/status` — returns current guardrail toggle state
- `POST /guardrail/toggle` — enable/disable input and/or output guardrail
- `POST /chat/stream` — streaming chat endpoint (Server-Sent Events / NDJSON)
  - Emits `node`, `node_done`, `start`, `token`, `done`, and `error` event types
  - Runs input guardrail and pipeline concurrently via `asyncio`
  - Applies output guardrail before streaming tokens to the client

### `app.py` — Streamlit UI

- Chat interface with streaming token display
- Sidebar with guardrail toggle switches (input + output)
- Pipeline node progress visualization (shows which LangGraph nodes ran)
- Download interaction log as JSON
- Connects to `http://127.0.0.1:8000`

### `logger.py` — Loguru Logging

Five log sinks:

| File | Content | Rotation |
|------|---------|----------|
| `logs/app.log` | All events (DEBUG+), JSON | 10 MB / 7 days |
| `logs/errors.log` | Errors only, with full tracebacks | 5 MB / 30 days |
| `logs/retrieval.log` | Retrieval/rerank events (tagged `retrieval=True`) | 10 MB / 7 days |
| `logs/llama_guard.log` | Llama Guard prompts, responses, verdicts, latency | Managed by `llama_guard.py` |
| `logs/queries.log` | Every query audit record (BLOCKED/PASSED/ERROR) | 20 MB / 90 days |

---

## 4. What Needs to Be Installed

### System Requirements

- **Python 3.10+**
- **Ollama** (for local LLM inference) — see Section 5
- Sufficient RAM: at minimum 8 GB (16 GB recommended for all models loaded simultaneously)

### Python Dependencies

Install all Python packages with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
# LangChain stack
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-ollama
langchain-chroma
langgraph

# Vector DB
chromadb

# Embeddings + Reranker (auto-downloaded from HuggingFace on first run)
sentence-transformers
# Models used:
#   BAAI/bge-base-en-v1.5   (embeddings)
#   BAAI/bge-reranker-base  (reranker)

# Sparse retrieval
bm25s

# Guardrails
guardrails-ai>=0.4.0

# API + UI
fastapi
uvicorn
streamlit
requests

# Tokenisation / language detection
tiktoken
langdetect
```

### Optional (for tracing)

```bash
pip install langsmith
```

Set environment variables if using LangSmith:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key_here
export LANGCHAIN_PROJECT=plant-advisor
```

### HuggingFace Models (auto-downloaded on first run)

These are pulled automatically by `sentence-transformers` on first startup — no manual download needed:

- `BAAI/bge-base-en-v1.5` — dense embedding model (~440 MB)
- `BAAI/bge-reranker-base` — cross-encoder reranker (~280 MB)

---

## 5. Ollama Integration

Plant Advisor uses **three Ollama models**. All inference is local — nothing leaves the machine.

### Step 1: Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download
```

Verify the installation:

```bash
ollama --version
```

### Step 2: Start the Ollama Server

```bash
ollama serve
```

Ollama listens on `http://localhost:11434` by default. Keep this running in a separate terminal.

### Step 3: Pull the Required Models

```bash
# Main LLM — answer generation
ollama pull qwen2.5:3b

# Translation model — EN↔DE↔KO
ollama pull translategemma:4b

# Safety classifier — input + output guardrail (Tier 3)
ollama pull llama-guard3:1b
```

> **Note:** The code references `qwen2.5:3b` as the main LLM (configured in `resources/load_resources.py`). The log message says `gemma3:4b` but the active model string is `qwen2.5:3b`. Verify the model name before pulling if you have a customised configuration.

### Step 4: Verify Models Are Available

```bash
ollama list
```

You should see all three models listed. You can also verify via the API:

```bash
curl http://localhost:11434/api/tags
```

### Model Configuration

| Model | Purpose | Configured in |
|-------|---------|---------------|
| `qwen2.5:3b` | Answer generation (main RAG LLM) | `resources/load_resources.py` |
| `translategemma:4b` | Translation (EN↔DE↔KO) | `agents/translator.py` |
| `llama-guard3:1b` | Safety classification (Tier 3 guardrail) | `guard/llama_guard.py` |

To change any model, edit the `model` variable in the corresponding file.

---

## 6. Running the Application

### Step 1: Clone / Navigate to the Project

```bash
cd plant_advisor
```

### Step 2: Create and Activate a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start Ollama (in a separate terminal)

```bash
ollama serve
```

### Step 5: Start the FastAPI Backend

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`. On first startup, it will:
- Download HuggingFace embedding and reranker models (first run only)
- Load ChromaDB from `chroma_backup/`
- Build the BM25S sparse index from the ChromaDB corpus
- Connect to Ollama for LLM inference

Expect the first startup to take 30–90 seconds while models load.

### Step 6: Start the Streamlit UI (in another terminal)

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

### Quick Start Summary

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — FastAPI backend
uvicorn api:app --host 127.0.0.1 --port 8000

# Terminal 3 — Streamlit UI
streamlit run app.py
```

### Environment Variables

Create a `.env` file in the project root if needed:

```env
# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=plant-advisor
```

The project uses `python-dotenv` — `api.py` calls `load_dotenv()` on startup.

---

## 7. Logs

All logs are written to the `logs/` directory (created automatically):

```
logs/
├── app.log          # Full application log (JSON, DEBUG+)
├── errors.log       # Errors only, with full tracebacks
├── retrieval.log    # Retrieval and reranking events
├── llama_guard.log  # Llama Guard prompt, response, verdict, latency
└── queries.log      # Every query audit record (90-day retention)
```

Useful tail commands for monitoring:

```bash
# Watch Llama Guard decisions in real time
tail -f logs/llama_guard.log | python -m json.tool

# Watch all queries (blocked + passed)
tail -f logs/queries.log | python -m json.tool

# Watch retrieval events
tail -f logs/retrieval.log
```

---

## 8. API Endpoints

### `GET /guardrail/status`

Returns the current state of both guardrails.

```json
{
  "input_enabled": true,
  "output_enabled": true
}
```

### `POST /guardrail/toggle`

Enable or disable guardrails at runtime.

```json
// Disable input guardrail only
{ "type": "input", "enabled": false }

// Disable both
{ "type": "all", "enabled": false }

// Re-enable output only
{ "type": "output", "enabled": true }
```

### `POST /chat/stream`

Streaming chat endpoint. Accepts a query and returns an NDJSON stream.

**Request:**

```json
{ "query": "What PPE is required near the furnace?" }
```

**Stream event types:**

| Event type | Description |
|------------|-------------|
| `node` | A LangGraph node has started |
| `node_done` | A node has completed |
| `start` | First token is about to arrive |
| `token` | Accumulated answer text so far |
| `done` | Stream complete — includes latency and TTFT |
| `error` | An error occurred |
| `blocked` | Query or output was blocked by guardrail |

---

## Notes

- The ChromaDB collection (`chroma_backup/`) must exist and be populated before running the application. The AIM manual PDF (`AIM_Aluminium_Ingot_Manufacturing_Manual_v2.pdf`) is the source document.
- Llama Guard (Tier 3) **fails open** — if `llama-guard3:1b` is unavailable, queries pass through Tiers 1 and 2 only. This ensures the service stays available even if the safety model is not loaded.
- All LLM inference is fully local via Ollama. No data is sent to any external API.
- The BM25S sparse index is rebuilt in memory on every startup from the ChromaDB corpus. This takes a few seconds and is logged at startup.
