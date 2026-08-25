"""
Central configuration for the school data assistant.
Adjust server URLs / model name here — nothing else needs to change.
"""

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# ------------------------------------------------------------------
# LLM provider switch. Options: "ollama", "gemini", "groq", or "bedrock"
#   "ollama"   -> local model via Ollama's OpenAI-compatible endpoint
#   "gemini"   -> Google Gemini via its OpenAI-compatible endpoint
#   "groq"     -> Groq Cloud via its OpenAI-compatible endpoint
#   "bedrock"  -> Amazon Bedrock (via LiteLLM / Bedrock OpenAI proxy)
# ------------------------------------------------------------------
PROVIDER = "bedrock"  # "ollama", "gemini", "groq", or "bedrock"

_PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # required by the OpenAI SDK, but unused by Ollama
        "model": "qwen3:4b-instruct",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "model": "gemini-3.5-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "model": "llama-3.3-70b-versatile",  # Or "llama-3.1-8b-instant" / "mixtral-8x7b-32768"
    },
    "bedrock": {
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key": "sk-dummy-key",
        "model": "us.anthropic.claude-opus-4-8",
    },
}

LLM_BASE_URL = _PROVIDERS[PROVIDER]["base_url"]
LLM_API_KEY = _PROVIDERS[PROVIDER]["api_key"]
LLM_MODEL = _PROVIDERS[PROVIDER]["model"]

# ------------------------------------------------------------------
# Agent loop limits.
# ------------------------------------------------------------------
# How many *failing* tool-call rounds the agent tolerates before it stops
# calling tools and answers with whatever it has. This must be >= 2, otherwise
# the model never gets a chance to read an error message and correct itself.
MAX_QUERY_RETRIES = 3

# Hard ceiling on LLM round-trips per question. Guarantees the agent loop
# always terminates, even if the model keeps requesting tools successfully.
MAX_AGENT_STEPS = 8

# How many previous user/assistant exchanges to replay as conversation context.
# The system prompt is always kept in addition to these — it is never trimmed.
MAX_HISTORY_TURNS = 3

# Budget for tool output handed back to the model. MongoDB documents are far
# larger than MySQL rows, so this needs headroom; anything trimmed is reported
# to the model with an explicit truncation marker.
MAX_TOOL_OUTPUT_CHARS = 6000
MAX_TOOL_OUTPUT_ITEMS = 25

# Wall-clock limits (seconds) so a hung server can never freeze the UI.
TOOL_CALL_TIMEOUT = 60
SERVER_CONNECT_TIMEOUT = 20

# How much live schema text may be injected per data source.
MAX_SCHEMA_BLOCK_CHARS = 4000

# ------------------------------------------------------------------
# Workflow monitoring / logging (see workflow_logger.py).
#
# WARNING: these logs contain the questions asked, the queries issued and the
# rows/documents returned — i.e. real student data. Keep LOG_DIR out of version
# control, and set LOG_RAW_TOOL_OUTPUT = False to record only sizes and status.
# ------------------------------------------------------------------
LOG_ENABLED = os.environ.get("WORKFLOW_LOG_ENABLED", "true").lower() != "false"
LOG_DIR = os.environ.get(
    "WORKFLOW_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
)
# Per-field cap for logged payloads; anything longer is clipped with a marker.
LOG_MAX_FIELD_CHARS = 4000
# Log the raw tool output and the compressed copy actually sent to the model.
LOG_RAW_TOOL_OUTPUT = True
# The system prompt is large and near-static, so only a hash + length is stored
# by default. Flip this on when you are debugging the prompt itself.
LOG_SYSTEM_PROMPT_TEXT = False

# ------------------------------------------------------------------
# Database identity.
#
# IMPORTANT ASYMMETRY: the MySQL MCP server is pinned to a single database via
# environment variables, so its tools need no database argument. Every MongoDB
# tool, by contrast, takes `database` (and usually `collection`) as a *required
# argument*. If the model does not know these names it cannot query MongoDB at
# all — so they are declared here, injected into the system prompt, and
# auto-filled by mcp_manager when the model omits them.
# ------------------------------------------------------------------
MONGODB_DATABASE = "test_db"
MONGODB_COLLECTIONS = ["student_activities"]
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "school_db")

SERVERS = {
    "mongodb": {
        "url": "http://127.0.0.1:8002/mcp",
        "transport": "streamable_http",
        "label": "MongoDB (unstructured student data)",
        "allowed_tools": [
            "find",
            "aggregate",
            "count",
            "list-databases",
            "list-collections",
            "collection-schema",
        ],
        # Auto-filled when the model leaves the argument out entirely.
        "default_arguments": {"database": MONGODB_DATABASE},
    },
    "mysql": {
        "url": "http://127.0.0.1:8001/sse",
        "transport": "sse",
        "label": "MySQL (structured student records)",
        "allowed_tools": ["execute_sql", "get_schema_info", "get_table_sample"],
        "default_arguments": {},
    },
}

# ------------------------------------------------------------------
# System prompt.
#
# Built in two parts so the "authoritative schema" the prompt promises is
# actually delivered: the app calls mcp_manager.discover_schema() at startup
# and passes the result to build_system_prompt().
# ------------------------------------------------------------------
_PROMPT_BODY = f"""You are a helpful data assistant for a school system.

You have access to tools from two different databases:
- A MySQL database containing STRUCTURED student records: names, roll numbers,
  marks per subject, attendance percentage.
- A MongoDB database containing UNSTRUCTURED student data: extracurricular
  activities, teacher remarks, disciplinary notes, project details, skills.

DECLARED SCHEMAS:
1. MySQL (structured data)
   - Database: `{MYSQL_DATABASE}` (the MySQL server is already connected to it,
     so you never pass a database name to MySQL tools)
   - Table: `students`
   - Columns: `roll_no`, `name`, `class`, `section`, `math_marks`,
     `science_marks`, `english_marks`, `attendance_percentage`

2. MongoDB (unstructured data)
   - Database: `{MONGODB_DATABASE}`
   - Collection: `{MONGODB_COLLECTIONS[0]}`
   - Fields: `student_id` (matches MySQL `roll_no`), `name`, `extracurricular`,
     `teacher_remarks`, `disciplinary_notes`, `counselor_notes`, `projects`,
     `skills`, `awards`, `attendance_flags`

USING THE MongoDB TOOLS — read this carefully, these tools fail differently
from the MySQL ones:
- Every MongoDB tool requires a `database` argument. Always pass
  `"database": "{MONGODB_DATABASE}"`. Never pass the MySQL database name to a
  MongoDB tool.
- `find`, `aggregate`, `count` and `collection-schema` also require a
  `collection` argument. Use `"collection": "{MONGODB_COLLECTIONS[0]}"` unless
  the live schema below lists a different collection.
- `filter` must be a JSON object, not a string. Use `{{}}` to match every
  document. `aggregate` takes a `pipeline` array.
- A MongoDB query that returns ZERO documents is NOT an error and does NOT
  prove the data is missing. Before telling the user nothing exists, confirm
  the names with `list-collections` / `collection-schema` and consider value
  types — `student_id` may be stored as a number while `roll_no` reads as a
  string, so try both (e.g. `5` and `"5"`).
- To join the two sources, fetch `roll_no` from MySQL and match it against
  `student_id` in MongoDB.

For every user question:
1. Decide whether it needs structured data, unstructured data, or both, based
   on the schema information provided.
2. Call the appropriate tool(s) with correctly formatted arguments — valid SQL
   for MySQL tools, valid JSON filters or aggregation pipelines for MongoDB
   tools — using the exact table/collection/column names given.
3. If a tool call returns an error, read the error message carefully — it
   usually means a name or syntax mistake. Fix it using the schema information
   and call the tool again. You have several attempts; use them.
4. Once you have the data you need, answer the user's question in clear,
   natural language. Do not show raw JSON, SQL, or tool output to the user
   unless they explicitly ask for the raw data.
5. Never invent data you do not have. If a tool genuinely cannot answer the
   question, say so honestly and state what you tried.
"""

_NO_LIVE_SCHEMA = """LIVE SCHEMA: not available — the schema probe did not run or both
servers were unreachable. Use the DECLARED SCHEMAS above, and verify names with
`list-collections` / `collection-schema` / `get_schema_info` before concluding
that data does not exist."""


def build_system_prompt(live_schema: str = "") -> str:
    """
    Compose the system prompt, appending schema details read live from the MCP
    servers. Live names are authoritative and override the declared block.
    """
    if live_schema and live_schema.strip():
        tail = (
            "LIVE SCHEMA (read directly from the running servers — these names are\n"
            "authoritative; prefer them over the declared block above):\n\n"
            f"{live_schema.strip()}"
        )
    else:
        tail = _NO_LIVE_SCHEMA
    return f"{_PROMPT_BODY}\n{tail}\n"


# Static fallback for callers that have no live schema to inject.
SYSTEM_PROMPT = build_system_prompt()
