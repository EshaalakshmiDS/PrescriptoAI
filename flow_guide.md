# PrescriptoAI — Codebase Flow Guide

> Reference document for understanding the internal architecture, LangGraph design, state management, node behaviour, and frontend structure. This is the deep-dive companion to the README.

---

## Table of Contents

1. [LangGraph State Design](#langgraph-state-design)
2. [Full Graph Topology](#full-graph-topology)
3. [Conditional Edges — Decision Logic](#conditional-edges--decision-logic)
4. [Retry Cycles](#retry-cycles)
5. [Parallel Branches & Fan-in](#parallel-branches--fan-in)
6. [State Reducers](#state-reducers)
7. [MemorySaver Checkpointing (Chat)](#memorysaver-checkpointing-chat)
8. [Node Reference](#node-reference)
9. [Pydantic Models (LLM Output)](#pydantic-models-llm-output)
10. [Prompts](#prompts)
11. [FastAPI Routes — How They Map to the Graph](#fastapi-routes--how-they-map-to-the-graph)
12. [Frontend Component Flow](#frontend-component-flow)
13. [Key Design Decisions](#key-design-decisions)
14. [Original Implementation Notes](#original-implementation-notes)

---

## LangGraph State Design

The entire pipeline shares a single `PrescriptionState` TypedDict (`backend/pipeline/state.py`). Each node reads from and writes to its own slice of this state. No node can corrupt another's data — writes are merged by LangGraph cleanly.

```python
class PrescriptionState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────
    image_bytes: Optional[bytes]         # set by /upload-prescription
    raw_text_input: Optional[str]        # set by /analyze (bypasses OCR)

    # ── OCR stage ────────────────────────────────────────────────────
    raw_text: Optional[str]
    ocr_confidence: Optional[float]      # 0.0–1.0, Gemini self-assessed
    ocr_retry_count: int                 # loop guard: max = settings.max_ocr_retries
    low_confidence_areas: Optional[list[str]]  # regions Gemini flagged

    # ── Cleaning stage ────────────────────────────────────────────────
    cleaned_text: Optional[str]
    detected_language: Optional[str]
    medical_tokens: Optional[list[str]]  # recognised abbreviations / keywords

    # ── Structuring stage ─────────────────────────────────────────────
    structured_data: Optional[dict]      # medications[], notes, doctor_instructions
    missing_fields: Optional[list[str]]  # fields Gemini could not extract
    structuring_retry_count: int         # loop guard

    # ── Parallel analysis ─────────────────────────────────────────────
    interpretation: Optional[dict]       # plain-English explanation (reasoning node)
    risk_assessment: Optional[dict]      # flagged issues (risk node)
    risk_level: Optional[str]            # "high" | "medium" | "low"

    # ── Severity-specific ─────────────────────────────────────────────
    critical_alerts: Optional[list[str]]

    # ── Final output ──────────────────────────────────────────────────
    final_output: Optional[dict]

    # ── Accumulating fields (reducer = list concatenation) ────────────
    # Safe for parallel branches: both reason + detect_risk can append
    warnings: Annotated[list[str], operator.add]
    processing_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
```

**Why TypedDict over dataclass/BaseModel?** LangGraph requires TypedDict for state — it uses the type annotations to understand which fields are mergeable and how.

---

## Full Graph Topology

```
                [image input]           [text input]
                      │                      │
                      ▼                      │
           add_conditional_edges(START, route_start)
                      │
           ┌──────────┴──────────┐
           ▼ "ocr"               ▼ "clean_text"
      ┌─────────┐           ┌────────────┐
      │   OCR   │           │            │
      │  Node   │           │            │
      └────┬────┘           │            │
           │                │            │
  [route_ocr_quality]       │            │
           │                │            │
  ┌────────┼──────────┐     │            │
  │        │          │     │            │
  ▼        ▼          ▼     │            │
enhance  clean_text  flag_  │            │
 _ocr       │       unread  │            │
  │         │        able   │            │
  └──► ocr  │         └─►  END           │
            │                            │
            ▼                            │
       ┌─────────┐ ◄───────────────────── ┘
       │  clean  │
       │  text   │
       └────┬────┘
            │
            ▼
       ┌─────────┐ ◄──────────────────────┐
       │structure│                        │
       │  node   │                        │
       └────┬────┘                        │
            │                             │
  [route_completeness]                    │
            │                             │
   ┌────────┴──────────┐                  │
   │                   │                  │
   ▼                   ▼                  │
recover           dispatch                │
_fields ──────────► _analysis            │
   │           (fan-out node)             │
   └──────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           │    PARALLEL EXECUTION  │
           ▼                        ▼
      ┌─────────┐            ┌─────────────┐
      │ reason  │            │ detect_risk │
      │  node   │            │    node     │
      └────┬────┘            └──────┬──────┘
           │                        │
           └──────────┬─────────────┘
                      ▼
                ┌───────────┐
                │   merge   │
                │ _analysis │
                └─────┬─────┘
                      │
            [route_severity]
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   critical_alert  advisory   format_output
         │            │            ▲
         └────────────┴────────────┘
                                   │
                                  END
```

Graph is built in `backend/pipeline/graph.py` using `StateGraph(PrescriptionState)`.

---

## Conditional Edges — Decision Logic

All conditional edge functions live in `backend/pipeline/edges.py`. Each is a pure function: takes state, returns a routing string.

### `route_start` — virtual entry point
```
image_bytes present   → "ocr"
raw_text_input set    → "clean_text"   (skip OCR entirely)
```
Registered with: `builder.add_conditional_edges(START, route_start, {"ocr": "ocr", "clean_text": "clean_text"})`

---

### `route_ocr_quality` — after every OCR attempt
```
confidence >= OCR_CONFIDENCE_THRESHOLD      → "clean_text"
confidence <  threshold, retries remain     → "enhance_ocr"   (CYCLE back)
confidence <  threshold, budget exhausted   → "flag_unreadable" (DEAD END → END)
```
`OCR_CONFIDENCE_THRESHOLD` defaults to 0.7. `max_ocr_retries` defaults to 3 (configurable via settings).

---

### `route_completeness` — after every structuring attempt
```
no missing_fields                            → "dispatch_analysis"
missing_fields present, retries remain       → "recover_fields"   (CYCLE back)
missing_fields present, budget exhausted     → "dispatch_analysis" (proceed with warnings)
```
Even when fields are missing, the pipeline continues rather than terminating — the risk node will flag the missing info.

---

### `route_severity` — after parallel analysis merges
```
risk_level == "high"    → "critical_alert"
risk_level == "medium"  → "advisory"
risk_level == "low"     → "format_output"
```
`critical_alert_node` and `advisory_node` both flow into `format_output` after processing.

---

## Retry Cycles

LangGraph natively supports cycles — a node can route back to an earlier node. PrescriptoAI uses two:

### OCR Retry Cycle
```
ocr_node → [low confidence] → ocr_enhancement_node → ocr_node → ...
```
- `ocr_enhancement_node` increments `ocr_retry_count` and re-runs OCR with a targeted prompt that focuses on the previously flagged low-confidence areas
- `route_ocr_quality` checks the counter against `max_ocr_retries`
- If budget exhausted → routes to `flag_unreadable_node` (terminal → END)

### Structuring Recovery Cycle
```
structuring_node → [missing fields] → field_recovery_node → structuring_node → ...
```
- `field_recovery_node` sends partial structured data + missing field list back to Gemini with a context-recovery prompt
- `route_completeness` checks `structuring_retry_count` against `max_structuring_retries` (default: 2)
- If budget exhausted → proceeds to analysis with whatever was recovered, with warnings appended

---

## Parallel Branches & Fan-in

After structuring, `dispatch_analysis_node` (a lightweight no-op that just logs) fans out to two branches that LangGraph executes **concurrently**:

```python
builder.add_edge("dispatch_analysis", "reason")
builder.add_edge("dispatch_analysis", "detect_risk")
```

- `reasoning_node` — interprets medications, decodes abbreviations → writes to `interpretation`
- `risk_node` — flags missing data, ambiguity, OCR issues → writes to `risk_assessment`, `risk_level`

Both write to **different state keys** so there is zero conflict. LangGraph automatically waits for both branches before advancing to `merge_analysis` (fan-in):

```python
builder.add_edge("reason", "merge_analysis")
builder.add_edge("detect_risk", "merge_analysis")
```

This is a fork-join pattern — identical to `asyncio.gather()` but expressed declaratively in the graph topology.

---

## State Reducers

Fields written by multiple concurrent nodes use `Annotated` with `operator.add` as a reducer:

```python
warnings:         Annotated[list[str], operator.add]
processing_steps: Annotated[list[str], operator.add]
errors:           Annotated[list[str], operator.add]
```

Without reducers, two parallel nodes writing to the same list field would cause a conflict — LangGraph would not know which write to keep. With `operator.add`, it **concatenates** the lists from both branches. This is a core LangGraph feature for safe parallel state mutation.

All other fields (e.g., `interpretation`, `risk_assessment`) are written by exactly one node each, so no reducer is needed.

---

## MemorySaver Checkpointing (Chat)

The graph is compiled in two modes:

| Mode | Checkpointer | Used by |
|---|---|---|
| `prescription_graph` | None (stateless) | `/upload-prescription`, `/analyze` |
| `chat_graph` | `MemorySaver` | `/chat` |

```python
prescription_graph = build_graph(with_memory=False)
chat_graph         = build_graph(with_memory=True)
```

The `/chat` endpoint passes a `thread_id` as the LangGraph config key:
```python
config = {"configurable": {"thread_id": request.thread_id}}
result = chat_graph.invoke(initial_state, config=config)
```

MemorySaver stores the **full conversation history** per thread in memory (not persisted to disk). This enables multi-turn Q&A grounded in the previously analysed prescription — without the client needing to resend context on every turn.

For production, swap `MemorySaver` with `PostgresSaver` or `SqliteSaver`.

---

## Node Reference

| Node | File | Type | Key Behaviour |
|---|---|---|---|
| `ocr_node` | `nodes/ocr_node.py` | LLM (vision) | Gemini Flash Vision → raw text + confidence score via direct `HumanMessage` (not template, because images need manual base64 encoding) |
| `ocr_enhancement_node` | `nodes/ocr_enhancement_node.py` | LLM (vision) | Re-OCR with targeted prompt focusing on low-confidence areas; increments `ocr_retry_count` |
| `flag_unreadable_node` | `nodes/flag_unreadable_node.py` | Logic | Sets error state, terminates pipeline via `END` |
| `cleaning_node` | `nodes/cleaning_node.py` | Pure Python | Regex normalisation, abbreviation standardisation, language detection — **no LLM call** |
| `structuring_node` | `nodes/structuring_node.py` | LLM (structured) | `.with_structured_output(StructuredPrescription)` → typed, validated JSON |
| `field_recovery_node` | `nodes/field_recovery_node.py` | LLM (structured) | Context-aware field inference, loops back to structuring |
| `dispatch_analysis_node` | `nodes/dispatch_analysis_node.py` | No-op | Logs dispatch step; triggers parallel branches by virtue of two outgoing edges |
| `reasoning_node` | `nodes/reasoning_node.py` | LLM (structured) | Parallel branch: medication interpretations, abbreviation decoding → `ReasoningOutput` |
| `risk_node` | `nodes/risk_node.py` | LLM (structured) | Parallel branch: risk flags, severity assessment, warnings → `RiskOutput` |
| `merge_node` | `nodes/merge_node.py` | Logic | Fan-in: waits for both parallel branches, normalises `risk_level` if None |
| `critical_alert_node` | `nodes/critical_alert_node.py` | Logic | Escalates high-risk flags → `critical_alerts` list |
| `advisory_node` | `nodes/advisory_node.py` | Logic | Converts medium-risk flags to advisory-level warnings |
| `output_formatter_node` | `nodes/output_formatter_node.py` | Logic | Assembles `final_output` dict from all state slices |

---

## Pydantic Models (LLM Output)

Defined in `backend/schemas/models.py`. All LLM output models have **defaults on every field** — this is critical to prevent `OutputParserException` when Gemini returns partial JSON.

| Model | Used by | Key fields |
|---|---|---|
| `OCROutput` | `ocr_node`, `ocr_enhancement_node` | `extracted_text`, `confidence_score`, `low_confidence_areas` |
| `StructuredPrescription` | `structuring_node`, `field_recovery_node` | `medications: list[MedicationEntry]`, `missing_fields` |
| `MedicationEntry` | nested in `StructuredPrescription` | `name`, `dosage`, `frequency`, `duration`, `route`, `instructions` |
| `ReasoningOutput` | `reasoning_node` | `medication_interpretations`, `abbreviations_decoded`, `instruction_summary` |
| `RiskOutput` | `risk_node` | `risk_level`, `flags: list[RiskFlag]`, `warnings`, `missing_critical_info` |
| `RiskFlag` | nested in `RiskOutput` | `field`, `issue`, `severity` |

FastAPI request/response schemas (also in `models.py`):

| Schema | Used by |
|---|---|
| `AnalyzeTextRequest` | `POST /analyze` |
| `PrescriptionResponse` | both prescription endpoints |
| `ChatRequest` / `ChatResponse` | `POST /chat` |

---

## Prompts

All prompts live in `backend/utils/prompts.py` as `ChatPromptTemplate` instances (except OCR, which uses direct `HumanMessage` construction).

| Prompt | Template vars | Purpose |
|---|---|---|
| `OCR_SYSTEM_PROMPT` | — | System instruction for OCR extraction + confidence self-assessment |
| `STRUCTURING_PROMPT` | `cleaned_text` | Convert normalised text to `StructuredPrescription` |
| `RECOVERY_PROMPT` | `cleaned_text`, `structured_data`, `missing_fields` | Fill in missing fields given partial data |
| `REASONING_PROMPT` | `structured_data` | Generate `ReasoningOutput` — interpret medications |
| `RISK_PROMPT` | `structured_data`, `low_confidence_areas` | Generate `RiskOutput` — flag issues |
| `CHAT_PROMPT` | `prescription_context`, `history`, `input` | Ground chat responses in prescription data |

**Why direct `HumanMessage` for OCR (not `ChatPromptTemplate`)?** `langchain-google-genai` 4.x requires the image to be embedded as a base64 data URL inside the `HumanMessage` content list. `ChatPromptTemplate` does not support this format natively, so `ocr_node` and `ocr_enhancement_node` build the message manually:
```python
message = HumanMessage(content=[
    {"type": "text", "text": prompt_text},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
])
```

---

## FastAPI Routes — How They Map to the Graph

**`POST /api/upload-prescription`** (`backend/api/routes/prescription.py`)
1. Reads image bytes from `UploadFile`
2. Calls `_build_initial_state(image_bytes=image_bytes)` → dict with all state fields initialised
3. Runs `asyncio.to_thread(prescription_graph.invoke, state)` (sync graph in async handler)
4. Maps final state to `PrescriptionResponse` and returns

**`POST /api/analyze`** (`backend/api/routes/prescription.py`)
1. Takes `AnalyzeTextRequest` (JSON body with `text`)
2. Calls `_build_initial_state(raw_text_input=text)` → `route_start` will skip OCR
3. Same invoke + mapping as above

**`POST /api/chat`** (`backend/api/routes/chat.py`)
1. Takes `ChatRequest` with `message`, `thread_id`, `prescription_context`
2. Builds state with chat-specific fields
3. Runs `chat_graph.invoke(state, config={"configurable": {"thread_id": thread_id}})`
4. MemorySaver stores history keyed by `thread_id`
5. Returns `ChatResponse` with `reply` and `thread_id`

**`_build_initial_state` pattern** — uses a defaults dict + `dict.update(kwargs)` to avoid "multiple values for keyword argument" errors that arise when both `image_bytes` and `raw_text_input` have defaults in the function signature:
```python
def _build_initial_state(**kwargs) -> dict:
    state = {
        "image_bytes": None,
        "raw_text_input": None,
        "ocr_retry_count": 0,
        "structuring_retry_count": 0,
        "warnings": [],
        "processing_steps": [],
        "errors": [],
        # ... all other fields = None
    }
    state.update(kwargs)
    return state
```

---

## Frontend Component Flow

```
App.jsx
  ├── UploadForm.jsx          ← user uploads image or pastes text
  │     │
  │     └── on submit → api.js: uploadPrescription() or analyzeText()
  │                             ↓ response
  ├── ResultDisplay.jsx       ← receives PrescriptionResponse
  │     ├── MedicationCard.jsx (one per medication)
  │     │     ├── structured fields (name, dosage, frequency, ...)
  │     │     └── interpretation panel (common_use, dosage_explanation)
  │     ├── WarningsPanel.jsx
  │     │     ├── risk level badge (red/yellow/green)
  │     │     ├── critical_alerts list
  │     │     └── warnings list
  │     └── processing_steps accordion (pipeline audit trail)
  │
  └── ChatInterface.jsx       ← multi-turn chat
        ├── generates thread_id on mount (uuid)
        ├── sends prescription_context with first message
        └── api.js: sendChat(message, thread_id, context)
```

**State in App.jsx:**
- `result` — the last `PrescriptionResponse` from the API
- `loading` — boolean for spinner
- `error` — error message string
- `activeTab` — "upload" | "text" (UploadForm tabs)

**API calls (`frontend/src/api.js`):**
```js
uploadPrescription(formData)  // POST /api/upload-prescription (multipart)
analyzeText(text)             // POST /api/analyze (JSON)
sendChat(message, threadId, context)  // POST /api/chat (JSON)
```

All calls use Axios with the base URL set to `http://localhost:8000`.

---

## Key Design Decisions

### Why LangGraph instead of a simple function chain?
A simple Python function chain would work for the happy path, but LangGraph gives:
- **Conditional routing** — the same codebase handles image vs. text input, retry vs. proceed, high vs. low risk, without if/else spaghetti in route handlers
- **Cycles** — retry loops are first-class graph features; implementing them in plain Python requires manual loop management
- **Parallel execution** — two branches run concurrently without any `asyncio` boilerplate
- **Checkpointing** — MemorySaver drops in for conversation memory without code changes to node logic

### Why not use a single "super prompt"?
A single prompt asking for OCR + structuring + interpretation + risk in one call:
- Is harder to debug (which part failed?)
- Cannot retry individual stages
- Forces a linear dependency (can't parallelize reasoning and risk)
- Produces less focused, more error-prone outputs

### Why `.with_structured_output()` over manual JSON parsing?
Gemini (and most LLMs) occasionally return malformed JSON when asked to produce it via a plain text prompt. `.with_structured_output(PydanticModel)` uses function calling under the hood, which dramatically improves adherence. Combined with Pydantic defaults on all fields, `OutputParserException` becomes effectively impossible.

### Why `load_dotenv(override=True)` in config.py?
On Windows, pydantic-settings reads system environment variables before `.env`. If you have `GEMINI_API_KEY` set in system env (e.g., from a previous project or a typo), it silently overrides your `.env`. The `override=True` flag + loading `.env` with an absolute path guarantees `.env` always wins.

### Why `asyncio.to_thread` for graph invocation?
LangGraph's `.invoke()` is synchronous. FastAPI route handlers are async. Calling a sync function directly inside an async handler blocks the event loop. `asyncio.to_thread` runs the sync call in a thread pool, keeping the event loop free for other requests.

---

## Original Implementation Notes

The following captures the original planning notes and rationale from the project design phase.

### Problem Statement

Medical prescriptions are often:
- Handwritten and difficult to interpret
- Unstructured and inconsistent
- Prone to misinterpretation by patients

Existing OCR tools extract text but fail to interpret context or meaning. PrescriptoAI bridges that gap using agentic AI reasoning.

### MVP → Advanced Phases

**MVP (built first)**
- OCR integration via Gemini Vision
- Basic FastAPI backend
- One LLM call for structuring
- Minimal React UI

**Phase 2**
- Split into multiple agents
- Add reasoning layer
- Improve prompts
- Parallel analysis

**Phase 3 (current state)**
- Full LangGraph orchestration with 13 nodes
- Chat layer with MemorySaver
- React frontend with all components

**Phase 4 (future)**
- Persistent storage (PostgreSQL)
- Drug database integration
- Multi-language support
- Real-time feedback loop

### Interview Talking Points

- **Why multi-agent instead of single prompt** — better control, explainability, independent retries, parallel execution
- **Handling OCR noise** — confidence self-assessment, targeted retry prompts, graceful fallback to `flag_unreadable`
- **Tradeoffs in LLM reliability** — structured output + Pydantic defaults eliminates `OutputParserException`; per-node try/except ensures graceful degradation; never 500s the client
- **API design decisions** — two endpoints (image vs. text) share the same graph, differentiated only by `route_start`; chat uses a separate compiled graph with MemorySaver
- **System scalability** — stateless graph = horizontally scalable; swap MemorySaver for PostgresSaver for multi-instance deployment

### Known Gotchas Encountered During Development

1. **`TypeError: got multiple values for keyword argument 'image_bytes'`** — Fixed by using the `_build_initial_state(**kwargs)` defaults dict pattern instead of function parameters with defaults
2. **Windows system env vars overriding `.env`** — Fixed with `load_dotenv(override=True)` and absolute path resolution in `config.py`
3. **`gemini-2.0-flash` not available on free tier** — Use `gemini-2.5-flash`; discovered by iterating available models
4. **`set_conditional_entry_point` removed in LangGraph 1.1.3** — Use `add_conditional_edges(START, route_fn, mapping)` with imported `START`
5. **`OutputParserException`** — Gemini returned partial JSON missing optional fields. Fixed by adding `default=...` to all fields in all Pydantic output models
6. **uvicorn `--reload` doesn't pick up `.env` changes** — Must restart the server after `.env` edits; added startup logger to verify model + key prefix on every start
7. **Image handling in langchain-google-genai 4.x** — `ChatPromptTemplate` doesn't support image content type; use direct `HumanMessage` construction with base64 data URL

---
