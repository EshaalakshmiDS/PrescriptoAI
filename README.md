# PrescriptoAI

> AI-powered prescription intelligence system — OCR + multi-agent LangGraph pipeline that transforms handwritten or printed medical prescriptions into structured, interpreted, and risk-assessed outputs.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [System Architecture](#system-architecture)
4. [LangGraph Pipeline — Deep Dive](#langgraph-pipeline--deep-dive)
   - [State Design](#state-design)
   - [Graph Topology](#graph-topology)
   - [Conditional Edges](#conditional-edges)
   - [Retry Cycles](#retry-cycles)
   - [Parallel Branches & Fan-in](#parallel-branches--fan-in)
   - [State Reducers](#state-reducers)
   - [MemorySaver Checkpointing](#memorysaver-checkpointing)
5. [Node Reference](#node-reference)
6. [API Reference](#api-reference)
7. [Design Decisions](#design-decisions)
8. [Project Structure](#project-structure)
9. [Setup & Running](#setup--running)
10. [Known Limitations](#known-limitations)
11. [Future Roadmap](#future-roadmap)

---

## Overview

PrescriptoAI is a **backend-first, reasoning-driven AI system** that goes far beyond basic OCR. Instead of a single LLM call, it models prescription understanding as a stateful, multi-agent workflow with decision gates, retry loops, parallel analysis, and severity-based routing.

**What it does:**

- Extracts text from prescription images using Gemini's vision capability
- Self-assesses OCR confidence and retries with targeted prompts when needed
- Structures extracted text into validated, typed JSON (medications, dosage, frequency, duration)
- Attempts field recovery if critical information is missing
- Runs reasoning and risk detection **in parallel**
- Routes output through severity-appropriate alert nodes before returning
- Supports follow-up Q&A via a chat endpoint with conversation memory

**Disclaimer:** This system is an assistive tool only. It does not provide medical diagnoses or clinical decisions.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Backend** | FastAPI | Async-native, automatic OpenAPI docs, clean dependency injection |
| **LLM** | Google Gemini 2.0 Flash | Free tier, multimodal (handles vision + text), fast response times |
| **LLM Framework** | LangChain (`langchain-google-genai`) | Clean Gemini integration, structured output, prompt templating |
| **Agent Orchestration** | LangGraph | Stateful graph with cycles, conditional edges, parallel branches, checkpointing |
| **Structured Output** | Pydantic v2 + `.with_structured_output()` | Type-safe, validated LLM responses — no fragile JSON parsing |
| **Settings** | pydantic-settings | Environment variable management with type validation |
| **Image Handling** | Pillow | Image preprocessing before base64 encoding for Gemini Vision |
| **Frontend** | React (Phase 2) | Minimal upload + display UI |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                                                                 │
│   POST /upload-prescription          POST /analyze              │
│   (image → full pipeline)            (text → skip OCR)         │
│              │                              │                   │
│              └──────────────┬───────────────┘                   │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │  LangGraph      │                          │
│                    │  StateGraph     │                          │
│                    │                 │                          │
│                    │  PrescriptionState flows                   │
│                    │  through nodes, each                       │
│                    │  reading + writing its slice               │
│                    └────────┬────────┘                          │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │  Gemini Flash   │ ← vision + text LLM      │
│                    │  (via LangChain)│                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## LangGraph Pipeline — Deep Dive

### State Design

The entire pipeline shares a single `PrescriptionState` TypedDict. Each node reads from and writes to its own slice of this state. No node can corrupt another node's data because writes are additive and LangGraph merges them cleanly.

```python
class PrescriptionState(TypedDict):
    # Input
    image_bytes: Optional[bytes]         # from /upload-prescription
    raw_text_input: Optional[str]        # from /analyze (bypasses OCR)

    # OCR stage
    raw_text: Optional[str]
    ocr_confidence: Optional[float]      # 0.0–1.0, Gemini self-assessed
    ocr_retry_count: int                 # loop guard

    # Cleaning
    cleaned_text: Optional[str]
    detected_language: Optional[str]
    medical_tokens: Optional[list[str]]

    # Structuring
    structured_data: Optional[dict]
    missing_fields: Optional[list[str]]
    structuring_retry_count: int         # loop guard

    # Parallel analysis outputs
    interpretation: Optional[dict]       # written by reasoning_node
    risk_assessment: Optional[dict]      # written by risk_node
    risk_level: Optional[str]            # "high" | "medium" | "low"

    # Severity nodes
    critical_alerts: Optional[list[str]]

    # Final
    final_output: Optional[dict]

    # Accumulating fields — use operator.add REDUCER (safe for parallel writes)
    warnings: Annotated[list[str], operator.add]
    processing_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
```

---

### Graph Topology

```
                    [image input]           [text input]
                         │                      │
                         ▼                      │
              ┌── set_conditional_entry_point ──┘
              │
              ▼ "ocr"                           ▼ "clean_text"
         ┌─────────┐                      ┌────────────┐
         │   OCR   │ ◄────────────────────┤            │
         │  Node   │   (retry cycle)      │            │
         └────┬────┘                      │            │
              │                           │            │
    [route_ocr_quality]                   │            │
              │                           │            │
    ┌─────────┼──────────┐                │            │
    │         │          │                │            │
    ▼         ▼          ▼                │            │
enhance   clean_text  flag_unread         │            │
  _ocr       │        able ──► END        │            │
    │        │                            │            │
    └──► ocr │ ◄──────────────────────────┘            │
             │                                         │
             ▼                                         │
        ┌─────────┐ ◄───────────────────────────────── ┘
        │ clean   │
        │  text   │
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │structure│ ◄──────────────────────┐
        │  node   │                        │
        └────┬────┘                        │
             │                             │
   [route_completeness]                    │
             │                             │
    ┌────────┴──────────┐                  │
    │                   │                  │
    ▼                   ▼                  │
recover            dispatch                │
_fields ───────────► _analysis            │
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

---

### Conditional Edges

Conditional edges are the decision-making backbone of the graph. Each one is a pure function that inspects state and returns a routing key.

**`route_start`** — virtual entry point
```
image_bytes present  → "ocr"
raw_text_input set   → "clean_text"   (skip OCR entirely)
```

**`route_ocr_quality`** — called after every OCR attempt
```
confidence >= threshold             → "clean_text"
confidence <  threshold, retries remain → "enhance_ocr"   (CYCLE)
confidence <  threshold, budget gone    → "flag_unreadable" (DEAD END)
```

**`route_completeness`** — called after every structuring attempt
```
no missing_fields                    → "dispatch_analysis"
missing_fields, retries remain       → "recover_fields"   (CYCLE)
missing_fields, budget gone          → "dispatch_analysis" (proceed with warnings)
```

**`route_severity`** — called after parallel analysis merges
```
risk_level == "high"    → "critical_alert"
risk_level == "medium"  → "advisory"
risk_level == "low"     → "format_output"
```

---

### Retry Cycles

LangGraph natively supports cycles — a node can route back to an earlier node. PrescriptoAI uses two retry cycles:

**OCR Retry Cycle**
```
ocr_node → [low confidence] → ocr_enhancement_node → ocr_node → ...
```
- `ocr_enhancement_node` increments `ocr_retry_count` and re-runs OCR with a targeted prompt that focuses on the previously flagged low-confidence areas
- `route_ocr_quality` checks the counter against `MAX_OCR_RETRIES` (default: 3)
- If budget exhausted → routes to `flag_unreadable` (terminal node → END)

**Structuring Recovery Cycle**
```
structuring_node → [missing fields] → field_recovery_node → structuring_node → ...
```
- `field_recovery_node` passes partial structured data + missing field list back to Gemini with a context-recovery prompt
- `route_completeness` checks `structuring_retry_count` against `MAX_STRUCTURING_RETRIES` (default: 2)
- If budget exhausted → proceeds to analysis with whatever data was recovered

---

### Parallel Branches & Fan-in

After structuring completes, `dispatch_analysis_node` (a lightweight no-op) fans out to two branches that LangGraph executes **concurrently**:

```python
builder.add_edge("dispatch_analysis", "reason")
builder.add_edge("dispatch_analysis", "detect_risk")
```

- `reasoning_node` — interprets medications, decodes abbreviations (writes → `interpretation`)
- `risk_node` — flags missing data, ambiguity, OCR issues (writes → `risk_assessment`, `risk_level`)

Both write to **different state keys** so there is no conflict. LangGraph automatically waits for both branches before proceeding to `merge_analysis` (fan-in):

```python
builder.add_edge("reason", "merge_analysis")
builder.add_edge("detect_risk", "merge_analysis")
```

This is a classic fork-join pattern — identical to parallel async task execution, but expressed declaratively in the graph topology.

---

### State Reducers

Fields that are written by multiple nodes (including concurrent parallel branches) use `Annotated` with `operator.add` as a reducer:

```python
warnings:         Annotated[list[str], operator.add]
processing_steps: Annotated[list[str], operator.add]
errors:           Annotated[list[str], operator.add]
```

Without reducers, two parallel nodes writing to the same list field would cause a conflict. With `operator.add`, LangGraph **concatenates** the lists from both branches. This is a core LangGraph feature for safe parallel state mutation.

---

### MemorySaver Checkpointing

The graph is compiled in two modes:

| Mode | Checkpointer | Used by |
|---|---|---|
| `prescription_graph` | None (stateless) | `/upload-prescription`, `/analyze` |
| `chat_graph` | `MemorySaver` | `/chat` |

```python
prescription_graph = build_graph(with_memory=False)
chat_graph = build_graph(with_memory=True)
```

The `/chat` endpoint passes a `thread_id` as the LangGraph config key. MemorySaver stores the full conversation history per thread, enabling multi-turn Q&A grounded in the previously analysed prescription — without the client needing to resend context on every turn.

---

## Node Reference

| Node | Type | LangGraph Role | Key Behaviour |
|---|---|---|---|
| `ocr_node` | LLM (vision) | Entry (image path) | Gemini Flash Vision → raw text + confidence score |
| `ocr_enhancement_node` | LLM (vision) | Cycle node | Re-OCR with targeted prompt, increments retry counter |
| `flag_unreadable_node` | Logic | Terminal dead-end | Sets error state, terminates pipeline |
| `cleaning_node` | Pure Python | Sequential | Regex normalisation, abbreviation standardisation, no LLM |
| `structuring_node` | LLM (structured) | Sequential + cycle source | `.with_structured_output(StructuredPrescription)` |
| `field_recovery_node` | LLM (structured) | Cycle node | Context-aware field inference, loops back to structuring |
| `dispatch_analysis_node` | No-op | Fan-out coordinator | Triggers parallel branches |
| `reasoning_node` | LLM (structured) | Parallel branch | Medication interpretations, abbreviation decoding |
| `risk_node` | LLM (structured) | Parallel branch | Risk flags, severity assessment, warnings |
| `merge_node` | Logic | Fan-in | Waits for both parallel branches, normalises risk_level |
| `critical_alert_node` | Logic | Conditional branch | Escalates high-risk flags to critical_alerts |
| `advisory_node` | Logic | Conditional branch | Converts medium-risk flags to advisory warnings |
| `output_formatter_node` | Logic | Terminal | Assembles final_output dict from all state slices |

---

## API Reference

### `POST /api/upload-prescription`

Upload a prescription image. Runs the full pipeline.

**Request:** `multipart/form-data`
```
file: <image file>   # JPEG, PNG, WEBP, HEIC
```

**Response:**
```json
{
  "structured_data": {
    "patient_name": "John Doe",
    "doctor_name": "Dr. Smith",
    "date": "2026-03-22",
    "medications": [
      {
        "name": "Amoxicillin",
        "dosage": "500mg",
        "frequency": "TDS",
        "duration": "7 days",
        "route": "oral",
        "instructions": "Take with food"
      }
    ],
    "notes": "Follow up in 1 week",
    "missing_fields": []
  },
  "interpretation": {
    "medication_interpretations": [
      {
        "medication_name": "Amoxicillin",
        "common_use": "Commonly used for bacterial infections such as respiratory tract infections",
        "dosage_explanation": "500mg three times a day for 7 days",
        "important_notes": "Complete the full course even if symptoms improve"
      }
    ],
    "instruction_summary": "Take Amoxicillin three times daily with food for one week. Return for follow-up.",
    "abbreviations_decoded": { "TDS": "Three times daily" },
    "disclaimer": "This is for informational purposes only and does not constitute medical advice."
  },
  "risk_assessment": {
    "risk_level": "low",
    "flags": [],
    "missing_critical_info": [],
    "ambiguous_instructions": [],
    "warnings": []
  },
  "risk_level": "low",
  "warnings": [],
  "critical_alerts": [],
  "ocr_confidence": 0.91,
  "processing_steps": [
    "[OCR] Extracted text — confidence: 91%",
    "[Cleaning] Normalised text — language: en, medical tokens found: 4",
    "[Structuring #1] Extracted 1 medication(s) — missing fields: none",
    "[Dispatch] Fanning out to Reasoning + Risk Detection in parallel.",
    "[Reasoning] Interpreted 1 medication(s) — 1 abbreviation(s) decoded.",
    "[Risk Detection] Level: LOW — 0 flag(s), 0 warning(s).",
    "[Merge] Parallel branches joined — risk level confirmed: LOW.",
    "[Output Formatter] Pipeline complete. Final output assembled."
  ],
  "errors": []
}
```

---

### `POST /api/analyze`

Analyze raw prescription text (no image needed).

**Request:**
```json
{
  "text": "Patient: John Doe\nAmoxicillin 500mg TDS x 7 days\nTake with food\nDr. Smith"
}
```

**Response:** Same schema as `/upload-prescription` (without `ocr_confidence`).

---

### `POST /api/chat`

Multi-turn Q&A about an analysed prescription.

**Request:**
```json
{
  "message": "What is Amoxicillin commonly used for?",
  "thread_id": "session-abc123",
  "prescription_context": { ... }   // from a previous /analyze or /upload call
}
```

**Response:**
```json
{
  "reply": "Amoxicillin is commonly prescribed for bacterial infections...",
  "thread_id": "session-abc123"
}
```

---

## Project Structure

```
PrescriptoAI/
├── backend/
│   ├── main.py                          # FastAPI app, CORS, router registration
│   ├── config.py                        # Settings from .env (pydantic-settings)
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── prescription.py          # POST /upload-prescription, POST /analyze
│   │       └── chat.py                  # POST /chat
│   │
│   ├── pipeline/
│   │   ├── state.py                     # PrescriptionState TypedDict
│   │   ├── graph.py                     # Graph assembly + compile
│   │   ├── edges.py                     # All conditional edge functions
│   │   └── nodes/
│   │       ├── ocr_node.py              # Gemini Vision → raw text
│   │       ├── ocr_enhancement_node.py  # Retry OCR with targeted prompt
│   │       ├── flag_unreadable_node.py  # Terminal dead-end
│   │       ├── cleaning_node.py         # Pure Python normalisation
│   │       ├── structuring_node.py      # → StructuredPrescription
│   │       ├── field_recovery_node.py   # Recover missing fields
│   │       ├── dispatch_analysis_node.py# Fan-out coordinator (no-op)
│   │       ├── reasoning_node.py        # Parallel: interpretation
│   │       ├── risk_node.py             # Parallel: risk assessment
│   │       ├── merge_node.py            # Fan-in: combine parallel results
│   │       ├── critical_alert_node.py   # High-risk escalation
│   │       ├── advisory_node.py         # Medium-risk advisory
│   │       └── output_formatter_node.py # Final output assembly
│   │
│   ├── schemas/
│   │   └── models.py                    # Pydantic models (LLM output + API I/O)
│   │
│   └── utils/
│       └── prompts.py                   # All LangChain ChatPromptTemplates
│
├── frontend/                            # React UI (Phase 2)
├── .env                                 # GEMINI_API_KEY (not committed)
├── .env.example                         # Template
├── requirements.txt
└── README.md
```

---

## Setup & Running

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd PrescriptoAI
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com).

### 3. Run the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Explore the API

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Known Limitations

| Limitation | Impact |
|---|---|
| Gemini self-reported confidence | Confidence scores are heuristic — not a calibrated probability |
| OCR on very low-quality images | Even 3 retry attempts may not recover severely degraded images |
| No real drug database | Medication interpretations are LLM knowledge, not a validated drug DB |
| LLM hallucination risk | Structured output reduces but doesn't eliminate hallucination |
| No persistent storage | Results are not stored between sessions (Phase 3) |
| English-optimised prompts | Non-English prescriptions may have reduced accuracy |

---

## Future Roadmap

### Phase 2 — Reliability
- Drug database integration (OpenFDA API) for validated medication lookup
- Confidence scoring per extracted field (not just overall OCR)
- Structured output validation layer with cross-field consistency checks

### Phase 3 — Production
- PostgreSQL persistence — store prescriptions, results, and chat sessions
- User authentication (JWT)
- Rate limiting and API key management
- Frontend React UI — upload, display, chat interface

### Phase 4 — Intelligence
- Multi-language support (Hindi, Arabic, French prescriptions)
- Drug interaction checking (flag potential interactions between medications on the same prescription)
- Real-time feedback loop — pharmacist corrections improve prompts over time

---
