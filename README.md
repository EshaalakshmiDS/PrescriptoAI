# PrescriptoAI

> AI-powered prescription intelligence system — upload a handwritten or printed prescription image and get structured medication data, plain-English interpretation, and risk warnings in seconds.

---

## ScreenShots
<img width="2560" height="678" alt="1" src="https://github.com/user-attachments/assets/a370edae-b094-4a2c-b7d2-577b2c7dc756" />
<img width="2404" height="1052" alt="2" src="https://github.com/user-attachments/assets/0366cff0-3414-47d1-a09a-301c5ae8dea8" />
<img width="2512" height="1116" alt="3" src="https://github.com/user-attachments/assets/150e672d-4d4b-4c9d-81c7-04b8a18d79b0" />
<img width="2402" height="1118" alt="4" src="https://github.com/user-attachments/assets/7c1d84ab-738a-4240-8156-62318382e679" />
<img width="648" height="994" alt="5" src="https://github.com/user-attachments/assets/4f597e1c-1d84-4920-b36f-4a0617fdfb40" />


## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [System Architecture](#system-architecture)
5. [Processing Pipeline](#processing-pipeline)
6. [Frontend](#frontend)
7. [API Reference](#api-reference)
8. [Project Structure](#project-structure)
9. [Setup & Running](#setup--running)
10. [Design Decisions](#design-decisions)
11. [Known Limitations](#known-limitations)
12. [Future Roadmap](#future-roadmap)

---

## Overview

PrescriptoAI transforms medical prescriptions into structured, actionable insights using a multi-agent AI pipeline. Instead of a single LLM call, it models prescription understanding as a stateful workflow with decision gates, retry logic, parallel analysis, and severity-based routing — all orchestrated by LangGraph.

The React frontend provides a clean interface for uploading prescriptions, viewing results, and asking follow-up questions via a built-in chat assistant.

**Disclaimer:** This is an assistive tool only. It does not provide medical diagnoses or clinical decisions.

---

## Features

### AI & Backend
- **Vision OCR** — Gemini Flash reads handwritten and printed prescriptions directly from images
- **Confidence-aware retries** — low-confidence OCR results trigger a targeted re-extraction attempt
- **Structured extraction** — medications, dosage, frequency, duration, route, and doctor instructions parsed into validated JSON
- **Plain-English interpretation** — medication purposes and dosing schedules explained in accessible language
- **Risk assessment** — flags missing dosages, ambiguous instructions, and low-confidence OCR areas with severity levels (high / medium / low)
- **Parallel analysis** — interpretation and risk detection run concurrently for faster response
- **Severity routing** — high-risk prescriptions trigger critical alert escalation; medium-risk produces advisory warnings
- **Chat Q&A** — multi-turn conversation about any analysed prescription, with full conversation memory per session

### Frontend
- **Drag-and-drop upload** — drop an image or click to browse; live preview before submission
- **Text input tab** — paste raw prescription text to skip OCR and go straight to analysis
- **Medication cards** — each medication displayed with its structured data and AI interpretation side by side
- **Color-coded risk panel** — warnings and alerts rendered with severity badges (red / yellow / green)
- **Processing timeline** — step-by-step log of what the pipeline did, great for understanding the AI's reasoning
- **Chat interface** — ask follow-up questions like "What is this medication for?" or "When should I take it?" with streaming-style responses

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI |
| **LLM** | Google Gemini 2.5 Flash (free tier) |
| **LLM Framework** | LangChain (`langchain-google-genai`) |
| **Agent Orchestration** | LangGraph (stateful graph with cycles, conditional edges, parallel branches) |
| **Structured Output** | Pydantic v2 + `.with_structured_output()` |
| **Settings** | pydantic-settings |
| **Image Handling** | Pillow |
| **Frontend** | React + Vite |
| **HTTP Client** | Axios |
| **Chat Memory** | LangGraph MemorySaver (in-memory, per thread_id) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│  UploadForm │ ResultDisplay │ MedicationCard │ ChatInterface │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP (Axios)
┌──────────────────────▼──────────────────────────────────────┐
│                       FastAPI Backend                        │
│                                                             │
│   POST /upload-prescription    POST /analyze    POST /chat  │
│              │                      │                │      │
│              └──────────┬───────────┘          LangGraph   │
│                         │                      chat_graph  │
│                ┌────────▼────────┐                         │
│                │  LangGraph      │                         │
│                │  StateGraph     │                         │
│                │  (13 nodes)     │                         │
│                └────────┬────────┘                         │
│                         │                                  │
│                ┌────────▼────────┐                         │
│                │  Gemini Flash   │  ← vision + text LLM   │
│                └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Processing Pipeline

```
[Image upload]          [Text input]
      │                      │
      ▼                      │
   OCR Node ◄── retry ──┐   │
      │                  │   │
  [confidence check]     │   │
      │                  │   │
  low → Enhance OCR ─────┘   │
  ok  → Clean Text ◄─────────┘
            │
      Structure Node ◄── retry ──┐
            │                    │
      [completeness check]       │
            │                    │
     gaps → Recover Fields ──────┘
     ok  → Dispatch Analysis
                │
       ┌────────┴────────┐
       │   PARALLEL      │
  Reasoning Node    Risk Node
       │                 │
       └────────┬────────┘
            Merge
              │
        [severity routing]
              │
    high → Critical Alert
    med  → Advisory
    low  → Format Output
              │
            END
```

Each stage produces structured, typed output. If any node fails, the pipeline returns a graceful fallback with error details rather than crashing.

---

## Frontend

The React frontend is built with Vite and communicates with the FastAPI backend via Axios.

**Components:**

| Component | Purpose |
|---|---|
| `UploadForm` | Drag-and-drop image upload with live preview; text input tab for raw paste |
| `ResultDisplay` | Orchestrates the full results layout |
| `MedicationCard` | Displays one medication with its structured fields and AI interpretation |
| `WarningsPanel` | Color-coded risk level badge and warning/alert list |
| `ChatInterface` | Multi-turn chat tied to a session `thread_id`; sends prescription context to ground answers |

**Running the frontend:**
```bash
cd frontend
npm install
npm run dev        # dev server at http://localhost:5173
npm run build      # production build → dist/
```

---

## API Reference

### `POST /api/upload-prescription`

Upload a prescription image. Runs the full pipeline including OCR.

**Request:** `multipart/form-data`
```
file: <image>    # JPEG, PNG, WEBP, HEIC
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
    "medication_interpretations": [...],
    "instruction_summary": "Take Amoxicillin three times daily with food for one week.",
    "abbreviations_decoded": { "TDS": "Three times daily" },
    "disclaimer": "..."
  },
  "risk_assessment": { "risk_level": "low", "flags": [], "warnings": [] },
  "risk_level": "low",
  "warnings": [],
  "critical_alerts": [],
  "ocr_confidence": 0.91,
  "processing_steps": ["[OCR] ...", "[Structuring] ...", "..."],
  "errors": []
}
```

---

### `POST /api/analyze`

Analyze raw prescription text (no image needed — skips OCR).

**Request:**
```json
{ "text": "Amoxicillin 500mg TDS x 7 days\nTake with food\nDr. Smith" }
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
  "prescription_context": {}
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
│   ├── main.py                         # FastAPI app, CORS, router registration
│   ├── config.py                       # Settings from .env (pydantic-settings)
│   ├── api/
│   │   └── routes/
│   │       ├── prescription.py         # /upload-prescription, /analyze
│   │       └── chat.py                 # /chat
│   ├── pipeline/
│   │   ├── state.py                    # PrescriptionState TypedDict
│   │   ├── graph.py                    # Graph assembly and compile
│   │   ├── edges.py                    # Conditional edge functions
│   │   └── nodes/
│   │       ├── ocr_node.py
│   │       ├── ocr_enhancement_node.py
│   │       ├── flag_unreadable_node.py
│   │       ├── cleaning_node.py
│   │       ├── structuring_node.py
│   │       ├── field_recovery_node.py
│   │       ├── dispatch_analysis_node.py
│   │       ├── reasoning_node.py
│   │       ├── risk_node.py
│   │       ├── merge_node.py
│   │       ├── critical_alert_node.py
│   │       ├── advisory_node.py
│   │       └── output_formatter_node.py
│   ├── schemas/
│   │   └── models.py                   # Pydantic models (LLM output + API I/O)
│   └── utils/
│       └── prompts.py                  # LangChain ChatPromptTemplates
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # Root layout
│   │   ├── api.js                      # Axios API calls
│   │   └── components/
│   │       ├── UploadForm.jsx
│   │       ├── ResultDisplay.jsx
│   │       ├── MedicationCard.jsx
│   │       ├── WarningsPanel.jsx
│   │       └── ChatInterface.jsx
│   ├── package.json
│   └── vite.config.js
│
├── .env                                # GEMINI_API_KEY (not committed)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup & Running

### Prerequisites
- Python 3.11+
- Node.js 18+
- A free [Google AI Studio](https://aistudio.google.com) API key

### Backend

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_actual_key

# Start the backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev       # runs at http://localhost:5173
```

The frontend proxies API requests to `http://localhost:8000` — both must be running for full functionality.

---

## Design Decisions

**Multi-agent pipeline over a single prompt** — breaking processing into discrete nodes (OCR → clean → structure → reason → risk) makes each step independently debuggable, retryable, and replaceable without touching the rest of the pipeline.

**Gemini Flash for OCR** — instead of a separate OCR library (Tesseract, etc.), Gemini's vision capability handles both extraction and initial interpretation in one call, with self-reported confidence that drives the retry logic.

**Structured output everywhere** — `.with_structured_output()` with Pydantic models eliminates fragile JSON parsing. Every LLM response is type-validated before entering state.

**Graceful degradation** — every node wraps its LLM call in try/except and returns a safe fallback with descriptive errors/warnings. The API never crashes with a 500; it always returns a usable (if partial) response.

**Parallel analysis** — reasoning and risk detection are independent operations, so they run as concurrent LangGraph branches and merge before severity routing. This cuts latency roughly in half for the analysis phase.

**React + Vite frontend** — Vite's fast HMR and component-based architecture pair well with the structured JSON responses from the backend, making it straightforward to map API fields to UI components.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| OCR on very degraded images | Even with retries, severely damaged or blurry images may not extract reliably |
| No drug database | Medication interpretations come from Gemini's training data, not a validated drug database |
| LLM hallucination risk | Structured output reduces but does not eliminate hallucination |
| No persistent storage | Results are not stored between sessions |
| English-optimised | Non-English prescriptions may have reduced accuracy |
| Free-tier rate limits | Gemini free tier has per-minute request limits |

---

## Future Roadmap

**Reliability**
- OpenFDA API integration for validated medication lookup
- Per-field confidence scoring (not just overall OCR confidence)
- Cross-field consistency validation (e.g., flag impossible dosages)

**Production**
- PostgreSQL persistence for prescriptions, results, and chat history
- User authentication (JWT)
- Containerisation (Docker + docker-compose)
- Deployment: backend on Railway/Render, frontend on Vercel

**Intelligence**
- Multi-language prescription support
- Drug interaction detection across medications on the same prescription
- Pharmacist feedback loop to improve prompts over time

---
