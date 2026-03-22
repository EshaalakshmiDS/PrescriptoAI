"""
PrescriptionState — the single source of truth flowing through the entire LangGraph.

Design notes:
- Fields shared across parallel branches (warnings, processing_steps, errors)
  use Annotated[list, operator.add] reducers so concurrent nodes can safely
  append without overwriting each other.
- Optional fields start as None; each node populates its own slice.
- Retry counters (ocr_retry_count, structuring_retry_count) are plain ints —
  only one node writes them at a time, so no reducer needed.
"""

import operator
from typing import Annotated, Optional, TypedDict


class PrescriptionState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    image_bytes: Optional[bytes]        # set by /upload-prescription
    raw_text_input: Optional[str]       # set by /analyze (skips OCR)

    # ── OCR stage ─────────────────────────────────────────────────────────────
    raw_text: Optional[str]
    ocr_confidence: Optional[float]     # 0.0–1.0 self-reported by Gemini
    ocr_retry_count: int                # loop guard: max = settings.max_ocr_retries
    low_confidence_areas: Optional[list[str]]   # regions Gemini flagged

    # ── Cleaning stage ────────────────────────────────────────────────────────
    cleaned_text: Optional[str]
    detected_language: Optional[str]
    medical_tokens: Optional[list[str]] # recognised medical abbreviations / keywords

    # ── Structuring stage ─────────────────────────────────────────────────────
    structured_data: Optional[dict]     # medications[], notes, doctor_instructions
    missing_fields: Optional[list[str]] # fields Gemini could not extract
    structuring_retry_count: int        # loop guard

    # ── Parallel analysis ─────────────────────────────────────────────────────
    interpretation: Optional[dict]      # plain-English explanation (reasoning node)
    risk_assessment: Optional[dict]     # flagged issues (risk node)
    risk_level: Optional[str]           # "high" | "medium" | "low"

    # ── Severity-specific ─────────────────────────────────────────────────────
    critical_alerts: Optional[list[str]]

    # ── Final output ──────────────────────────────────────────────────────────
    final_output: Optional[dict]

    # ── Accumulating fields (reducer = list concatenation) ────────────────────
    # Safe for parallel branches: both reason + detect_risk can append
    warnings: Annotated[list[str], operator.add]
    processing_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
