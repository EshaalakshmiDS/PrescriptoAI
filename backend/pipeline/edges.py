"""
Conditional Edge Functions
--------------------------
Each function inspects the current state and returns a string key
that LangGraph uses to select the next node.

These are the core of the graph's decision-making logic:

  route_start          — image upload vs. plain-text input
  route_ocr_quality    — OCR confidence gate + retry loop
  route_completeness   — structuring completeness gate + recovery loop
  route_severity       — final risk-level routing to alert / advisory / pass
"""

from backend.config import settings
from backend.pipeline.state import PrescriptionState


def route_start(state: PrescriptionState) -> str:
    """
    If raw_text_input is provided (POST /analyze), skip OCR entirely
    and jump straight to the cleaning node.
    Otherwise begin with OCR.
    """
    if state.get("raw_text_input"):
        return "clean_text"
    return "ocr"


def route_ocr_quality(state: PrescriptionState) -> str:
    """
    Decision tree after each OCR attempt:

    confidence >= threshold          → proceed to cleaning
    confidence <  threshold
      AND retry_count < max_retries  → enhance and retry  (CYCLE)
    confidence <  threshold
      AND retry budget exhausted     → flag as unreadable  (DEAD END)
    """
    confidence = state.get("ocr_confidence") or 0.0
    retry_count = state.get("ocr_retry_count", 0)

    if confidence >= settings.ocr_confidence_threshold:
        return "clean_text"

    if retry_count < settings.max_ocr_retries:
        return "enhance_ocr"

    return "flag_unreadable"


def route_completeness(state: PrescriptionState) -> str:
    """
    Decision tree after structuring:

    No missing critical fields                   → fan out to parallel analysis
    Missing fields AND retry budget remains      → recover and re-structure (CYCLE)
    Missing fields BUT retry budget exhausted    → proceed anyway, warnings will capture it
    """
    missing_fields = state.get("missing_fields") or []
    retry_count = state.get("structuring_retry_count", 0)

    if not missing_fields:
        return "dispatch_analysis"

    if retry_count <= settings.max_structuring_retries:
        return "recover_fields"

    # Retry budget exhausted — proceed with what we have
    return "dispatch_analysis"


def route_severity(state: PrescriptionState) -> str:
    """
    After parallel analysis merges, route based on overall risk level:

    high   → critical_alert node (urgent escalation)
    medium → advisory node (cautionary warnings)
    low    → skip straight to output formatter
    """
    risk_level = (state.get("risk_level") or "low").lower()

    if risk_level == "high":
        return "critical_alert"
    if risk_level == "medium":
        return "advisory"
    return "format_output"
