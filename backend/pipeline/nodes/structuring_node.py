"""
Structuring Node — with graceful error handling.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import StructuredPrescription
from backend.utils.prompts import STRUCTURING_PROMPT

logger = logging.getLogger(__name__)


def structuring_node(state: PrescriptionState) -> dict:
    cleaned_text = state.get("cleaned_text") or ""
    retry_count = state.get("structuring_retry_count", 0) + 1

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(StructuredPrescription)

        prompt_value = STRUCTURING_PROMPT.invoke({"cleaned_text": cleaned_text})
        result: StructuredPrescription = structured_llm.invoke(prompt_value)

        missing = result.missing_fields
        med_count = len(result.medications)

        return {
            "structured_data": result.model_dump(),
            "missing_fields": missing,
            "structuring_retry_count": retry_count,
            "processing_steps": [
                f"[Structuring #{retry_count}] Extracted {med_count} medication(s) — "
                f"missing fields: {missing or 'none'}"
            ],
        }

    except Exception as exc:
        logger.error("[Structuring #%d] Failed: %s", retry_count, exc)
        return {
            "structured_data": {"medications": [], "missing_fields": ["all fields — structuring failed"]},
            "missing_fields": ["all fields — structuring failed"],
            "structuring_retry_count": retry_count,
            "errors": [f"Structuring failed: {str(exc)}"],
            "warnings": ["Could not extract structured data from prescription text. Please verify manually."],
            "processing_steps": [f"[Structuring #{retry_count}] Failed — returning empty structure."],
        }
