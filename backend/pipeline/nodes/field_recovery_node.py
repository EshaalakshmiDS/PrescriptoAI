"""
Field Recovery Node — with graceful error handling.
"""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import StructuredPrescription
from backend.utils.prompts import RECOVERY_PROMPT

logger = logging.getLogger(__name__)


def field_recovery_node(state: PrescriptionState) -> dict:
    cleaned_text = state.get("cleaned_text") or ""
    missing_fields = state.get("missing_fields") or []
    partial_structured = state.get("structured_data") or {}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )
        structured_llm = llm.with_structured_output(StructuredPrescription)

        prompt_value = RECOVERY_PROMPT.invoke({
            "missing_fields": ", ".join(missing_fields),
            "cleaned_text": cleaned_text,
            "partial_structured": json.dumps(partial_structured, indent=2),
        })
        result: StructuredPrescription = structured_llm.invoke(prompt_value)

        recovered = [f for f in missing_fields if f not in (result.missing_fields or [])]

        return {
            "structured_data": result.model_dump(),
            "missing_fields": result.missing_fields,
            "processing_steps": [
                f"[Field Recovery] Recovered {len(recovered)} field(s): "
                f"{recovered or 'none'}. Still missing: {result.missing_fields or 'none'}"
            ],
        }

    except Exception as exc:
        logger.error("[Field Recovery] Failed: %s", exc)
        return {
            "errors": [f"Field recovery failed: {str(exc)}"],
            "warnings": [f"Could not recover missing fields: {', '.join(missing_fields)}"],
            "processing_steps": ["[Field Recovery] Failed — proceeding with partial data."],
        }
