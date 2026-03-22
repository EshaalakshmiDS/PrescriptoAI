"""
Risk Detection Node [parallel] — with graceful error handling.
"""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import RiskOutput
from backend.utils.prompts import RISK_PROMPT

logger = logging.getLogger(__name__)


def risk_node(state: PrescriptionState) -> dict:
    structured_data = state.get("structured_data") or {}
    low_confidence_areas = state.get("low_confidence_areas") or []

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(RiskOutput)

        prompt_value = RISK_PROMPT.invoke({
            "structured_data": json.dumps(structured_data, indent=2),
            "low_confidence_areas": ", ".join(low_confidence_areas) or "none",
        })
        result: RiskOutput = structured_llm.invoke(prompt_value)

        return {
            "risk_assessment": result.model_dump(),
            "risk_level": result.risk_level,
            "warnings": result.warnings,
            "processing_steps": [
                f"[Risk Detection] Level: {result.risk_level.upper()} — "
                f"{len(result.flags)} flag(s), {len(result.warnings)} warning(s)."
            ],
        }

    except Exception as exc:
        logger.error("[Risk Detection] Failed: %s", exc)
        return {
            "risk_assessment": {
                "risk_level": "medium",
                "flags": [],
                "missing_critical_info": [],
                "ambiguous_instructions": [],
                "warnings": ["Risk assessment could not be completed. Please verify prescription manually."],
            },
            "risk_level": "medium",
            "warnings": ["Risk assessment unavailable — manual verification recommended."],
            "errors": [f"Risk detection failed: {str(exc)}"],
            "processing_steps": ["[Risk Detection] Failed — defaulting to medium risk. Verify manually."],
        }
