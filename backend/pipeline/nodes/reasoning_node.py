"""
Reasoning Node [parallel] — with graceful error handling.
"""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import ReasoningOutput
from backend.utils.prompts import REASONING_PROMPT

logger = logging.getLogger(__name__)


def reasoning_node(state: PrescriptionState) -> dict:
    structured_data = state.get("structured_data") or {}

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
        structured_llm = llm.with_structured_output(ReasoningOutput)

        prompt_value = REASONING_PROMPT.invoke({
            "structured_data": json.dumps(structured_data, indent=2)
        })
        result: ReasoningOutput = structured_llm.invoke(prompt_value)

        return {
            "interpretation": result.model_dump(),
            "processing_steps": [
                f"[Reasoning] Interpreted {len(result.medication_interpretations)} medication(s) — "
                f"{len(result.abbreviations_decoded)} abbreviation(s) decoded."
            ],
        }

    except Exception as exc:
        logger.error("[Reasoning] Failed: %s", exc)
        return {
            "interpretation": {
                "medication_interpretations": [],
                "instruction_summary": "Interpretation unavailable due to a processing error.",
                "abbreviations_decoded": {},
                "disclaimer": "This is for informational purposes only and does not constitute medical advice.",
            },
            "errors": [f"Reasoning failed: {str(exc)}"],
            "processing_steps": ["[Reasoning] Failed — interpretation unavailable."],
        }
