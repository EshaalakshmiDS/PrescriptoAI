"""
OCR Enhancement Node — with graceful error handling.
"""

import base64
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import OCROutput
from backend.utils.prompts import OCR_ENHANCE_SYSTEM

logger = logging.getLogger(__name__)


def ocr_enhancement_node(state: PrescriptionState) -> dict:
    image_bytes = state["image_bytes"]
    retry_count = state["ocr_retry_count"] + 1
    prev_confidence = state.get("ocr_confidence") or 0.0
    prev_text = state.get("raw_text") or ""
    low_confidence_areas = state.get("low_confidence_areas") or []

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(OCROutput)

        enhance_text = (
            f"This is attempt #{retry_count} to extract text from this prescription.\n\n"
            f"Previous extraction (low confidence: {prev_confidence:.0%}):\n{prev_text}\n\n"
            f"Flagged difficult areas: {', '.join(low_confidence_areas) or 'none specified'}\n\n"
            "Re-examine the image carefully. Focus on:\n"
            "1. Medical abbreviations (od, bd, tds, qid, prn, sos, ac, pc, hs)\n"
            "2. Drug names that may look like handwriting\n"
            "3. Numbers (dosages, dates) that may be ambiguous\n"
            "4. Any text you missed previously\n\n"
            "Return the same JSON format with updated extracted_text and confidence_score."
        )

        messages = [
            SystemMessage(content=OCR_ENHANCE_SYSTEM),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": enhance_text},
            ]),
        ]

        result: OCROutput = structured_llm.invoke(messages)

        return {
            "raw_text": result.extracted_text,
            "ocr_confidence": result.confidence_score,
            "low_confidence_areas": result.low_confidence_areas,
            "ocr_retry_count": retry_count,
            "processing_steps": [
                f"[OCR Enhancement #{retry_count}] Retry — new confidence: {result.confidence_score:.0%}"
            ],
        }

    except Exception as exc:
        logger.error("[OCR Enhancement #%d] Failed: %s", retry_count, exc)
        return {
            "ocr_retry_count": retry_count,
            "errors": [f"OCR enhancement #{retry_count} failed: {str(exc)}"],
            "processing_steps": [f"[OCR Enhancement #{retry_count}] Failed — keeping previous extraction."],
        }
