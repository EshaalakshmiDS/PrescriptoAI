"""
OCR Node — with graceful error handling.
On any failure (API error, parse error), returns what it can and adds to errors[].
"""

import base64
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import OCROutput
from backend.utils.prompts import OCR_SYSTEM, OCR_USER

logger = logging.getLogger(__name__)


def ocr_node(state: PrescriptionState) -> dict:
    image_bytes = state["image_bytes"]

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(OCROutput)

        messages = [
            SystemMessage(content=OCR_SYSTEM),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": OCR_USER},
            ]),
        ]

        result: OCROutput = structured_llm.invoke(messages)

        return {
            "raw_text": result.extracted_text,
            "ocr_confidence": result.confidence_score,
            "low_confidence_areas": result.low_confidence_areas,
            "processing_steps": [
                f"[OCR] Extracted text — confidence: {result.confidence_score:.0%}"
            ],
        }

    except Exception as exc:
        logger.error("[OCR] Failed: %s", exc)
        return {
            "raw_text": "",
            "ocr_confidence": 0.0,
            "low_confidence_areas": ["entire image"],
            "errors": [f"OCR failed: {str(exc)}"],
            "processing_steps": ["[OCR] Failed — could not extract text from image."],
        }
