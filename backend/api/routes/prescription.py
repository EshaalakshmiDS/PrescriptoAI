"""
Prescription Routes
-------------------
POST /upload-prescription  — accepts an image, runs the full pipeline
POST /analyze              — accepts raw text, skips OCR, runs steps 2-5
"""

import logging
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

from backend.pipeline.graph import prescription_graph
from backend.pipeline.state import PrescriptionState
from backend.schemas.models import AnalyzeTextRequest, PrescriptionResponse

router = APIRouter(prefix="/api", tags=["prescription"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}


def _build_initial_state(**kwargs) -> PrescriptionState:
    """Returns a fully initialised state dict with safe defaults, overridden by kwargs."""
    defaults: dict = {
        "image_bytes": None,
        "raw_text_input": None,
        "raw_text": None,
        "ocr_confidence": None,
        "ocr_retry_count": 0,
        "low_confidence_areas": [],
        "cleaned_text": None,
        "detected_language": None,
        "medical_tokens": [],
        "structured_data": None,
        "missing_fields": [],
        "structuring_retry_count": 0,
        "interpretation": None,
        "risk_assessment": None,
        "risk_level": None,
        "critical_alerts": [],
        "final_output": None,
        "warnings": [],
        "processing_steps": [],
        "errors": [],
    }
    defaults.update(kwargs)
    return PrescriptionState(**defaults)


def _state_to_response(state: PrescriptionState) -> PrescriptionResponse:
    final = state.get("final_output") or {}
    return PrescriptionResponse(
        structured_data=state.get("structured_data"),
        interpretation=state.get("interpretation"),
        risk_assessment=state.get("risk_assessment"),
        risk_level=state.get("risk_level"),
        warnings=state.get("warnings", []),
        critical_alerts=state.get("critical_alerts", []),
        processing_steps=state.get("processing_steps", []),
        errors=state.get("errors", []),
        ocr_confidence=state.get("ocr_confidence"),
        final_output=final,
    )


@router.post("/upload-prescription", response_model=PrescriptionResponse)
async def upload_prescription(file: UploadFile = File(...)):
    """
    Upload a prescription image (JPEG, PNG, WEBP, HEIC).
    Runs the full LangGraph pipeline:
      OCR → Clean → Structure → [Reason ‖ Risk] → Merge → Alert/Advisory → Format
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    initial_state = _build_initial_state(image_bytes=image_bytes)

    try:
        result = prescription_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline error:\n%s", traceback.format_exc())
        return PrescriptionResponse(
            errors=[f"Pipeline encountered an unexpected error: {str(exc)}"],
            warnings=["Processing could not be completed. Please try again or use /analyze with plain text."],
            processing_steps=["[Pipeline] Terminated due to unexpected error."],
        )

    return _state_to_response(result)


@router.post("/analyze", response_model=PrescriptionResponse)
async def analyze_text(request: AnalyzeTextRequest):
    """
    Analyze raw prescription text (OCR already done or text typed manually).
    Skips OCR and enhancement nodes, starts from the cleaning step.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    initial_state = _build_initial_state(raw_text_input=request.text.strip())

    try:
        result = prescription_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline error:\n%s", traceback.format_exc())
        return PrescriptionResponse(
            errors=[f"Pipeline encountered an unexpected error: {str(exc)}"],
            warnings=["Processing could not be completed. Please try again."],
            processing_steps=["[Pipeline] Terminated due to unexpected error."],
        )

    return _state_to_response(result)
