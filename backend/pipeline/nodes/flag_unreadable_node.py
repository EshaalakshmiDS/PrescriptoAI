"""
Flag Unreadable Node
--------------------
Terminal dead-end node. Reached when OCR confidence is irrecoverably low
after exhausting all retries. Marks the prescription as unprocessable
and sets a human-readable error in state.

State reads  : ocr_retry_count, ocr_confidence
State writes : errors, warnings, processing_steps, final_output
"""

from backend.pipeline.state import PrescriptionState


def flag_unreadable_node(state: PrescriptionState) -> dict:
    confidence = state.get("ocr_confidence") or 0.0
    retries = state.get("ocr_retry_count", 0)

    message = (
        f"Prescription image could not be reliably read after {retries} attempt(s). "
        f"Final OCR confidence: {confidence:.0%}. "
        "Please upload a clearer image or enter the prescription text manually via /analyze."
    )

    return {
        "errors": [message],
        "warnings": ["Image quality too low for automated processing."],
        "processing_steps": ["[Flag Unreadable] Pipeline terminated — image irrecoverable."],
        "final_output": {
            "status": "failed",
            "reason": "unreadable_image",
            "message": message,
        },
    }
