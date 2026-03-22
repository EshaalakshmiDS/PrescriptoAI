"""
Output Formatter Node
---------------------
Final node in the graph. Assembles all state slices into a single,
clean final_output dict that the API route returns to the client.

State reads  : structured_data, interpretation, risk_assessment, risk_level,
               warnings, critical_alerts, ocr_confidence, processing_steps, errors
State writes : final_output, processing_steps
"""

from backend.pipeline.state import PrescriptionState


def output_formatter_node(state: PrescriptionState) -> dict:
    final_output = {
        "status": "success",
        "ocr_confidence": state.get("ocr_confidence"),
        "structured_data": state.get("structured_data"),
        "interpretation": state.get("interpretation"),
        "risk_assessment": state.get("risk_assessment"),
        "risk_level": state.get("risk_level", "low"),
        "warnings": state.get("warnings", []),
        "critical_alerts": state.get("critical_alerts", []),
        "processing_steps": state.get("processing_steps", []),
        "errors": state.get("errors", []),
    }

    return {
        "final_output": final_output,
        "processing_steps": ["[Output Formatter] Pipeline complete. Final output assembled."],
    }
