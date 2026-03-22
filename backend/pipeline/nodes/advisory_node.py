"""
Advisory Node
-------------
Reached when risk_level == "medium".
Converts medium-severity flags into advisory warnings — less urgent than
critical alerts but still requiring attention before dispensing.

State reads  : risk_assessment
State writes : warnings, processing_steps
"""

from backend.pipeline.state import PrescriptionState


def advisory_node(state: PrescriptionState) -> dict:
    risk_assessment = state.get("risk_assessment") or {}
    flags = risk_assessment.get("flags", [])
    ambiguous = risk_assessment.get("ambiguous_instructions", [])

    advisory_warnings: list[str] = []

    for flag in flags:
        if flag.get("severity") in ("medium", "high"):
            advisory_warnings.append(
                f"Advisory — {flag.get('field', 'unknown')}: {flag.get('issue', '')}"
            )

    for item in ambiguous:
        advisory_warnings.append(f"Advisory — Ambiguous instruction: {item}")

    if not advisory_warnings:
        advisory_warnings.append(
            "Advisory — Some fields may need pharmacist clarification before dispensing."
        )

    return {
        "warnings": advisory_warnings,
        "processing_steps": [
            f"[Advisory] {len(advisory_warnings)} advisory warning(s) added."
        ],
    }
