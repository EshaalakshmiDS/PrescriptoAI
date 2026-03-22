"""
Merge Analysis Node
-------------------
Fan-in point after the parallel reasoning + risk branches.
LangGraph waits for BOTH branches to complete before executing this node.

Responsibilities:
  - Confirm risk_level is set (fallback to "low" if risk_node somehow missed it)
  - Enrich warnings with any missing-field warnings from structuring
  - Log the merge step for the audit trail

State reads  : risk_level, missing_fields, warnings
State writes : risk_level (normalised), warnings (enriched), processing_steps
"""

from backend.pipeline.state import PrescriptionState


def merge_node(state: PrescriptionState) -> dict:
    risk_level = (state.get("risk_level") or "low").lower()
    missing_fields = state.get("missing_fields") or []

    extra_warnings: list[str] = []
    for field in missing_fields:
        extra_warnings.append(f"Could not extract field: {field}")

    return {
        "risk_level": risk_level,
        "warnings": extra_warnings,
        "processing_steps": [
            f"[Merge] Parallel branches joined — risk level confirmed: {risk_level.upper()}."
        ],
    }
