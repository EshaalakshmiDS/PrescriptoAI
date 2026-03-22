"""
Critical Alert Node
-------------------
Reached when risk_level == "high".
Escalates all risk flags into prominent critical_alerts and ensures
warnings list is populated with urgent, actionable messages.

State reads  : risk_assessment, warnings
State writes : critical_alerts, warnings, processing_steps
"""

from backend.pipeline.state import PrescriptionState


def critical_alert_node(state: PrescriptionState) -> dict:
    risk_assessment = state.get("risk_assessment") or {}
    flags = risk_assessment.get("flags", [])
    missing_critical = risk_assessment.get("missing_critical_info", [])

    critical_alerts: list[str] = []

    for flag in flags:
        if flag.get("severity") == "high":
            critical_alerts.append(
                f"CRITICAL — {flag.get('field', 'unknown field')}: {flag.get('issue', '')}"
            )

    for item in missing_critical:
        critical_alerts.append(f"CRITICAL — Missing required information: {item}")

    if not critical_alerts:
        critical_alerts.append(
            "CRITICAL — High risk detected. Please verify this prescription with your pharmacist or doctor before use."
        )

    urgent_warnings = [
        "This prescription has been flagged as HIGH RISK.",
        "Do NOT dispense or consume medication without pharmacist or doctor verification.",
    ]

    return {
        "critical_alerts": critical_alerts,
        "warnings": urgent_warnings,
        "processing_steps": [
            f"[Critical Alert] {len(critical_alerts)} critical alert(s) raised."
        ],
    }
