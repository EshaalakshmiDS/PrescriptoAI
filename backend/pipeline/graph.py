"""
LangGraph Graph Assembly
------------------------
Wires all nodes and conditional edges into a compiled StateGraph.

Graph topology at a glance:

  start_router ──(conditional)──► ocr ◄──────────────────────┐
                │                  │                           │
                │         (conditional: route_ocr_quality)    │
                │                  ├──► enhance_ocr ──────────┘  [CYCLE]
                │                  ├──► flag_unreadable ──► END
                │                  └──► clean_text
                │                            │
                └────────────────────────────┘
                                             │
                                         structure ◄──────────────────┐
                                             │                         │
                                  (conditional: route_completeness)    │
                                             ├──► recover_fields ──────┘  [CYCLE]
                                             └──► dispatch_analysis
                                                       │
                                            ┌──────────┴──────────┐
                                            ▼                      ▼
                                         reason            detect_risk
                                            │                      │
                                            └──────────┬───────────┘
                                                       ▼
                                                  merge_analysis
                                                       │
                                          (conditional: route_severity)
                                                       ├──► critical_alert ──┐
                                                       ├──► advisory ────────┤
                                                       └──► format_output ◄──┘
                                                                  │
                                                                 END
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.pipeline.edges import (
    route_completeness,
    route_ocr_quality,
    route_severity,
    route_start,
)
from backend.pipeline.nodes.advisory_node import advisory_node
from backend.pipeline.nodes.cleaning_node import cleaning_node
from backend.pipeline.nodes.critical_alert_node import critical_alert_node
from backend.pipeline.nodes.dispatch_analysis_node import dispatch_analysis_node
from backend.pipeline.nodes.field_recovery_node import field_recovery_node
from backend.pipeline.nodes.flag_unreadable_node import flag_unreadable_node
from backend.pipeline.nodes.merge_node import merge_node
from backend.pipeline.nodes.ocr_enhancement_node import ocr_enhancement_node
from backend.pipeline.nodes.ocr_node import ocr_node
from backend.pipeline.nodes.output_formatter_node import output_formatter_node
from backend.pipeline.nodes.reasoning_node import reasoning_node
from backend.pipeline.nodes.risk_node import risk_node
from backend.pipeline.nodes.structuring_node import structuring_node
from backend.pipeline.state import PrescriptionState


def build_graph(with_memory: bool = False):
    """
    Build and compile the prescription processing graph.

    Args:
        with_memory: If True, attach a MemorySaver checkpointer.
                     Required for the /chat endpoint (thread-level memory).
                     Not needed for stateless /upload-prescription calls.
    """
    builder = StateGraph(PrescriptionState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("ocr", ocr_node)
    builder.add_node("enhance_ocr", ocr_enhancement_node)
    builder.add_node("flag_unreadable", flag_unreadable_node)
    builder.add_node("clean_text", cleaning_node)
    builder.add_node("structure", structuring_node)
    builder.add_node("recover_fields", field_recovery_node)
    builder.add_node("dispatch_analysis", dispatch_analysis_node)
    builder.add_node("reason", reasoning_node)
    builder.add_node("detect_risk", risk_node)
    builder.add_node("merge_analysis", merge_node)
    builder.add_node("critical_alert", critical_alert_node)
    builder.add_node("advisory", advisory_node)
    builder.add_node("format_output", output_formatter_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    # START is a virtual node in LangGraph 1.x — conditional edges from it
    # replace the deprecated set_conditional_entry_point().
    builder.add_conditional_edges(
        START,
        route_start,
        {"ocr": "ocr", "clean_text": "clean_text"},
    )

    # ── OCR retry cycle ───────────────────────────────────────────────────────
    # ocr → [clean_text | enhance_ocr | flag_unreadable]
    builder.add_conditional_edges(
        "ocr",
        route_ocr_quality,
        {
            "clean_text": "clean_text",
            "enhance_ocr": "enhance_ocr",
            "flag_unreadable": "flag_unreadable",
        },
    )
    # enhance_ocr feeds back into ocr, completing the retry cycle
    builder.add_edge("enhance_ocr", "ocr")
    builder.add_edge("flag_unreadable", END)

    # ── Main pipeline ─────────────────────────────────────────────────────────
    builder.add_edge("clean_text", "structure")

    # ── Structuring + field recovery cycle ────────────────────────────────────
    builder.add_conditional_edges(
        "structure",
        route_completeness,
        {
            "recover_fields": "recover_fields",
            "dispatch_analysis": "dispatch_analysis",
        },
    )
    # recover_fields feeds back into structure, completing the recovery cycle
    builder.add_edge("recover_fields", "structure")

    # ── Parallel fan-out ──────────────────────────────────────────────────────
    # LangGraph executes both edges from dispatch_analysis concurrently
    builder.add_edge("dispatch_analysis", "reason")
    builder.add_edge("dispatch_analysis", "detect_risk")

    # ── Fan-in ────────────────────────────────────────────────────────────────
    # merge_analysis waits for BOTH reason and detect_risk to finish
    builder.add_edge("reason", "merge_analysis")
    builder.add_edge("detect_risk", "merge_analysis")

    # ── Severity routing ──────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "merge_analysis",
        route_severity,
        {
            "critical_alert": "critical_alert",
            "advisory": "advisory",
            "format_output": "format_output",
        },
    )

    # Both alert branches converge on format_output
    builder.add_edge("critical_alert", "format_output")
    builder.add_edge("advisory", "format_output")
    builder.add_edge("format_output", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    checkpointer = MemorySaver() if with_memory else None
    return builder.compile(checkpointer=checkpointer)


# Module-level singletons — built once at startup
prescription_graph = build_graph(with_memory=False)
chat_graph = build_graph(with_memory=True)
