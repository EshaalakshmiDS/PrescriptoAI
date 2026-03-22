"""
Dispatch Analysis Node
-----------------------
A lightweight fan-out coordinator. After structuring is complete, this
no-op node acts as the single source for the two parallel branches:
  - reasoning_node
  - risk_node

LangGraph will execute both branches concurrently once this node completes.
Having an explicit dispatch node (vs. two edges from structuring_node) keeps
the conditional routing in structuring_node clean and single-purpose.

State reads  : (nothing new)
State writes : processing_steps
"""

from backend.pipeline.state import PrescriptionState


def dispatch_analysis_node(state: PrescriptionState) -> dict:
    return {
        "processing_steps": [
            "[Dispatch] Fanning out to Reasoning + Risk Detection in parallel."
        ]
    }
