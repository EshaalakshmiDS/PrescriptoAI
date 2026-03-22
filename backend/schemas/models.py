"""
Pydantic models used in two places:
1. LLM structured output (.with_structured_output) — each node gets a typed response
2. FastAPI request/response schemas — what the API accepts and returns
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── LLM output models (one per node) ─────────────────────────────────────────

class OCROutput(BaseModel):
    extracted_text: str = Field(description="All text extracted from the prescription image")
    confidence_score: float = Field(default=0.8, description="Confidence in extraction accuracy, 0.0 to 1.0")
    low_confidence_areas: list[str] = Field(default_factory=list, description="Areas that were hard to read")
    observations: str = Field(default="", description="Notes about image quality or legibility issues")


class MedicationEntry(BaseModel):
    name: str = Field(description="Medication name as written")
    dosage: Optional[str] = Field(None, description="Dosage amount and unit e.g. 500mg")
    frequency: Optional[str] = Field(None, description="How often e.g. twice daily, every 8 hours")
    duration: Optional[str] = Field(None, description="How long e.g. 7 days, 2 weeks")
    route: Optional[str] = Field(None, description="Route of administration e.g. oral, topical")
    instructions: Optional[str] = Field(None, description="Special instructions e.g. take with food")


class StructuredPrescription(BaseModel):
    patient_name: Optional[str] = Field(None, description="Patient name if visible")
    patient_age: Optional[str] = Field(None, description="Patient age if visible")
    doctor_name: Optional[str] = Field(None, description="Doctor name if visible")
    doctor_credentials: Optional[str] = Field(None, description="Doctor credentials if visible")
    date: Optional[str] = Field(None, description="Prescription date if visible")
    medications: list[MedicationEntry] = Field(default_factory=list)
    notes: Optional[str] = Field(None, description="General notes on the prescription")
    doctor_instructions: Optional[str] = Field(None, description="Doctor's general instructions")
    missing_fields: list[str] = Field(default_factory=list, description="Critical fields that could not be extracted")


class MedicationInterpretation(BaseModel):
    medication_name: str = Field(default="Unknown")
    common_use: str = Field(default="", description="What this medication is commonly prescribed for — educational only")
    dosage_explanation: str = Field(default="", description="Plain-English explanation of the dosing schedule")
    important_notes: Optional[str] = Field(None, description="Common considerations for this medication class")


class ReasoningOutput(BaseModel):
    medication_interpretations: list[MedicationInterpretation] = Field(default_factory=list)
    instruction_summary: str = Field(default="", description="Plain-English summary of all doctor instructions")
    abbreviations_decoded: dict[str, str] = Field(default_factory=dict, description="Medical abbreviations found and their meanings")
    disclaimer: str = Field(default="This is for informational purposes only and does not constitute medical advice.")


class RiskFlag(BaseModel):
    field: str = Field(default="unknown", description="Which field or medication this flag refers to")
    issue: str = Field(default="", description="Description of the issue")
    severity: str = Field(default="low", description="'high', 'medium', or 'low'")


class RiskOutput(BaseModel):
    risk_level: str = Field(default="low", description="Overall risk level: 'high', 'medium', or 'low'")
    flags: list[RiskFlag] = Field(default_factory=list)
    missing_critical_info: list[str] = Field(default_factory=list, description="Critical information that is absent")
    ambiguous_instructions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="Human-readable warning messages")


# ── FastAPI request / response schemas ───────────────────────────────────────

class AnalyzeTextRequest(BaseModel):
    text: str = Field(description="Raw prescription text to analyze")


class PrescriptionResponse(BaseModel):
    structured_data: Optional[dict] = None
    interpretation: Optional[dict] = None
    risk_assessment: Optional[dict] = None
    risk_level: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    critical_alerts: list[str] = Field(default_factory=list)
    processing_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    ocr_confidence: Optional[float] = None
    final_output: Optional[dict] = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(description="Unique session ID to maintain conversation memory")
    prescription_context: Optional[dict] = Field(None, description="Structured prescription data to ground the conversation")


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
