"""
Cleaning Node
-------------
Normalises raw OCR output before it reaches the LLM structuring step.
Pure Python — no LLM call here keeps latency low and cost zero.

Tasks:
  - Strip OCR artefacts (|, _, excessive whitespace, stray punctuation)
  - Normalise common medical abbreviations to consistent casing
  - Detect likely language (heuristic)
  - Identify medical tokens present in the text

State reads  : raw_text
State writes : cleaned_text, detected_language, medical_tokens, processing_steps
"""

import re

from backend.pipeline.state import PrescriptionState

# Common medical abbreviations worth preserving / normalising
MEDICAL_ABBREVIATIONS = {
    r"\bod\b": "OD",   # once daily
    r"\bbd\b": "BD",   # twice daily
    r"\btds\b": "TDS", # three times daily
    r"\bqid\b": "QID", # four times daily
    r"\bprn\b": "PRN", # as needed
    r"\bsos\b": "SOS", # if needed (emergency)
    r"\bac\b": "AC",   # before meals
    r"\bpc\b": "PC",   # after meals
    r"\bhs\b": "HS",   # at bedtime
    r"\bstat\b": "STAT",
    r"\bim\b": "IM",   # intramuscular
    r"\biv\b": "IV",   # intravenous
    r"\bpo\b": "PO",   # by mouth
    r"\bsc\b": "SC",   # subcutaneous
}

KNOWN_MEDICAL_TOKENS = [
    "mg", "ml", "mcg", "units", "tablet", "tab", "capsule", "cap",
    "syrup", "injection", "cream", "drops", "ointment", "patch",
    "OD", "BD", "TDS", "QID", "PRN", "SOS", "AC", "PC", "HS",
    "STAT", "IM", "IV", "PO", "SC",
]


def _normalise_abbreviations(text: str) -> str:
    for pattern, replacement in MEDICAL_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _strip_artefacts(text: str) -> str:
    # Remove common OCR noise characters
    text = re.sub(r"[|\\]", " ", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely punctuation or symbols
    lines = text.splitlines()
    lines = [l for l in lines if re.search(r"[a-zA-Z0-9]", l)]
    return "\n".join(lines).strip()


def _detect_language(text: str) -> str:
    # Simple heuristic: if >80% ASCII alpha, assume English
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return "unknown"
    ascii_ratio = sum(1 for c in alpha_chars if ord(c) < 128) / len(alpha_chars)
    return "en" if ascii_ratio > 0.8 else "non-english"


def _find_medical_tokens(text: str) -> list[str]:
    found = []
    for token in KNOWN_MEDICAL_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE):
            found.append(token)
    return list(set(found))


def cleaning_node(state: PrescriptionState) -> dict:
    raw = state.get("raw_text") or state.get("raw_text_input") or ""

    cleaned = _strip_artefacts(raw)
    cleaned = _normalise_abbreviations(cleaned)
    language = _detect_language(cleaned)
    tokens = _find_medical_tokens(cleaned)

    return {
        "cleaned_text": cleaned,
        "detected_language": language,
        "medical_tokens": tokens,
        "processing_steps": [
            f"[Cleaning] Normalised text — language: {language}, "
            f"medical tokens found: {len(tokens)}"
        ],
    }
