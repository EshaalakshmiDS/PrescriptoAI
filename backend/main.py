"""
PrescriptoAI — FastAPI Application Entry Point
"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes.chat import router as chat_router
from backend.api.routes.prescription import router as prescription_router
from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Startup diagnostic — always visible in uvicorn output
logger.info("=== PrescriptoAI starting ===")
logger.info("Model  : %s", settings.gemini_model)
logger.info("API key: %s...", settings.gemini_api_key[:12])

app = FastAPI(
    title="PrescriptoAI",
    description=(
        "AI-powered prescription analysis using OCR + multi-agent LangGraph pipeline. "
        "Transforms handwritten or printed prescriptions into structured, interpreted, "
        "and risk-assessed outputs."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("Unhandled exception:\n%s", tb)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": tb},
    )

app.include_router(prescription_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "PrescriptoAI"}
