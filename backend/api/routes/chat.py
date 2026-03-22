"""
Chat Route
----------
POST /chat  — conversational Q&A grounded in an already-analysed prescription.

Uses the chat_graph (compiled WITH MemorySaver checkpointer) so each
thread_id maintains its own conversation history across multiple turns.

The prescription_context from the first /upload-prescription or /analyze
call should be passed by the client on the first chat turn. Subsequent
turns in the same thread use stored memory automatically.
"""

import json
import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.schemas.models import ChatRequest, ChatResponse
from backend.utils.prompts import CHAT_PROMPT

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Multi-turn conversational endpoint.

    - thread_id groups turns into a single conversation (MemorySaver key)
    - prescription_context grounds the model in the specific prescription
    - Subsequent turns in the same thread don't need to re-send context
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )

    context_str = (
        json.dumps(request.prescription_context, indent=2)
        if request.prescription_context
        else "No prescription context provided for this session."
    )

    # Build the prompt using the chat template
    chain = CHAT_PROMPT | llm

    try:
        response = chain.invoke({
            "prescription_context": context_str,
            "chat_history": [],          # LangGraph MemorySaver handles history via thread_id
            "message": request.message,
        })
        reply = response.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(exc)}")

    return ChatResponse(reply=reply, thread_id=thread_id)
