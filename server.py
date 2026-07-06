#!/usr/bin/env python3
"""
Qwen OpenAI-Compatible API Server
Wraps chat.qwen.ai with a full OpenAI-compatible /v1 interface.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ─── g4f imports ────────────────────────────────────────────────────────────────
try:
    from g4f.client import AsyncClient
    from g4f.Provider.qwen.Qwen import Qwen          # web (no auth)
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    Qwen = None

try:
    from g4f.Provider.qwen.QwenCode import QwenCode  # OAuth2 (token)
    HAS_QWEN_CODE = True
except ImportError:
    HAS_QWEN_CODE = False
    QwenCode = None

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("qwen-api")

# ─── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qwen OpenAI-Compatible API",
    description=(
        "Drop-in OpenAI API replacement backed by Qwen (chat.qwen.ai).\n\n"
        "**Providers:**\n"
        "- `Qwen` – no login needed, cookie-based auth (default)\n"
        "- `QwenCode` – OAuth2 token, accesses `dashscope.aliyuncs.com`\n\n"
        "Set `Authorization: Bearer <your-qwen-token>` to pass a user token."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Available models ────────────────────────────────────────────────────────────
MODELS = [
    # Flagship
    "qwen3.7-plus", "qwen3.7-max",
    # Qwen 3.6
    "qwen3.6-plus", "qwen3.6-max-preview", "qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.6-plus-preview",
    # Qwen 3.5
    "qwen3.5-plus", "qwen3.5-omni-plus", "qwen3.5-flash",
    "qwen3.5-max-2026-03-08", "qwen3.5-397b-a17b", "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash", "qwen3.5-27b", "qwen3.5-35b-a3b",
    # Qwen 3
    "qwen3-max-2026-01-23", "qwen3-coder-plus", "qwen3-vl-plus", "qwen3-omni-flash-2025-12-01",
    # Other
    "qwen-plus-2025-07-28",
    "qwen-latest-series-invite-beta-v24", "qwen-latest-series-invite-beta-v16",
]

QWEN_CODE_MODELS = ["qwen3-coder-plus"]

# ─── Schemas ─────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen3.7-plus", description="Qwen model ID")
    messages: List[Message]
    stream: bool = Field(default=False)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Qwen-specific extras (passed through via extra_body)
    reasoning_effort: Optional[str] = Field(
        default=None,
        description='Thinking depth: "low" | "medium" | "high"'
    )
    chat_type: Optional[str] = Field(
        default="t2t",
        description='Mode: "t2t" | "search" | "artifacts" | "t2i" | "image_edit" | "t2v" | "web_dev" | "deep_research"'
    )
    provider: Optional[str] = Field(
        default=None,
        description='"Qwen" (default) or "QwenCode" (requires OAuth token)'
    )

# ─── Helpers ──────────────────────────────────────────────────────────────────────

def _chunk(model: str, content: str, req_id: str, finish_reason=None) -> str:
    delta = {}
    if content:
        delta["content"] = content
    payload = {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _completion(model: str, content: str, req_id: str) -> dict:
    return {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _pick_provider(request: ChatCompletionRequest, token: Optional[str]):
    """Choose between Qwen and QwenCode based on request + token."""
    name = (request.provider or "").lower()
    if name == "qwencode" or (request.model in QWEN_CODE_MODELS and token):
        if not HAS_QWEN_CODE:
            raise HTTPException(500, "QwenCode provider not installed.")
        return QwenCode
    if not HAS_QWEN:
        raise HTTPException(500, "Qwen provider not installed. Run: python install.py")
    return Qwen


# ─── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Qwen OpenAI-Compatible API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/v1/models", "/v1/chat/completions"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "qwen_available": HAS_QWEN,
        "qwen_code_available": HAS_QWEN_CODE,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 1720000000,
                "owned_by": "qwen",
                "permission": [],
                "root": m,
                "parent": None,
            }
            for m in MODELS
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if model_id not in MODELS:
        raise HTTPException(404, f"Model '{model_id}' not found.")
    return {
        "id": model_id,
        "object": "model",
        "created": 1720000000,
        "owned_by": "qwen",
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    # Parse optional Bearer token (Qwen user token)
    token: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip() or None

    provider = _pick_provider(request, token)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    model = request.model
    req_id = uuid.uuid4().hex

    # Extra kwargs passed to g4f (provider-specific)
    extra: dict = {}
    if request.reasoning_effort:
        extra["reasoning_effort"] = request.reasoning_effort
    if request.chat_type:
        extra["chat_type"] = request.chat_type
    if token and provider is Qwen:
        extra["token"] = token

    client = AsyncClient(provider=provider)

    # ── Streaming ──────────────────────────────────────────────────────────────
    if request.stream:
        async def stream_gen() -> AsyncGenerator[str, None]:
            # Opening delta with role
            opening = {
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(opening)}\n\n"

            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **extra,
                )
                async for chunk in response:
                    text = chunk.choices[0].delta.content
                    if text:
                        yield _chunk(model, text, req_id)
            except Exception as exc:
                logger.error("Streaming error: %s", exc, exc_info=True)
                err_payload = {"error": {"message": str(exc), "type": "provider_error", "code": 500}}
                yield f"data: {json.dumps(err_payload)}\n\n"

            yield _chunk(model, "", req_id, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming ──────────────────────────────────────────────────────────
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **extra,
        )
        content = response.choices[0].message.content or ""
        return _completion(model, content, req_id)
    except Exception as exc:
        logger.error("Completion error: %s", exc, exc_info=True)
        raise HTTPException(500, detail=str(exc))
