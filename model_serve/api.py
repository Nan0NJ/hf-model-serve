"""FastAPI application for the Transformers backend."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

# CUDA selection must happen before importing the backend (and therefore torch).
_config_path = os.environ.get("MODEL_SERVER_CONFIG")
if not _config_path:
    raise RuntimeError("MODEL_SERVER_CONFIG is required; launch through model-server start")
_raw = json.loads(Path(_config_path).read_text(encoding="utf-8"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(_raw["gpus"])

from fastapi import Depends, FastAPI, Header, HTTPException

from .backends.transformers_backend import TransformersBackend
from .config import ServerConfig
from .schemas import ChatCompletionRequest, ChatCompletionResponse, Choice, ChoiceMessage, Usage

config = ServerConfig(**_raw).normalized()
backend = TransformersBackend(config)
app = FastAPI(title="HF Model Serve", version="1.0.0")


def authorize(authorization: str | None = Header(default=None)) -> None:
    if not config.api_key_env:
        return
    expected = os.environ.get(config.api_key_env)
    if not expected:
        raise HTTPException(503, f"API key environment variable {config.api_key_env} is not set")
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, "invalid or missing bearer token")


@app.on_event("startup")
def load_model() -> None:
    backend.load()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok" if backend.loaded else "loading", "model": config.served_model_name, "loaded": backend.loaded}


@app.get("/v1/models", dependencies=[Depends(authorize)])
def models() -> dict[str, object]:
    return {"object": "list", "data": [{"id": config.served_model_name, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(authorize)])
def chat(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(400, "streaming is not implemented; send stream=false")
    if request.model and request.model != config.served_model_name:
        raise HTTPException(404, f"model {request.model!r} is not served")
    try:
        result = backend.generate(
            [m.model_dump() for m in request.messages],
            request.max_tokens or config.max_output_tokens,
            request.temperature,
            request.top_p,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        status = 507 if "out of memory" in str(exc).lower() else 500
        raise HTTPException(status, str(exc)) from exc
    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        created=int(time.time()),
        model=str(config.served_model_name),
        choices=[Choice(message=ChoiceMessage(content=result.text), finish_reason=result.finish_reason)],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


@app.get("/internal/metrics", dependencies=[Depends(authorize)])
def metrics() -> dict[str, object]:
    return {"loaded": backend.loaded, **backend.metrics()}
