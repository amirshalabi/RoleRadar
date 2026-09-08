"""
Centralized exception -> HTTP response mapping.

Every handler here translates an exception TYPE the backend already
raises (never a new exception hierarchy invented for the API) into a
meaningful HTTP status code, so a route handler never needs its own
try/except translating errors - it just calls backend.services.*/
backend.db.* and lets that layer's exceptions propagate.

FastAPI's own HTTPException handling (used for explicit 404s etc. in
route handlers) is untouched by any of this - these handlers only fire
for the backend's own exception types.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.db.client import SupabaseNotConfiguredError
from backend.llm.client import LLMExtractionError, OpenAINotConfiguredError
from backend.rag.vector_store import QdrantNotConfiguredError


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every backend-exception -> HTTP-response mapping to `app`."""

    @app.exception_handler(SupabaseNotConfiguredError)
    async def _supabase_not_configured(request: Request, exc: SupabaseNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(OpenAINotConfiguredError)
    async def _openai_not_configured(request: Request, exc: OpenAINotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(QdrantNotConfiguredError)
    async def _qdrant_not_configured(request: Request, exc: QdrantNotConfiguredError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMExtractionError)
    async def _llm_extraction_error(request: Request, exc: LLMExtractionError) -> JSONResponse:
        # The LLM provider returned something unusable - a failure of an
        # upstream dependency, not of this request's own input.
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        # Every ValueError our service layer raises is a rejected input
        # (e.g. "no interview date set yet") - a client error, not a server fault.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def _runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
        # e.g. "Insert into X returned no data" - an unexpected persistence
        # failure with nothing the client could have done differently.
        return JSONResponse(status_code=500, content={"detail": str(exc)})
