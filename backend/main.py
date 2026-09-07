"""
FastAPI application entrypoint.

Currently exposes only a health check endpoint. Feature routers
(candidate, ingestion, matching, planning, etc.) will be added here
as they are implemented.
"""

from fastapi import FastAPI

from backend.utils.logging import configure_logging

configure_logging()

app = FastAPI(title="RoleRadar API")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok"}
