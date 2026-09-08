"""
FastAPI application entrypoint.

Wires together the API routers (backend/api/routes/*), one per resource
area matching backend/services' own module boundaries, and registers
the shared exception handlers (backend/api/errors.py) that translate
backend.* exceptions into meaningful HTTP status codes.

Route handlers never contain business logic themselves - they call
backend.services.* (or, for simple persistence with no service-layer
equivalent, backend.db.* directly, the same directness several
Streamlit pages already use) and shape the result into a Pydantic
response model (backend/api/schemas.py). Run locally with:

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from backend.api.errors import register_exception_handlers
from backend.api.routes import analytics, applications, assessments, candidate, favorites, health, plans, roles
from backend.utils.logging import configure_logging

configure_logging()

app = FastAPI(title="RoleRadar API")

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(candidate.router)
app.include_router(roles.router)
app.include_router(favorites.router)
app.include_router(applications.router)
app.include_router(plans.router)
app.include_router(assessments.router)
app.include_router(analytics.router)
