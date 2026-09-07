"""
Shared logging configuration for the FastAPI backend and Streamlit frontend.

Call configure_logging() once at process startup (already done in
backend/main.py and app.py) so log format and level stay consistent
across both processes and can be controlled via the LOG_LEVEL
environment variable.
"""

import logging

from backend.utils.config import get_settings


def configure_logging() -> None:
    """Configure root logging handlers and level from application settings."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
