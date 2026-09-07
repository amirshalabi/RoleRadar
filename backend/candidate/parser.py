"""
Resume parsing.

Extracts raw text from uploaded resume PDFs using PyMuPDF, producing
plain text for downstream structured extraction. Contains no LLM calls -
this is pure text extraction.

Callers that already have raw resume text (e.g. pasted into a textbox)
should skip this module entirely and pass that text directly to
backend.candidate.profile.extract_candidate_profile().
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """
    Extract plain text from a PDF resume.

    `source` may be a filesystem path (str or Path) or raw PDF bytes
    (e.g. from a Streamlit file uploader). Returns concatenated text
    from every page, stripped of leading/trailing whitespace.
    """
    if isinstance(source, (str, Path)):
        logger.info("Extracting resume text from PDF path: %s", source)
        document = pymupdf.open(str(source))
    elif isinstance(source, (bytes, bytearray)):
        logger.info("Extracting resume text from PDF bytes (%d bytes)", len(source))
        document = pymupdf.open(stream=bytes(source), filetype="pdf")
    else:
        raise TypeError(f"Unsupported PDF source type: {type(source)!r}")

    try:
        pages_text = [page.get_text() for page in document]
    finally:
        document.close()

    return "\n".join(pages_text).strip()
