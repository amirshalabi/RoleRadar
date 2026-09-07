"""
Tests for backend.candidate.parser (PDF text extraction).

Builds tiny in-memory PDFs with PyMuPDF itself rather than relying on
fixture files, so these tests need no LLM calls and no network access.
"""

from __future__ import annotations

import pymupdf
import pytest

from backend.candidate.parser import extract_text_from_pdf


def _make_pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_extract_text_from_pdf_bytes() -> None:
    pdf_bytes = _make_pdf_bytes("Hello Resume Text")

    result = extract_text_from_pdf(pdf_bytes)

    assert "Hello Resume Text" in result


def test_extract_text_from_pdf_path(tmp_path) -> None:
    pdf_bytes = _make_pdf_bytes("Jane Doe Software Engineer")
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(pdf_bytes)

    result = extract_text_from_pdf(pdf_path)

    assert "Jane Doe Software Engineer" in result


def test_extract_text_from_pdf_accepts_str_path(tmp_path) -> None:
    pdf_bytes = _make_pdf_bytes("Str Path Test")
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(pdf_bytes)

    result = extract_text_from_pdf(str(pdf_path))

    assert "Str Path Test" in result


def test_extract_text_from_pdf_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        extract_text_from_pdf(12345)  # type: ignore[arg-type]
