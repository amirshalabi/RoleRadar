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


def test_extract_text_from_pdf_empty_page_returns_empty_string() -> None:
    """A PDF with a page but no inserted text - not a crash, just an empty result for the caller to handle."""
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()

    result = extract_text_from_pdf(data)

    assert result == ""


def test_extract_text_from_pdf_malformed_bytes_raises() -> None:
    """Not a real PDF at all - PyMuPDF should raise, not silently return garbage or an empty string."""
    with pytest.raises(Exception):  # noqa: B017 - the exact PyMuPDF exception type is an implementation detail
        extract_text_from_pdf(b"this is not a pdf file at all, just plain bytes")


def test_extract_text_from_pdf_extremely_long_document_does_not_truncate() -> None:
    """A resume-length outlier (many pages) is extracted in full, not silently cut off at some page limit."""
    document = pymupdf.open()
    page_count = 50
    for i in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"PAGE_MARKER_{i}")
    data = document.tobytes()
    document.close()

    result = extract_text_from_pdf(data)

    assert result.count("PAGE_MARKER_") == page_count
    assert "PAGE_MARKER_0" in result
    assert f"PAGE_MARKER_{page_count - 1}" in result
