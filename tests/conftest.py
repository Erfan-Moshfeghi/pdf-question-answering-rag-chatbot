"""
Pytest fixtures for the PDF RAG Chatbot tests.

These fixtures generate temporary PDF files with known content using
reportlab.  They are used by multiple test modules.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator

import pytest
from reportlab.pdfgen import canvas

# Ensure the src package is importable when running tests.
# By inserting the parent directory's src into sys.path, we allow
# imports such as ``from src.pdf_loader import PDFLoader`` to succeed.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_pdf(tmp_path: os.PathLike) -> Iterator[str]:
    """Create a simple two‑page PDF with text on each page."""
    path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "Hello world page 1")
    c.showPage()
    c.drawString(72, 720, "This is page 2 text")
    c.save()
    yield str(path)


@pytest.fixture
def empty_pdf(tmp_path: os.PathLike) -> Iterator[str]:
    """Create a PDF with two blank pages (no extractable text)."""
    path = tmp_path / "empty.pdf"
    c = canvas.Canvas(str(path))
    # Create two blank pages without drawing any text
    c.showPage()
    c.showPage()
    c.save()
    yield str(path)