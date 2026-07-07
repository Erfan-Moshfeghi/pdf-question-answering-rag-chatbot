"""
Tests for the PDFLoader class.
"""

from __future__ import annotations

import os
import pytest

from src.pdf_loader import PDFLoader


def test_pdf_loader_reads_pages(sample_pdf: str) -> None:
    loader = PDFLoader()
    pages = loader.load(sample_pdf)
    # Should have two pages
    assert len(pages) == 2
    # Each page should have non‑zero text
    assert all(p["char_count"] > 0 for p in pages)
    # Metadata should match file name and page numbers 1..2
    basename = os.path.basename(sample_pdf)
    assert pages[0]["document"] == basename
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2


def test_pdf_loader_missing_file():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/path/does/not/exist.pdf")


def test_pdf_loader_empty_pdf(empty_pdf: str):
    loader = PDFLoader()
    with pytest.raises(ValueError):
        loader.load(empty_pdf)