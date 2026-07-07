"""
Tests for the TextSplitter class.
"""

from __future__ import annotations

import pytest

from src.text_splitter import TextSplitter


def test_splitter_produces_overlapping_chunks():
    text = "123456789"
    pages = [
        {
            "document": "dummy.pdf",
            "page_number": 1,
            "text": text,
            "char_count": len(text),
        }
    ]
    splitter = TextSplitter(chunk_size=5, chunk_overlap=2)
    chunks = splitter.split_pages(pages)
    # Expected chunks: '12345', '45678', '789'
    expected = ["12345", "45678", "789"]
    assert [c["text"] for c in chunks] == expected
    # Check metadata
    assert all(c["document"] == "dummy.pdf" for c in chunks)
    assert all(c["page_number"] == 1 for c in chunks)
    # chunk_id increments
    assert [c["chunk_id"] for c in chunks] == [1, 2, 3]


def test_splitter_invalid_parameters():
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=0, chunk_overlap=1)
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=10, chunk_overlap=-1)