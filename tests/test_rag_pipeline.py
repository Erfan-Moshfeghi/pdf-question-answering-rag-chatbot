"""
Tests for the RAGPipeline class.
"""

from __future__ import annotations

import pytest

from src.rag_pipeline import RAGPipeline


def test_pipeline_requires_loading_before_asking(sample_pdf: str) -> None:
    pipeline = RAGPipeline()
    # Asking before loading should return an error
    result = pipeline.ask("What is this?")
    assert not result["success"]
    assert "load documents" in result["error"].lower()


def test_pipeline_returns_fallback_when_answer_not_found(sample_pdf: str) -> None:
    pipeline = RAGPipeline(chunk_size=200, chunk_overlap=50, top_k=3)
    pipeline.load_documents([sample_pdf])
    result = pipeline.ask("What is the capital of France?")
    assert result["success"]
    # Since 'capital of France' does not appear, answer should contain fallback message
    assert "could not find" in result["answer"].lower()


def test_pipeline_returns_answer_from_document(sample_pdf: str) -> None:
    pipeline = RAGPipeline(chunk_size=50, chunk_overlap=10, top_k=3)
    pipeline.load_documents([sample_pdf])
    # Ask about a known phrase
    question = "What does the first page say?"
    result = pipeline.ask(question)
    assert result["success"]
    answer = result["answer"]
    # The answer should mention 'Hello world'
    assert "hello world" in answer.lower()