"""
Core package for the PDF Question Answering RAG Chatbot.

This package exposes high‑level classes and functions that implement the
retrieval‑augmented generation pipeline.  See the individual modules
for detailed documentation.
"""

__all__ = [
    "PDFLoader",
    "TextSplitter",
    "EmbeddingModel",
    "VectorStore",
    "Retriever",
    "AnswerGenerator",
    "RAGPipeline",
]

from .pdf_loader import PDFLoader
from .text_splitter import TextSplitter
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from .answer_generator import AnswerGenerator
from .rag_pipeline import RAGPipeline