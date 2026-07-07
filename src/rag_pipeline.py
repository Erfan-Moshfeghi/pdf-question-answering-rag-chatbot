"""
Orchestration of the RAG pipeline.

The :class:`RAGPipeline` class ties together the PDF loader,
text splitter, embedding model, vector store, retriever and answer
generator.  It exposes a simple interface for loading documents and
answering questions.  Errors during loading or processing are
captured and returned as part of the status.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any

from .pdf_loader import PDFLoader
from .text_splitter import TextSplitter
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from .answer_generator import AnswerGenerator
from .config import DEFAULTS


class RAGPipeline:
    """High‑level pipeline for PDF question answering."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        # Use defaults if values are not provided
        self.chunk_size = chunk_size or DEFAULTS.chunk_size
        self.chunk_overlap = chunk_overlap or DEFAULTS.chunk_overlap
        self.top_k = top_k or DEFAULTS.top_k
        self.embedding_model_name = embedding_model_name or DEFAULTS.embedding_model_name

        # Create components
        self.loader = PDFLoader()
        self.splitter = TextSplitter(self.chunk_size, self.chunk_overlap)
        self.embedder = EmbeddingModel(self.embedding_model_name)
        self.store = VectorStore()
        self.retriever = Retriever(self.store, self.embedder, top_k=self.top_k)
        self.answerer = AnswerGenerator()

        # Keep track of loaded chunks for session state
        self._loaded = False

    def load_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Load and index a collection of PDF files.

        Parameters
        ----------
        file_paths : List[str]
            Paths to PDF files.

        Returns
        -------
        Dict[str, Any]
            A dictionary with status information.  Keys:
            - ``success``: boolean indicating if any documents were loaded.
            - ``errors``: list of error messages.
            - ``num_pages``: total number of pages processed.
            - ``num_chunks``: total number of chunks created.
        """
        errors: List[str] = []
        all_pages: List[Dict[str, object]] = []
        for path in file_paths:
            try:
                pages = self.loader.load(path)
                all_pages.extend(pages)
            except FileNotFoundError as exc:
                errors.append(str(exc))
            except ValueError as exc:
                errors.append(str(exc))
        if not all_pages:
            return {"success": False, "errors": errors, "num_pages": 0, "num_chunks": 0}
        # Split pages into chunks
        chunks = self.splitter.split_pages(all_pages)
        if not chunks:
            # All pages were empty or skipped
            errors.append(
                "No text found in the provided documents after splitting. Scanned PDFs may require OCR."
            )
            return {"success": False, "errors": errors, "num_pages": len(all_pages), "num_chunks": 0}
        # Fit embedding model and encode chunks
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        embeddings = self.embedder.encode(texts)
        # Add to vector store
        self.store = VectorStore()  # Reset store for new documents
        self.store.add(embeddings, chunks)
        # Update retriever with new store
        self.retriever = Retriever(self.store, self.embedder, top_k=self.top_k)
        self._loaded = True
        return {
            "success": True,
            "errors": errors,
            "num_pages": len(all_pages),
            "num_chunks": len(chunks),
        }

    def ask(self, question: str, top_k: int | None = None) -> Dict[str, Any]:
        """Answer a question using the indexed documents.

        Parameters
        ----------
        question : str
            The user query.
        top_k : Optional[int]
            Override the number of chunks to retrieve.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the answer, sources and status.
        """
        if not self._loaded:
            return {
                "success": False,
                "error": "No documents have been loaded. Please load documents before asking questions.",
            }
        results = self.retriever.retrieve(question, top_k=top_k)
        answer, sources = self.answerer.generate(question, results)
        return {
            "success": True,
            "answer": answer,
            "sources": sources,
        }