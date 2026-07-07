"""
Retriever component.

This module defines the :class:`Retriever` class, which wraps a
vector store and an embedding model to find the most relevant chunks
for a given query.  It computes the embedding of the query using
the embedding model and performs a cosine similarity search through
the vector store.
"""

from __future__ import annotations

from typing import List, Dict

from .vector_store import VectorStore
from .embeddings import EmbeddingModel


class Retriever:
    """Retrieve relevant chunks from a vector store for a query."""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel, top_k: int = 3) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, object]]:
        """Retrieve the most relevant chunks for a given query.

        Parameters
        ----------
        query : str
            The user question.
        top_k : Optional[int]
            Override the default number of results to return.

        Returns
        -------
        List[Dict[str, object]]
            A list of metadata dictionaries for the top results, each
            including a similarity score.
        """
        k = top_k if top_k is not None else self.top_k
        if not query.strip():
            return []
        query_vec = self.embedding_model.encode_query(query)
        return self.vector_store.query(query_vec, top_k=k)