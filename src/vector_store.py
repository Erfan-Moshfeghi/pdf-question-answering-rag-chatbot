"""
Vector store implementation.

This module defines the :class:`VectorStore` class, which stores
embeddings and their associated metadata.  It supports cosine
similarity search for retrieval.  The default implementation keeps
everything in memory.  Persisting to disk can be added later
without changing the interface.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np


class VectorStore:
    """Simple in-memory vector store using cosine similarity.

    Each item in the store consists of an embedding vector and a
    metadata dictionary describing the original chunk (e.g. text,
    document name, page number, chunk id).  The store supports adding
    multiple items at once and querying for the top‑k most similar
    items to a given vector.
    """

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._metadatas: List[Dict[str, object]] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, object]]) -> None:
        """Add multiple embeddings and their metadata to the store.

        Parameters
        ----------
        embeddings : numpy.ndarray
            A two‑dimensional array of shape (n_items, embedding_dim).
        metadatas : List[Dict[str, object]]
            A list of metadata dictionaries of length n_items.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(metadatas) != embeddings.shape[0]:
            raise ValueError("Number of metadatas must match number of embeddings")
        # Initialise or append
        if self._embeddings is None:
            self._embeddings = embeddings.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        self._metadatas.extend(metadatas)

    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all stored vectors."""
        if self._embeddings is None or self._embeddings.size == 0:
            return np.array([])
        # Ensure query is 1D
        query_vec = query.reshape(1, -1)
        # Compute dot products and norms
        denom = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec)
        # Avoid division by zero
        denom[denom == 0] = 1e-10
        sims = (self._embeddings @ query_vec.T).flatten() / denom
        return sims

    def query(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, object]]:
        """Return the top‑k most similar items to the query vector.

        Parameters
        ----------
        query_vector : numpy.ndarray
            The query embedding as a 1D vector.
        top_k : int, default 3
            The number of top results to return.

        Returns
        -------
        List[Dict[str, object]]
            A list of result dictionaries ordered by descending similarity.
            Each dictionary contains the metadata of the chunk plus a
            ``score`` field.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._embeddings is None or len(self._metadatas) == 0:
            return []
        sims = self._cosine_similarity(query_vector)
        if sims.size == 0:
            return []
        # Get indices of top_k similarities
        sorted_indices = np.argsort(-sims)
        results: List[Dict[str, object]] = []
        for idx in sorted_indices[: min(top_k, len(sorted_indices))]:
            meta = self._metadatas[idx].copy()
            meta["score"] = float(sims[idx])
            results.append(meta)
        return results