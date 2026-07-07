"""
Embedding utilities.

This module defines a small abstraction layer for computing vector
representations of text.  Two concrete implementations are provided:

* :class:`SentenceTransformerEmbeddings` uses a pre‑trained model from
  the `sentence_transformers` package.  The default model name is
  ``"all-MiniLM-L6-v2"``, which offers a good balance between speed
  and semantic quality.  This backend is the recommended choice for
  portfolio‑quality RAG applications.

* :class:`TfidfEmbeddings` uses a scikit‑learn TF–IDF vectorizer to
  compute sparse, lexical embeddings.  It serves as a lightweight
  fallback when neural models cannot be loaded (e.g. in offline
  environments) or for testing.

A factory function :func:`get_embedding_model` is provided to
instantiate the appropriate backend based on a model name.  The
wrapper class :class:`EmbeddingModel` is retained for backwards
compatibility with earlier versions of the project; it delegates all
operations to a concrete embedding implementation returned by
``get_embedding_model``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Union

import numpy as np


class BaseEmbeddings:
    """Abstract base class for embedding models."""

    def fit(self, texts: Iterable[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def encode(self, texts: Iterable[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class TfidfEmbeddings(BaseEmbeddings):
    """Compute TF–IDF embeddings using scikit‑learn.

    This class lazily imports scikit‑learn and fits a TF–IDF
    vectorizer on demand.  It yields sparse lexical vectors which are
    lightweight and deterministic.  They do not capture semantics
    beyond exact word matching but serve as a useful fallback when
    neural models are unavailable.
    """

    def __init__(self, stop_words: str | None = "english") -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for TF–IDF embeddings. Please install it via 'pip install scikit-learn'."
            ) from exc
        self._vectorizer = TfidfVectorizer(stop_words=stop_words)
        self._fitted = False

    def fit(self, texts: Iterable[str]) -> None:
        texts_list = list(texts)
        if texts_list:
            self._vectorizer.fit(texts_list)
            self._fitted = True

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not self._fitted:
            raise RuntimeError("TF–IDF vectorizer has not been fitted; call fit() first")
        matrix = self._vectorizer.transform(texts_list).astype(np.float64)
        return matrix.toarray()

    def encode_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF–IDF vectorizer has not been fitted; call fit() first")
        vec = self._vectorizer.transform([text]).astype(np.float64)
        return vec.toarray()[0]


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Compute semantic embeddings using a pre‑trained SentenceTransformer.

    Parameters
    ----------
    model_name : str, default "all-MiniLM-L6-v2"
        The name of the SentenceTransformer model to load.  If
        `sentence_transformers` is not installed or the model cannot be
        loaded, an exception will be raised.
    cache_model : bool, default True
        Whether to reuse a loaded model across instances to avoid
        redundant downloads and initialisation.
    """

    # Class‑level cache for loaded models
    _model_cache: dict[str, object] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_model: bool = True) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'sentence-transformers' package is required for semantic embeddings. Please install it via 'pip install sentence-transformers'."
            ) from exc

        self.model_name = model_name
        self.cache_model = cache_model
        if cache_model and model_name in SentenceTransformerEmbeddings._model_cache:
            self._model = SentenceTransformerEmbeddings._model_cache[model_name]
        else:
            # Loading the model may download weights on the first run
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as exc:
                raise ValueError(f"Failed to load SentenceTransformer model '{model_name}': {exc}") from exc
            if cache_model:
                SentenceTransformerEmbeddings._model_cache[model_name] = self._model

    def fit(self, texts: Iterable[str]) -> None:
        # SentenceTransformer models do not require fitting
        return None

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        # normalize_embeddings=True yields unit vectors
        embeddings = self._model.encode(texts_list, normalize_embeddings=True)
        return np.array(embeddings)

    def encode_query(self, text: str) -> np.ndarray:
        embedding = self._model.encode([text], normalize_embeddings=True)[0]
        return np.array(embedding)


def get_embedding_model(model_name: str | None = None) -> BaseEmbeddings:
    """Factory function to obtain an embedding model by name.

    Parameters
    ----------
    model_name : Optional[str]
        Name of the embedding backend.  Supported values:

        * ``None`` or ``"default"`` – return a semantic embedding model
          using the default SentenceTransformer.  If the
          sentence-transformers package is not installed or the model
          fails to load, fall back to TF–IDF.
        * ``"tfidf"`` – return a :class:`TfidfEmbeddings` instance.
        * any other string – attempt to load the corresponding
          SentenceTransformer model.  If it fails, an exception is
          propagated.

    Returns
    -------
    BaseEmbeddings
        An embedding model instance.
    """
    name = (model_name or "default").lower()
    if name == "tfidf":
        return TfidfEmbeddings()
    # Semantic embeddings are the default
    try:
        return SentenceTransformerEmbeddings(model_name or "all-MiniLM-L6-v2")
    except ImportError:
        # sentence-transformers not available; fall back to TF-IDF
        return TfidfEmbeddings()


class EmbeddingModel(BaseEmbeddings):
    """Wrapper class for backwards compatibility.

    The original API exposed a single :class:`EmbeddingModel` class
    parameterised by a ``model_name``.  For compatibility with
    existing code and tests, this wrapper delegates all operations
    to a concrete embedding instance returned by
    :func:`get_embedding_model`.
    """

    def __init__(self, model_name: str = "tfidf") -> None:
        self._impl = get_embedding_model(model_name)

    def fit(self, texts: Iterable[str]) -> None:
        return self._impl.fit(texts)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self._impl.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._impl.encode_query(text)