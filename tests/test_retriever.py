"""
Tests for the Retriever class.
"""

from __future__ import annotations

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever


def test_retriever_returns_relevant_chunks():
    # Simple corpus
    texts = ["apple banana", "banana carrot", "carrot date", "elephant fig"]
    metadatas = [
        {"document": "doc.pdf", "page_number": 1, "chunk_id": i + 1, "text": t}
        for i, t in enumerate(texts)
    ]
    model = EmbeddingModel("tfidf")
    model.fit(texts)
    embeddings = model.encode(texts)
    store = VectorStore()
    store.add(embeddings, metadatas)
    retriever = Retriever(store, model, top_k=2)
    # Query for 'banana' should match the first two texts
    results = retriever.retrieve("banana")
    assert len(results) == 2
    top_texts = [res["text"] for res in results]
    assert "apple banana" in top_texts
    assert "banana carrot" in top_texts