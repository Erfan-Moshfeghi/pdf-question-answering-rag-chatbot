"""
Configuration module for the PDF Question Answering RAG Chatbot.

All default parameters for chunking, retrieval and embedding are defined
here to avoid scattering magic numbers throughout the code.  Users of
the library can override these values by passing explicit parameters
to the classes or using environment variables in the future.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Defaults:
    """Default configuration values for the RAG pipeline."""

    # Text splitting
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 3

    # Embedding model
    # Embedding model name: use a semantic model by default
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector store path (unused in memory implementation)
    vector_store_dir: str = "data/vector_store"


DEFAULTS = Defaults()