"""
Command‑line interface for the PDF Question Answering RAG Chatbot.

This script allows you to process PDF documents and ask a single
question from the command line.  It uses the RAG pipeline defined
in :mod:`src.rag_pipeline`.

Example
-------

    python main.py --pdf data/sample_pdfs/example.pdf --question "What is this document about?"
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from src.rag_pipeline import RAGPipeline


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process PDF files and answer a question using a retrieval‑augmented pipeline."
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        help="Path(s) to one or more PDF files to process.",
        required=True,
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question to ask about the contents of the PDF documents.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of characters per chunk (default from config).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Number of overlapping characters between chunks (default from config).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top results to retrieve (default from config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (default 'tfidf').  Requires 'sentence-transformers' if not tfidf.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embedding_model_name=args.model,
    )
    load_result = pipeline.load_documents(args.pdf)
    if not load_result.get("success"):
        print("Failed to load documents:")
        for err in load_result.get("errors", []):
            print(f"  - {err}")
        return 1
    result = pipeline.ask(args.question)
    if not result.get("success"):
        print(result.get("error"))
        return 1
    print("Answer:")
    print(result["answer"])
    print()
    # Print sources
    sources = result.get("sources", [])
    if sources:
        print("Sources:")
        for src in sources:
            doc = src.get("document")
            page = src.get("page_number")
            chunk_id = src.get("chunk_id")
            score = src.get("score")
            print(f"  - {doc}, page {page}, chunk {chunk_id}, score {score:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())