"""
Answer generation for the PDF RAG Chatbot.

This module defines the AnswerGenerator class. It takes retrieved chunks
and the user question as input and constructs a concise, source-grounded
answer using a simple extractive approach.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


class AnswerGenerator:
    """Generate an answer from retrieved chunks."""

    DEFAULT_STOPWORDS = {
        "the", "and", "or", "but", "for", "with", "from", "that", "this",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "can", "could", "should", "would", "will",
        "about", "into", "onto", "than", "then", "there", "their", "they",
        "you", "your", "its", "also", "only", "known",
    }

    def __init__(
        self,
        insufficient_msg: str | None = None,
        min_relevance_score: float = 0.15,
    ) -> None:
        """Initialize the answer generator.

        Parameters
        ----------
        insufficient_msg:
            Message returned when the document does not contain enough evidence.
        min_relevance_score:
            Minimum similarity score required to trust retrieved chunks.
        """
        self.insufficient_msg = insufficient_msg or (
            "I could not find enough information in the uploaded document to answer this question."
        )
        self.min_relevance_score = min_relevance_score

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract simple keywords from the user query."""
        tokens = re.findall(r"\b\w+\b", query.lower())
        return [
            token
            for token in tokens
            if len(token) > 2 and token not in self.DEFAULT_STOPWORDS
        ]

    def _split_sentences(self, text: str) -> List[str]:
        """Split a chunk of text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _best_score(self, retrieved: List[Dict[str, object]]) -> float:
        """Return the best numeric retrieval score."""
        scores: List[float] = []

        for item in retrieved:
            score = item.get("score")
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue

        return max(scores) if scores else 0.0

    def generate(
        self,
        query: str,
        retrieved: List[Dict[str, object]],
    ) -> Tuple[str, List[Dict[str, object]]]:
        """Generate an answer from retrieved chunks."""
        if not retrieved:
            return self.insufficient_msg, []

        if self._best_score(retrieved) < self.min_relevance_score:
            return self.insufficient_msg, []

        keywords = self._extract_keywords(query)

        if not keywords:
            return self.insufficient_msg, []

        matched_sentences: List[str] = []
        sources: List[Dict[str, object]] = []

        for item in retrieved:
            text = item.get("text", "")

            if not text:
                continue

            sentences = self._split_sentences(str(text))

            for sentence in sentences:
                sentence_lower = sentence.lower()

                if any(keyword in sentence_lower for keyword in keywords):
                    matched_sentences.append(sentence.strip())

            source = {
                "document": item.get("document"),
                "page_number": item.get("page_number"),
                "chunk_id": item.get("chunk_id"),
                "text": text,
                "score": item.get("score"),
            }
            sources.append(source)

        if not matched_sentences:
            return self.insufficient_msg, []

        unique_sentences: List[str] = []

        for sentence in matched_sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)

            if len(unique_sentences) >= 3:
                break

        answer = " ".join(unique_sentences)
        return answer, sources
