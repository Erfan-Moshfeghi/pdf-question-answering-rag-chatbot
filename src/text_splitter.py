"""
Text splitting utilities.

This module defines the :class:`TextSplitter` class, which divides raw
text from PDF pages into overlapping chunks.  Splitting text into
smaller pieces allows the retrieval model to find relevant passages
without indexing entire pages.  Each chunk retains metadata about
its originating document and page.
"""

from __future__ import annotations

from typing import List, Dict


class TextSplitter:
    """Split page text into overlapping chunks.

    Parameters
    ----------
    chunk_size : int, default 800
        The target number of characters in each chunk.  Must be greater
        than ``chunk_overlap``.
    chunk_overlap : int, default 150
        The number of overlapping characters between consecutive chunks.
        Must be non‑negative.

    Raises
    ------
    ValueError
        If ``chunk_size`` is not greater than ``chunk_overlap``.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Split a list of page dictionaries into chunks.

        Each page dictionary must contain ``text``, ``document`` and
        ``page_number`` keys.  The returned list contains chunk
        dictionaries with ``document``, ``page_number``, ``chunk_id`` and
        ``text`` keys.

        Parameters
        ----------
        pages : List[dict]
            A list of page metadata dictionaries as returned by
            :meth:`PDFLoader.load`.

        Returns
        -------
        List[dict]
            A list of chunk dictionaries.
        """
        chunks: List[Dict[str, object]] = []
        for page in pages:
            text: str = page.get("text", "")
            # Skip pages with no text
            if not text:
                continue
            page_len = len(text)
            start = 0
            chunk_idx = 1
            while start < page_len:
                end = start + self.chunk_size
                chunk_text = text[start:end]
                # Strip leading/trailing whitespace from the chunk
                chunk_text = chunk_text.strip()
                if chunk_text:
                    chunks.append(
                        {
                            "document": page["document"],
                            "page_number": page["page_number"],
                            "chunk_id": chunk_idx,
                            "text": chunk_text,
                        }
                    )
                    chunk_idx += 1
                # Move the start forward; ensure that we do not move
                # backwards if page_len < chunk_size
                if end >= page_len:
                    break
                start = end - self.chunk_overlap
        return chunks