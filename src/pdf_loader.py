"""
PDF loading utilities.

This module defines the :class:`PDFLoader` class, which is responsible
for reading PDF files from disk and extracting their textual content
page by page.  Each page is returned with associated metadata such as
the document name, page number and character count.  The loader
gracefully handles missing files, encrypted PDFs and scanned
documents that contain no extractable text.
"""

from __future__ import annotations

import os
from typing import List, Dict

from pypdf import PdfReader


class PDFLoader:
    """Load and extract text from PDF documents.

    The loader wraps ``pypdf.PdfReader`` to iterate over pages and
    produce a list of dictionaries with the following keys:

    ``document`` – The base name of the PDF file.
    ``page_number`` – One‑indexed page number in the document.
    ``text`` – The extracted text content of the page (empty if none).
    ``char_count`` – The length of the extracted text in characters.

    If the file cannot be read or contains no extractable text on any
    page, a :class:`ValueError` will be raised with a human‑readable
    explanation.
    """

    def load(self, file_path: str) -> List[Dict[str, object]]:
        """Read a PDF file from disk and extract text per page.

        Parameters
        ----------
        file_path : str
            Path to the PDF file.

        Returns
        -------
        List[Dict[str, object]]
            A list of dictionaries, one per page, containing the
            document name, page number, extracted text and character count.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the PDF cannot be opened or contains no extractable text.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        document_name = os.path.basename(file_path)

        try:
            reader = PdfReader(file_path)
        except Exception as exc:
            raise ValueError(f"Failed to read PDF '{document_name}': {exc}") from exc

        if reader.is_encrypted:
            # Attempt to decrypt with an empty password; some PDFs are
            # encrypted but can be opened without a password.  If
            # decryption fails, inform the user.
            try:
                success = reader.decrypt("")
            except Exception:
                success = 0
            if success == 0:
                raise ValueError(
                    f"The PDF '{document_name}' is encrypted and cannot be read without a password."
                )

        pages_data: List[Dict[str, object]] = []
        for i, page in enumerate(reader.pages, start=1):
            # Extract text; pypdf may return None for pages with images only
            try:
                text = page.extract_text() or ""
            except Exception:
                # If extraction fails for a page, continue with empty text
                text = ""
            pages_data.append(
                {
                    "document": document_name,
                    "page_number": i,
                    "text": text.strip(),
                    "char_count": len(text.strip()),
                }
            )

        # Check if any page contained text
        if all(page["char_count"] == 0 for page in pages_data):
            raise ValueError(
                f"No extractable text found in '{document_name}'. Scanned or image‑only PDFs require OCR, "
                "which is not implemented in this project."
            )

        return pages_data