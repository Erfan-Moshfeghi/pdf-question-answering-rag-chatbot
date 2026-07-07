"""
Miscellaneous helper functions.

This module defines small utility functions that are used across the
project.  Additional helpers can be added here to avoid duplicating
common logic.
"""

from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer used for tests and keyword extraction."""
    return re.findall(r"\b\w+\b", text.lower())