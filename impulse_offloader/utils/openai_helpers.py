"""Helpers for interacting with the OpenAI API."""
from __future__ import annotations

from typing import Dict


def summarize_and_tag(text: str) -> Dict[str, object]:
    """Return a dictionary with summary, type, and tags extracted via GPT-4.

    This function is intentionally left as a stub so that it can be mocked
    during testing and implemented later with real API calls.
    """
    raise NotImplementedError("OpenAI summarization is not implemented yet.")
