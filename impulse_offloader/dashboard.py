"""Simple dashboard utilities for resurfacing processed notes."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

from .utils import file_ops


def sample_processed_notes(count: int = 3, vault_path: Optional[Path] = None) -> Iterable[Path]:
    """Return a random sample of processed notes."""
    resolved_vault = vault_path or file_ops.get_vault_path()
    processed_dir = resolved_vault / "Processed"
    notes = list(file_ops.list_markdown_files(processed_dir))
    if not notes:
        return []
    return random.sample(notes, k=min(count, len(notes)))


def display_dashboard(count: int = 3, vault_path: Optional[Path] = None) -> None:
    """Print a lightweight dashboard of processed notes."""
    notes = sample_processed_notes(count=count, vault_path=vault_path)
    if not notes:
        print("No processed notes available yet.")
        return
    print("Here are some resurfaced impulses:")
    for note in notes:
        print(f"- {note.name}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    display_dashboard()
