"""Processing pipeline for captured impulses."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .utils import env
from .utils import file_ops
from .utils import openai_helpers


def _build_front_matter(metadata: Dict[str, object]) -> str:
    """Format processing metadata as YAML front matter."""
    return file_ops.render_front_matter(metadata)


def process_inbox(vault_path: Optional[Path] = None) -> None:
    """Process all Markdown notes in the Inbox directory."""
    env.load_environment()
    resolved_vault = vault_path or file_ops.get_vault_path()
    inbox_dir = resolved_vault / "Inbox"
    processed_dir = resolved_vault / "Processed"

    for note_path in file_ops.list_markdown_files(inbox_dir):
        raw_text = note_path.read_text(encoding="utf-8")
        metadata = openai_helpers.summarize_and_tag(raw_text)
        front_matter = _build_front_matter(metadata)
        content = f"{front_matter}\n\n{raw_text.strip()}\n"
        file_ops.move_to_processed(note_path, processed_dir, content=content)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    process_inbox()
