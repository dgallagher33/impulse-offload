"""Tests for the processing pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from impulse_offloader import process
from impulse_offloader.utils import file_ops


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))


def test_process_inbox_moves_and_enriches_notes(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "Inbox"
    processed = tmp_path / "Processed"
    file_ops.ensure_directory(inbox)

    note = inbox / "2024-01-01T00-00-00.md"
    note.write_text("Raw note content", encoding="utf-8")

    monkeypatch.setattr(
        process.openai_helpers,
        "summarize_and_tag",
        lambda text: {"summary": "Summary", "type": "idea", "tags": ["test"]},
    )

    process.process_inbox(vault_path=tmp_path)

    assert not note.exists()
    processed_notes = list(processed.glob("*.md"))
    assert len(processed_notes) == 1

    contents = processed_notes[0].read_text(encoding="utf-8")
    assert "summary" in contents
    assert "Raw note content" in contents
