"""Tests for the capture module."""
from __future__ import annotations

from pathlib import Path

import pytest

from impulse_offloader import capture


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))


def test_capture_impulse_creates_file(tmp_path: Path) -> None:
    note_path = capture.capture_impulse(text="Test impulse", vault_path=tmp_path)
    assert note_path.exists()
    assert note_path.parent.name == "Inbox"
    assert note_path.suffix == ".md"

    contents = note_path.read_text(encoding="utf-8")
    assert "---" in contents
    assert "Test impulse" in contents
