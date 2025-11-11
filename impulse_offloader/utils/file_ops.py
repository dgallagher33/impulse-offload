"""Utilities for file operations within the impulse vault."""
from __future__ import annotations

import importlib.util
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

_yaml_spec = importlib.util.find_spec("yaml")
if _yaml_spec:
    import yaml  # type: ignore[import]
else:  # pragma: no cover - fallback path
    yaml = None

DEFAULT_VAULT_SUBDIR = Path("vault")


def get_vault_path() -> Path:
    """Resolve the vault path from the environment or default location."""
    env_path = os.getenv("VAULT_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path.cwd() / DEFAULT_VAULT_SUBDIR


def ensure_directory(directory: Path) -> None:
    """Ensure that the provided directory exists."""
    directory.mkdir(parents=True, exist_ok=True)


def timestamped_filename(timestamp: Optional[datetime] = None, suffix: str = ".md") -> str:
    """Create a timestamped filename using ISO-like formatting."""
    ts = timestamp or datetime.now()
    return f"{ts.strftime('%Y-%m-%dT%H-%M-%S')}{suffix}"


def safe_write(path: Path, content: str) -> None:
    """Write content to a file, ensuring the parent directory exists."""
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def move_to_processed(source: Path, destination_dir: Path, content: Optional[str] = None) -> Path:
    """Move a note to the processed directory with optional updated content."""
    ensure_directory(destination_dir)
    destination = destination_dir / source.name
    if content is not None:
        destination.write_text(content, encoding="utf-8")
        source.unlink(missing_ok=True)
    else:
        source.replace(destination)
    return destination


def render_front_matter(metadata: Dict[str, object]) -> str:
    """Render metadata as YAML front matter."""
    if yaml is not None:
        yaml_str = yaml.safe_dump(metadata, sort_keys=False).strip()
    else:
        lines = []
        for key, value in metadata.items():
            if isinstance(value, list):
                joined = ", ".join(map(str, value))
                lines.append(f"{key}: [{joined}]")
            else:
                lines.append(f"{key}: {value}")
        yaml_str = "\n".join(lines)
    return f"---\n{yaml_str}\n---"


def list_markdown_files(directory: Path) -> Iterable[Path]:
    """Yield Markdown files within the provided directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".md")
