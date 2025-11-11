"""Command line interface for capturing impulses."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from .utils import env
from .utils import file_ops
from .utils import whisper_integration


def _prompt_for_text() -> str:
    """Prompt the user for free-form input until a non-empty value is provided."""
    prompt = "What is on your mind? "
    text = input(prompt).strip()
    while not text:
        click.echo("Please enter a thought to capture.")
        text = input(prompt).strip()
    return text


def capture_impulse(text: Optional[str] = None, use_voice: bool = False, vault_path: Optional[Path] = None) -> Path:
    """Capture an impulse as a Markdown file inside the vault inbox.

    Args:
        text: Optional pre-supplied text input.
        use_voice: Whether to capture audio using Whisper.
        vault_path: Optional override for the vault location. Defaults to the
            path resolved by :func:`file_ops.get_vault_path`.

    Returns:
        Path: The path to the created Markdown file.
    """
    env.load_environment()
    resolved_vault = vault_path or file_ops.get_vault_path()
    inbox_dir = resolved_vault / "Inbox"
    file_ops.ensure_directory(inbox_dir)

    if use_voice:
        try:
            text = whisper_integration.capture_voice_input()
        except NotImplementedError as exc:  # pragma: no cover - placeholder path
            raise click.ClickException("Voice capture is not implemented yet.") from exc

    if text is None:
        text = _prompt_for_text()

    if not text:
        raise click.ClickException("No impulse provided.")

    timestamp = datetime.now()
    filename = file_ops.timestamped_filename(timestamp=timestamp)
    file_path = inbox_dir / filename

    front_matter = file_ops.render_front_matter({
        "captured_at": timestamp.isoformat(),
    })
    content = f"{front_matter}\n{text.strip()}\n"
    file_ops.safe_write(file_path, content)

    return file_path


@click.command()
@click.argument("text", required=False)
@click.option("--voice", "use_voice", is_flag=True, help="Capture voice input via Whisper.")
def capture(text: Optional[str], use_voice: bool) -> None:
    """Capture an impulse as Markdown in the Inbox."""
    file_path = capture_impulse(text=text, use_voice=use_voice)
    click.echo(f"Impulse captured: {file_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    capture()
