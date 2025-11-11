"""Command line interface for capturing impulses."""
from __future__ import annotations

import sys
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


def _format_audio_path(vault_dir: Path, audio_path: Path) -> str:
    """Return a user and YAML-friendly representation of the audio path."""
    if audio_path.is_relative_to(vault_dir.parent):
        return str(audio_path.relative_to(vault_dir.parent))
    return str(audio_path)


def capture_impulse(
    text: Optional[str] = None,
    use_voice: bool = False,
    vault_path: Optional[Path] = None,
    model: str = "base",
    device: Optional[str] = None,
    max_seconds: int = 90,
    lang: str = "en",
    keep_audio: bool = True,
) -> Path:
    """Capture an impulse as a Markdown file inside the vault inbox."""

    env.load_environment()
    resolved_vault = vault_path or file_ops.get_vault_path()
    inbox_dir = resolved_vault / "Inbox"
    file_ops.ensure_directory(inbox_dir)

    if use_voice:
        audio_path, transcription = whisper_integration.capture_and_transcribe_local(
            vault_dir=resolved_vault,
            model=model,
            device=device,
            max_seconds=max_seconds,
            language=lang,
            keep_audio=keep_audio,
        )
        timestamp_str = audio_path.stem
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
        except ValueError:  # pragma: no cover - fallback to now if unexpected format
            timestamp = datetime.now()

        note_path = inbox_dir / f"{timestamp_str}.md"
        front_matter = file_ops.render_front_matter(
            {
                "date": timestamp.isoformat(),
                "type": "voice-dump",
                "backend": "local",
                "audio_path": _format_audio_path(resolved_vault, audio_path),
                "lang": lang,
                "summary": "",
                "tags": ["voice", "offload"],
            }
        )
        content = f"{front_matter}\n{transcription.strip()}\n"
        file_ops.safe_write(note_path, content)
        return note_path

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
@click.option("--model", default="base", show_default=True, help="Local Whisper model to use (tiny/base/small/medium/large).")
@click.option("--device", default=None, help="Input device name or index for the microphone.")
@click.option("--max-seconds", default=90, show_default=True, type=int, help="Maximum recording duration in seconds.")
@click.option("--lang", "lang", default="en", show_default=True, help="Language code to guide transcription.")
@click.option(
    "--keep-audio/--discard-audio",
    default=sys.platform.startswith("linux"),
    show_default=True,
    help="Keep or discard the captured WAV file after transcription.",
)
def capture(
    text: Optional[str],
    use_voice: bool,
    model: str,
    device: Optional[str],
    max_seconds: int,
    lang: str,
    keep_audio: bool,
) -> None:
    """Capture an impulse as Markdown in the Inbox."""

    if use_voice:
        text = None

    try:
        file_path = capture_impulse(
            text=text,
            use_voice=use_voice,
            model=model,
            device=device,
            max_seconds=max_seconds,
            lang=lang,
            keep_audio=keep_audio,
        )
    except (whisper_integration.AudioCaptureError, whisper_integration.TranscriptionError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Impulse captured: {file_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    capture()
