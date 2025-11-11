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


def _format_relative_path(path: Path) -> str:
    """Return a user-friendly string for ``path`` relative to the CWD when possible."""

    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def capture_impulse(
    text: Optional[str] = None,
    use_voice: bool = False,
    vault_path: Optional[Path] = None,
    model: str = "base",
    device: Optional[str] = None,
    max_seconds: int = 90,
    language: str = "en",
    keep_audio: bool = True,
) -> Path:
    """Capture an impulse as a Markdown file inside the vault inbox."""

    env.load_environment()
    resolved_vault = vault_path or file_ops.get_vault_path()
    inbox_dir = resolved_vault / "Inbox"
    file_ops.ensure_directory(inbox_dir)

    audio_path: Optional[Path] = None

    if use_voice:
        click.echo(f"Recording (max {max_seconds}s)…")
        click.echo(f"Transcribing locally with Whisper (model={model})…")
        try:
            audio_path, text = whisper_integration.capture_and_transcribe_local(
                vault_dir=resolved_vault,
                model=model,
                device=device,
                max_seconds=max_seconds,
                language=language,
                keep_audio=keep_audio,
            )
        except whisper_integration.AudioCaptureError as exc:
            raise click.ClickException(str(exc)) from exc
        except whisper_integration.TranscriptionError as exc:
            path_hint = exc.audio_path
            message = str(exc)
            if path_hint is not None:
                message = f"{message}\nRaw audio saved at: {_format_relative_path(path_hint)}"
            raise click.ClickException(message) from exc

        click.echo(f"Saved audio: {_format_relative_path(audio_path)}")

    if text is None:
        text = _prompt_for_text()

    if not text:
        raise click.ClickException("No impulse provided.")

    if audio_path is not None:
        timestamp_str = audio_path.stem
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
        note_filename = f"{timestamp_str}.md"
        try:
            relative_to_vault = audio_path.relative_to(resolved_vault)
            audio_relative = str(Path(resolved_vault.name) / relative_to_vault)
        except ValueError:
            audio_relative = _format_relative_path(audio_path)
        metadata = {
            "date": timestamp.isoformat(),
            "type": "voice-dump",
            "backend": "local",
            "audio_path": audio_relative,
            "lang": language,
            "summary": "",
            "tags": ["voice", "offload"],
        }
    else:
        timestamp = datetime.now()
        note_filename = file_ops.timestamped_filename(timestamp=timestamp)
        metadata = {
            "captured_at": timestamp.isoformat(),
        }

    file_path = inbox_dir / note_filename
    front_matter = file_ops.render_front_matter(metadata)
    content = f"{front_matter}\n\n{text.strip()}\n"
    file_ops.safe_write(file_path, content)

    return file_path


@click.command()
@click.argument("text", required=False)
@click.option("--voice", "use_voice", is_flag=True, help="Capture voice input via Whisper.")
@click.option("--model", default="base", show_default=True, help="Whisper model to use for transcription.")
@click.option("--device", default=None, help="Input device index or name for recording.")
@click.option("--max-seconds", default=90, show_default=True, type=int, help="Maximum recording duration in seconds.")
@click.option("--lang", "language", default="en", show_default=True, help="Language hint for Whisper transcription.")
@click.option(
    "--keep-audio/--discard-audio",
    default=True,
    show_default=True,
    help="Keep or discard the raw audio file after transcription.",
)
def capture(
    text: Optional[str],
    use_voice: bool,
    model: str,
    device: Optional[str],
    max_seconds: int,
    language: str,
    keep_audio: bool,
) -> None:
    """Capture an impulse as Markdown in the Inbox."""

    if use_voice:
        text = None

    file_path = capture_impulse(
        text=text,
        use_voice=use_voice,
        model=model,
        device=device,
        max_seconds=max_seconds,
        language=language,
        keep_audio=keep_audio,
    )

    click.echo(f"Impulse captured: {_format_relative_path(file_path)}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    capture()
