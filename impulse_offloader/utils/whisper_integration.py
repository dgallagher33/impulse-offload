"""Audio capture and local Whisper transcription utilities."""
from __future__ import annotations

from datetime import datetime
import shutil
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING, Any

try:  # pragma: no cover - import guard exercised in tests via monkeypatching
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    _np = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as np

try:  # pragma: no cover - import guard exercised in tests via monkeypatching
    import sounddevice  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    sounddevice = None  # type: ignore

try:  # pragma: no cover - import guard exercised in tests via monkeypatching
    from scipy.io import wavfile  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    wavfile = None  # type: ignore

import click


class AudioCaptureError(RuntimeError):
    """Raised when audio capture fails."""


class TranscriptionError(RuntimeError):
    """Raised when Whisper transcription fails."""


def _require_numpy() -> Any:
    if _np is None:
        raise AudioCaptureError(
            "numpy is required for audio processing. Install it with 'pip install numpy'."
        )
    return _np


def _require_sounddevice() -> "sounddevice":
    if sounddevice is None:
        raise AudioCaptureError(
            "sounddevice is required for recording audio. Install it with 'pip install sounddevice'."
        )
    return sounddevice  # type: ignore[return-value]


def _require_wavfile() -> "wavfile":
    if wavfile is None:
        raise AudioCaptureError(
            "scipy is required for writing WAV files. Install it with 'pip install scipy'."
        )
    return wavfile  # type: ignore[return-value]


def _validate_device(
    sd_module: "sounddevice", device: Optional[str]
) -> Optional[int | str]:
    if device is None:
        return None
    try:
        # Allow passing either an index or a device name.
        maybe_index = int(device)
        sd_module.query_devices(maybe_index, "input")
        return maybe_index
    except ValueError:
        try:
            sd_module.query_devices(device, "input")
            return device
        except Exception as exc:  # pragma: no cover - defensive
            raise AudioCaptureError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - device lookup failure
        raise AudioCaptureError(str(exc)) from exc


def _normalize_audio(buffer: Any) -> Any:
    np_module = _require_numpy()
    buffer = np_module.asarray(buffer, dtype=np_module.float32)
    max_abs = float(np_module.max(np_module.abs(buffer)))
    if max_abs == 0.0:
        raise AudioCaptureError("No audio captured from the microphone.")
    if max_abs > 1.0:
        buffer = buffer / max_abs
    buffer = np_module.clip(buffer, -1.0, 1.0)
    scaled = (buffer * np_module.iinfo(np_module.int16).max).astype(np_module.int16)
    return scaled


def record_audio(
    out_wav: Path,
    device: Optional[str] = None,
    max_seconds: int = 90,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Record microphone audio to ``out_wav`` for up to ``max_seconds`` seconds.

    The function records audio using :mod:`sounddevice` and persists it as a 16kHz
    mono WAV file using :mod:`scipy.io.wavfile`. If the requested input device is
    unavailable, the exception includes a list of discoverable devices to aid
    troubleshooting. The output path is always returned when audio was captured.

    Args:
        out_wav: Destination path for the WAV file.
        device: Optional device name or index understood by sounddevice.
        max_seconds: Hard time limit for the recording.
        sample_rate: Recording sample rate in Hz.
        channels: Number of input channels. Defaults to mono.

    Returns:
        Path: The ``out_wav`` path provided.

    Raises:
        AudioCaptureError: If dependencies are missing, no audio is captured, or
            the device configuration fails.
    """

    sd_module = _require_sounddevice()
    wav_module = _require_wavfile()

    if max_seconds <= 0:
        raise AudioCaptureError("max_seconds must be greater than zero.")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    try:
        resolved_device = _validate_device(sd_module, device)
        sd_module.check_input_settings(device=resolved_device, samplerate=sample_rate, channels=channels)
    except Exception as exc:
        devices_listing = sd_module.query_devices()
        device_table = "\n".join(str(d) for d in devices_listing)
        message = (
            f"Failed to initialise the input device '{device}'.\n"
            "Available devices:\n"
            f"{device_table}\n"
            "Use --device to select a device by index or name."
        )
        raise AudioCaptureError(message) from exc

    frames = int(sample_rate * max_seconds)
    click.echo(f"Recording (max {max_seconds}s)…")
    try:
        recording = sd_module.rec(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=resolved_device,
        )
        sd_module.wait()
    except Exception as exc:  # pragma: no cover - runtime capture issues
        raise AudioCaptureError(f"Recording failed: {exc}") from exc

    np_module = _require_numpy()
    buffer = np_module.asarray(recording, dtype=np_module.float32)
    buffer = np_module.squeeze(buffer)
    if hasattr(buffer, "ndim") and buffer.ndim == 0:
        buffer = np_module.expand_dims(buffer, axis=0)
    normalized = _normalize_audio(buffer)

    wav_module.write(str(out_wav), sample_rate, normalized)
    return out_wav


def transcribe_local(
    audio_path: Path,
    model: str = "base",
    language: str = "en",
) -> str:
    """Transcribe audio using a locally loaded Whisper model."""

    if shutil.which("ffmpeg") is None:
        raise TranscriptionError(
            "ffmpeg is required for Whisper transcription. Install it via 'sudo apt-get install ffmpeg'."
        )

    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise TranscriptionError(
            "The whisper package is required for local transcription. Install it with 'pip install whisper'."
        ) from exc

    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    click.echo(f"Transcribing locally with Whisper (model={model})…")
    try:
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(str(audio_path), language=language)
    except Exception as exc:  # pragma: no cover - runtime transcription issues
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    text = result.get("text", "").strip()
    if not text:
        raise TranscriptionError("Whisper returned an empty transcription.")
    return text


def capture_and_transcribe_local(
    vault_dir: Path,
    model: str = "base",
    device: Optional[str] = None,
    max_seconds: int = 90,
    language: str = "en",
    keep_audio: bool = True,
) -> Tuple[Path, str]:
    """Capture audio and transcribe it using the local Whisper backend."""

    inbox_dir = vault_dir / "Inbox"
    audio_dir = inbox_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    inbox_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().replace(microsecond=0)
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    audio_path = audio_dir / f"{timestamp_str}.wav"

    recorded_path = record_audio(
        audio_path,
        device=device,
        max_seconds=max_seconds,
    )

    try:
        relative_audio = recorded_path.relative_to(vault_dir.parent)
    except ValueError:
        relative_audio = recorded_path
    click.echo(f"Saved audio: {relative_audio}")

    transcription = transcribe_local(recorded_path, model=model, language=language)

    if not keep_audio and recorded_path.exists():
        recorded_path.unlink()

    return audio_path, transcription
