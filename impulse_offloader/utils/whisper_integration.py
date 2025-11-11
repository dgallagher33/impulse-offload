"""Prompt for Codex - Implement Local Whisper Voice Capture (Linux, 90s, keep audio).

Utilities for recording microphone input and transcribing it locally with the
Whisper model suite. Tailored for Linux environments with ``ffmpeg`` and ALSA/
PulseAudio support.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:  # pragma: no cover - import guard for optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard for optional dependency
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime
    sd = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard for optional dependency
    from scipy.io import wavfile
except ImportError:  # pragma: no cover - handled at runtime
    wavfile = None  # type: ignore[assignment]


class AudioCaptureError(RuntimeError):
    """Raised when audio capture fails for a user-facing reason."""


class TranscriptionError(RuntimeError):
    """Raised when transcription fails for a user-facing reason."""

    def __init__(self, message: str, *, audio_path: Optional[Path] = None) -> None:
        super().__init__(message)
        self.audio_path = audio_path


def _resolve_device(device: Optional[str]) -> Optional[int]:
    """Resolve a user-provided device string into a sounddevice identifier."""

    if device is None:
        return None

    devices = sd.query_devices()
    if device.isdigit():
        idx = int(device)
        if 0 <= idx < len(devices):
            return idx
    for idx, entry in enumerate(devices):
        if device.lower() in (str(entry.get("name", "")) or "").lower():
            return idx

    device_list = "\n".join(
        f"[{idx}] {entry.get('name', 'Unknown')}" for idx, entry in enumerate(devices)
    )
    message = (
        "Unable to use audio input device "
        f"{device!r}. Available devices:\n{device_list}\n"
        "Specify a device index or name using --device."
    )
    raise AudioCaptureError(message)


def record_audio(
    out_wav: Path,
    device: Optional[str] = None,
    max_seconds: int = 90,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """Record microphone audio to ``out_wav`` for up to ``max_seconds`` seconds.

    Parameters
    ----------
    out_wav:
        Destination path for the recorded WAV file.
    device:
        Optional sounddevice identifier. Can be a device index or partial name.
    max_seconds:
        Maximum number of seconds to record. Recording stops automatically once
        this limit is reached.
    sample_rate:
        Sampling rate for recording in Hertz.
    channels:
        Number of audio channels to record.

    Returns
    -------
    Path
        The path to the recorded WAV file.

    Raises
    ------
    AudioCaptureError
        If no audio could be captured or the device configuration fails.
    """

    if sd is None:
        raise AudioCaptureError(
            "The `sounddevice` package is required for audio recording."
        )
    if np is None:
        raise AudioCaptureError("The `numpy` package is required for audio recording.")
    if wavfile is None:
        raise AudioCaptureError("The `scipy` package is required for audio recording.")

    resolved_device = _resolve_device(device)
    duration_frames = max_seconds * sample_rate

    try:
        sd.check_input_settings(
            device=resolved_device, samplerate=sample_rate, channels=channels
        )
    except Exception as exc:  # pragma: no cover - relies on system configuration
        raise AudioCaptureError(str(exc)) from exc

    try:
        recording = sd.rec(
            frames=duration_frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=resolved_device,
        )
        sd.wait()
    except Exception as exc:  # pragma: no cover - depends on audio backend
        devices = sd.query_devices()
        device_list = "\n".join(
            f"[{idx}] {entry.get('name', 'Unknown')}" for idx, entry in enumerate(devices)
        )
        raise AudioCaptureError(
            f"Failed to record audio: {exc}\nAvailable devices:\n{device_list}"
        ) from exc

    if recording.size == 0:
        raise AudioCaptureError("No audio was captured from the microphone.")

    peak = float(np.max(np.abs(recording)))
    if peak > 1.0:
        recording = recording / peak

    pcm_audio = np.clip(recording, -1.0, 1.0)
    pcm_audio = (pcm_audio * np.iinfo(np.int16).max).astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out_wav, sample_rate, pcm_audio)
    return out_wav


def _ensure_ffmpeg_available() -> None:
    """Ensure ``ffmpeg`` is available on the system path."""

    if shutil.which("ffmpeg"):
        return
    hint = (
        "ffmpeg is required for Whisper transcription. "
        "Install it via `sudo apt-get install ffmpeg` or your distro's package manager."
    )
    raise TranscriptionError(hint)


def transcribe_local(audio_path: Path, model: str = "base", language: str = "en") -> str:
    """Transcribe ``audio_path`` using a local Whisper model."""

    _ensure_ffmpeg_available()

    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TranscriptionError(
            "The `whisper` package is not installed.", audio_path=audio_path
        ) from exc

    try:
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(str(audio_path), language=language)
    except Exception as exc:  # pragma: no cover - heavy dependency path
        raise TranscriptionError(
            f"Failed to transcribe audio: {exc}", audio_path=audio_path
        ) from exc

    text = result.get("text", "").strip()
    if not text:
        raise TranscriptionError(
            "Whisper produced an empty transcription.", audio_path=audio_path
        )
    return text


def capture_and_transcribe_local(
    vault_dir: Path,
    model: str = "base",
    device: Optional[str] = None,
    max_seconds: int = 90,
    language: str = "en",
    keep_audio: bool = True,
) -> Tuple[Path, str]:
    """Capture audio and transcribe it locally with Whisper."""

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    inbox_dir = vault_dir / "Inbox"
    audio_dir = inbox_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    inbox_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{timestamp}.wav"
    record_audio(audio_path, device=device, max_seconds=max_seconds)

    try:
        transcription = transcribe_local(audio_path, model=model, language=language)
    except TranscriptionError as exc:
        if not keep_audio:
            audio_path.unlink(missing_ok=True)
        raise TranscriptionError(str(exc), audio_path=audio_path) from exc
    except Exception as exc:
        if not keep_audio:
            audio_path.unlink(missing_ok=True)
        raise TranscriptionError(str(exc), audio_path=audio_path) from exc

    if not keep_audio:
        audio_path.unlink(missing_ok=True)

    return audio_path, transcription
