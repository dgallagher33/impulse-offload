"""Tests for the Whisper integration utilities."""
from __future__ import annotations

import types
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from impulse_offloader.utils import whisper_integration as wi


def test_record_audio_creates_wav_linux(monkeypatch, tmp_path: Path) -> None:
    frames = 16_000
    buffer = np.zeros((frames, 1), dtype="float32")

    stub_sd = types.SimpleNamespace()
    stub_sd.query_devices = lambda: [{"name": "Default"}]
    stub_sd.check_input_settings = lambda **kwargs: None

    def fake_rec(*, frames: int, samplerate: int, channels: int, dtype: str, device: object):
        assert frames == 16_000
        assert samplerate == 16_000
        assert channels == 1
        assert dtype == "float32"
        return buffer

    stub_sd.rec = lambda *args, **kwargs: fake_rec(*args, **kwargs)
    stub_sd.wait = lambda: None

    monkeypatch.setattr(wi, "sd", stub_sd)

    class FakeWavfile:
        @staticmethod
        def write(path: Path, rate: int, data: np.ndarray) -> None:
            channels = data.shape[1] if data.ndim > 1 else 1
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(rate)
                wav_file.writeframes(data.tobytes())

    monkeypatch.setattr(wi, "wavfile", FakeWavfile)

    output = tmp_path / "audio.wav"
    path = wi.record_audio(output, max_seconds=1)
    assert path == output
    assert path.exists()

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnchannels() == 1


def test_transcribe_local_missing_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    dummy_audio = tmp_path / "dummy.wav"
    dummy_audio.write_bytes(b"")

    monkeypatch.setattr(wi.shutil, "which", lambda _: None)

    with pytest.raises(wi.TranscriptionError) as exc:
        wi.transcribe_local(dummy_audio)

    assert "ffmpeg is required" in str(exc.value)


def test_capture_and_transcribe_local_paths(monkeypatch, tmp_path: Path) -> None:
    created_files: dict[str, Path] = {}

    def fake_record(out: Path, **kwargs) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"RIFF")
        created_files["audio"] = out
        return out

    def fake_transcribe(path: Path, **kwargs) -> str:
        created_files["transcribed"] = path
        return "hello world"

    monkeypatch.setattr(wi, "record_audio", fake_record)
    monkeypatch.setattr(wi, "transcribe_local", fake_transcribe)

    audio_path, text = wi.capture_and_transcribe_local(tmp_path, model="base", keep_audio=True)

    assert text == "hello world"
    assert audio_path.parent == tmp_path / "Inbox" / "audio"
    assert audio_path.name.endswith(".wav")
    assert audio_path == created_files["audio"]
    assert created_files["transcribed"] == audio_path
